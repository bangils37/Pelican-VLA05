#!/usr/bin/env python
"""Training script for Pelican-VLA 0.5.

Supports:
- Multi-GPU distributed training (DDP via torchrun) or Single GPU
- bfloat16 mixed precision
- Freezing backbone (action head + bottleneck fine-tuning) or full fine-tuning
- LeRobot v2.1/v3 formatted datasets
- Checkpointing compatible with Pelican-VLA inference
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure src/ packages are available
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "pelican_vla0.5_infer" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lerobot.policies.pelicanvla05 import PelicanVLA05Policy, PelicanVLA05Config
from training.dataset import LeRobotPelicanDataset, collate_pelican_batch


def setup_distributed():
    """Check if launched via torchrun and initialize process group."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return rank, world_size, local_rank, True
    else:
        device = 0 if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return 0, 1, 0, False


def parse_args():
    parser = argparse.ArgumentParser(description="Train / Finetune Pelican-VLA 0.5")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/anhnb9/Documents/datasets/astri_making_coffee_v21",
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=str(_ROOT / "pretrained_model" / "pelican_vla05"),
        help="Path to pretrained Pelican-VLA checkpoint",
    )
    parser.add_argument(
        "--cosmos_tokenizer_path",
        type=str,
        default=str(_ROOT / "pretrained_model" / "cosmos_tokenizer"),
        help="Path to Cosmos Tokenizer jit assets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_ROOT / "checkpoints" / "pelican_run"),
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--qwen3_vl_path",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Local dir or HF id for Qwen3-VL weights",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log_steps", type=int, default=10, help="Log metrics every N steps")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze Qwen3-VL backbone (fine-tune heads only)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="pelican-vla-training", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    return parser.parse_args()


def save_checkpoint(policy, output_dir: Path, step: int, dataset: LeRobotPelicanDataset, is_best: bool = False):
    """Save checkpoint in native Pelican-VLA format."""
    save_dir = output_dir / f"checkpoint-{step:06d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = policy.module if isinstance(policy, DDP) else policy
    unwrapped.save_pretrained(str(save_dir))

    # Save dataset stats for inference engine
    stats = {}
    if dataset.state_mean is not None and dataset.action_mean is not None:
        stats = {
            "state": {
                "mean": dataset.state_mean.tolist(),
                "std": dataset.state_std.tolist(),
            },
            "action": {
                "mean": dataset.action_mean.tolist(),
                "std": dataset.action_std.tolist(),
            },
        }
        with open(save_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    logging.info(f"Saved checkpoint to {save_dir}")

    if is_best:
        best_dir = output_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(str(best_dir))
        if stats:
            with open(best_dir / "stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        logging.info(f"Updated best model at {best_dir}")


def main():
    args = parse_args()
    rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main_process = (rank == 0)

    # Configure logging
    logging.basicConfig(
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO if is_main_process else logging.WARNING,
    )

    if args.cosmos_tokenizer_path:
        os.environ["COSMOS_TOKENIZER_PATH"] = args.cosmos_tokenizer_path
    if args.qwen3_vl_path:
        os.environ["QWEN3_VL_PATH"] = args.qwen3_vl_path

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & DataLoader
    camera_map = {
        "cam_head": "image0",
        "cam_left_wrist": "image1",
        "cam_right_wrist": "image2",
    }
    dataset = LeRobotPelicanDataset(
        dataset_root=args.dataset_path,
        camera_map=camera_map,
        chunk_size=50,
        future_horizon=25,
        qwen3_vl_path=args.qwen3_vl_path,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_pelican_batch,
        pin_memory=True,
        drop_last=True,
    )

    if is_main_process:
        logging.info(f"Loaded dataset from {args.dataset_path} with {len(dataset)} samples.")
        logging.info(f"World size: {world_size}, Batch size per GPU: {args.batch_size}, Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")

    # 2. Model
    logging.info(f"Loading pretrained policy from {args.pretrained_model_path}...")
    policy = PelicanVLA05Policy.from_pretrained(args.pretrained_model_path, strict=True)

    if args.freeze_backbone:
        logging.info("Freezing Qwen3-VL backbone. Fine-tuning action heads and bottleneck tokens only.")
        policy.model.config.freeze_backbone = True
        policy.model.set_requires_grad()

    policy.to(device)

    if is_distributed:
        policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 3. Optimizer & Scheduler
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    if is_main_process:
        num_trainable = sum(p.numel() for p in trainable_params)
        num_total = sum(p.numel() for p in policy.parameters())
        logging.info(f"Trainable parameters: {num_trainable:,} / {num_total:,} ({100 * num_trainable / num_total:.2f}%)")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. Logger
    tb_writer = None
    if is_main_process:
        tb_writer = SummaryWriter(log_dir=str(output_dir / "logs"))
        if args.use_wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )

    # 5. Training loop
    global_step = 0
    best_loss = float("inf")
    policy.train()

    pbar = tqdm(total=args.max_steps, desc="Training Pelican-VLA", disable=not is_main_process)
    data_iter = iter(dataloader)

    while global_step < args.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_loss_dict = {}

        for accum_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(global_step)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass with bfloat16 autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, loss_dict = policy(batch, current_step=global_step)
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            accum_loss += loss.item()
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    accum_loss_dict[k] = accum_loss_dict.get(k, 0.0) + (v / args.gradient_accumulation_steps)

        # Gradient clipping & step
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        global_step += 1
        pbar.update(1)

        # Logging
        if global_step % args.log_steps == 0 and is_main_process:
            current_lr = scheduler.get_last_lr()[0]
            log_str = f"Step {global_step}/{args.max_steps} | Loss: {accum_loss:.4f} | LR: {current_lr:.2e}"
            if "loss_action" in accum_loss_dict:
                log_str += f" | Act: {accum_loss_dict['loss_action']:.4f}"
            if "loss_gen" in accum_loss_dict:
                log_str += f" | Gen: {accum_loss_dict['loss_gen']:.4f}"
            pbar.set_postfix_str(log_str)

            if tb_writer:
                tb_writer.add_scalar("train/loss", accum_loss, global_step)
                tb_writer.add_scalar("train/lr", current_lr, global_step)
                for k, v in accum_loss_dict.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(f"train/{k}", v, global_step)

            if args.use_wandb:
                import wandb
                wandb.log({"train/loss": accum_loss, "train/lr": current_lr, **{f"train/{k}": v for k, v in accum_loss_dict.items()}}, step=global_step)

        # Checkpointing
        if global_step % args.save_steps == 0 and is_main_process:
            is_best = accum_loss < best_loss
            if is_best:
                best_loss = accum_loss
            save_checkpoint(policy, output_dir, global_step, dataset, is_best=is_best)

    # Final save
    if is_main_process:
        save_checkpoint(policy, output_dir, global_step, dataset, is_best=(accum_loss < best_loss))
        logging.info("Training completed successfully!")
        if tb_writer:
            tb_writer.close()
        if args.use_wandb:
            import wandb
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

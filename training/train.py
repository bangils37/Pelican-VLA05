#!/usr/bin/env python
"""Training script for Pelican-VLA 0.5.

Supports:
- Multi-GPU distributed training (DDP via torchrun) or Single GPU
- bfloat16 mixed precision
- Freezing backbone (action head + bottleneck fine-tuning) or full fine-tuning
- LeRobot v2.1/v3 formatted datasets
- Formal episode-level Train/Validation splitting and validation loss evaluation
- Checkpointing compatible with Pelican-VLA inference (including best_model tracking)
- TensorBoard and Weights & Biases (W&B) logging
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure src/ packages are available
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "pelican_vla0.5_infer" / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
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
        default="/home/aitt/data/serving_brewed_coffee_lrb_annotated",
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
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers per GPU")
    parser.add_argument("--max_video_readers", type=int, default=6, help="Max open VideoReader instances per DataLoader worker")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--max_steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--log_steps", type=int, default=10, help="Log metrics every N steps")
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluate validation loss every N steps")
    parser.add_argument("--eval_batches", type=int, default=50, help="Number of val batches to evaluate per eval")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Ratio of held-out episodes for validation")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze Qwen3-VL backbone (fine-tune heads only)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="astribot_making_coffee", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    return parser.parse_args()


def save_checkpoint(policy, output_dir: Path, step: int, dataset: LeRobotPelicanDataset, is_best: bool = False):
    """Save checkpoint in native Pelican-VLA format."""
    unwrapped = policy.module if isinstance(policy, DDP) else policy

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

    if not is_best:
        save_dir = output_dir / f"checkpoint-{step:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(str(save_dir))
        if stats:
            with open(save_dir / "stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        logging.info(f"Saved regular checkpoint to {save_dir}")
    else:
        best_dir = output_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(str(best_dir))
        if stats:
            with open(best_dir / "stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        logging.info(f"Updated best model at {best_dir}")


def compute_batch_action_mse(policy, batch, action_mean=None, action_std=None):
    """Compute Action MSE Error between predicted trajectory and ground truth actions."""
    raw_policy = policy.module if isinstance(policy, DDP) else policy
    orig_mode = raw_policy.training
    raw_policy.eval()
    try:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_actions, _ = raw_policy.predict_action_chunk(batch)
            orig_dim = raw_policy.config.output_features["action"].shape[0]
            gt_actions = batch["action"][:, :, :orig_dim]

            if action_mean is not None and action_std is not None:
                mean_t = torch.as_tensor(action_mean, device=pred_actions.device, dtype=pred_actions.dtype)[:orig_dim]
                std_t = torch.as_tensor(action_std, device=pred_actions.device, dtype=pred_actions.dtype)[:orig_dim]
                unnorm_pred = pred_actions * std_t + mean_t
                unnorm_gt = gt_actions * std_t + mean_t
                mse = F.mse_loss(unnorm_pred, unnorm_gt).item()
            else:
                mse = F.mse_loss(pred_actions, gt_actions).item()
            return mse
    except Exception as e:
        logging.warning(f"Failed to compute action MSE error: {e}")
        return 0.0
    finally:
        if orig_mode:
            raw_policy.train()


def evaluate(
    policy,
    val_dataloader,
    device,
    max_batches=50,
    is_distributed=False,
    world_size=1,
    current_step=10000,
    action_mean=None,
    action_std=None,
):
    """Run validation evaluation loop.
    
    Args:
        current_step: Passed to policy.forward() to correctly compute bottleneck warmup;
                      use a large value (default 10000) so warmup is fully active during eval.
    """
    policy.eval()
    val_loss_accum = 0.0
    val_loss_dict = {}
    batches_run = 0
    val_action_mses = []

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if max_batches > 0 and i >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, loss_dict = policy(batch, current_step=current_step)

            val_loss_accum += loss.item()
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    val_loss_dict[k] = val_loss_dict.get(k, 0.0) + v

            # Compute action MSE on first 3 val batches
            if i < 3:
                mse = compute_batch_action_mse(policy, batch, action_mean, action_std)
                val_action_mses.append(mse)

            batches_run += 1

    if batches_run > 0:
        val_loss_accum /= batches_run
        for k in val_loss_dict:
            val_loss_dict[k] /= batches_run

    val_action_mse = float(np.mean(val_action_mses)) if val_action_mses else 0.0

    # In distributed mode: sync val loss AND every sub-loss across all GPUs
    if is_distributed:
        # Sync main loss and action MSE
        loss_tensor = torch.tensor([val_loss_accum, val_action_mse], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        val_loss_accum = loss_tensor[0].item() / world_size
        val_action_mse = loss_tensor[1].item() / world_size

        # Sync each sub-loss in val_loss_dict
        keys = sorted(val_loss_dict.keys())
        if keys:
            values_tensor = torch.tensor([val_loss_dict[k] for k in keys], device=device)
            dist.all_reduce(values_tensor, op=dist.ReduceOp.SUM)
            for i, k in enumerate(keys):
                val_loss_dict[k] = values_tensor[i].item() / world_size

    policy.train()
    return val_loss_accum, val_loss_dict, val_action_mse


def main():
    args = parse_args()
    rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main_process = (rank == 0)

    # Configure logging
    logging.basicConfig(
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if is_main_process else logging.WARN,
    )

    if args.cosmos_tokenizer_path:
        os.environ["COSMOS_TOKENIZER_PATH"] = args.cosmos_tokenizer_path
    if args.qwen3_vl_path:
        os.environ["QWEN3_VL_PATH"] = args.qwen3_vl_path

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & DataLoader (Train & Validation Splits)
    camera_map = {
        "cam_head": "image0",
        "cam_left_wrist": "image1",
        "cam_right_wrist": "image2",
    }
    
    if is_main_process:
        logging.info(f"Loading datasets with {args.val_ratio * 100:.1f}% validation split...")

    train_dataset = LeRobotPelicanDataset(
        dataset_root=args.dataset_path,
        camera_map=camera_map,
        chunk_size=50,
        future_horizon=15,
        qwen3_vl_path=args.qwen3_vl_path,
        split="train",
        val_ratio=args.val_ratio,
        max_video_readers=args.max_video_readers,
    )
    val_dataset = LeRobotPelicanDataset(
        dataset_root=args.dataset_path,
        camera_map=camera_map,
        chunk_size=50,
        future_horizon=15,
        qwen3_vl_path=args.qwen3_vl_path,
        split="val",
        val_ratio=args.val_ratio,
        max_video_readers=args.max_video_readers,
        _shared_data=train_dataset.get_shared_data(),
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_pelican_batch,
        pin_memory=True,
        drop_last=True,
    )

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, args.num_workers // 2),
        collate_fn=collate_pelican_batch,
        pin_memory=True,
        drop_last=False,
    )

    if is_main_process:
        logging.info(f"Train samples: {len(train_dataset)} ({len(train_dataset.active_episodes_set)} episodes)")
        logging.info(f"Val samples:   {len(val_dataset)} ({len(val_dataset.active_episodes_set)} episodes)")
        logging.info(f"World size: {world_size}, Batch size per GPU: {args.batch_size}, Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")

    # 2. Model
    logging.info(f"Loading pretrained policy from {args.pretrained_model_path}...")
    policy = PelicanVLA05Policy.from_pretrained(args.pretrained_model_path, strict=True)

    if args.freeze_backbone:
        logging.info("Freezing Qwen3-VL backbone. Fine-tuning action heads and bottleneck tokens only.")
        policy.model.config.freeze_backbone = True
        policy.model.set_requires_grad()
    else:
        logging.info("Configuring FULL fine-tuning mode (Qwen3-VL backbone + Action heads + Bottleneck tokens).")
        policy.model.config.freeze_backbone = False
        policy.model.gradient_checkpointing_enable()

    # Update output features to match Astribot's 16 active joints for action loss
    policy.config.output_features["action"].shape = [16]
    policy.config.input_features["observation.state"].shape = [16]

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
    best_val_loss = float("inf")
    policy.train()

    pbar = tqdm(total=args.max_steps, desc="Training Pelican-VLA", disable=not is_main_process)
    data_iter = iter(train_dataloader)

    while global_step < args.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_loss_dict = {}

        for accum_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                if train_sampler is not None:
                    train_sampler.set_epoch(global_step)
                data_iter = iter(train_dataloader)
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

        # Logging (Train metrics)
        if global_step % args.log_steps == 0 and is_main_process:
            current_lr = scheduler.get_last_lr()[0]
            current_epoch = round((global_step * world_size * args.batch_size) / len(train_dataset), 2)
            log_str = f"Step {global_step}/{args.max_steps} (Epoch {current_epoch}) | Loss: {accum_loss:.4f} | LR: {current_lr:.2e}"
            if "loss_action" in accum_loss_dict:
                log_str += f" | Act: {accum_loss_dict['loss_action']:.4f}"
            if "loss_gen" in accum_loss_dict:
                log_str += f" | Gen: {accum_loss_dict['loss_gen']:.4f}"
            pbar.set_postfix_str(log_str)

            if tb_writer:
                tb_writer.add_scalar("train_loss", accum_loss, global_step)
                tb_writer.add_scalar("lr", current_lr, global_step)
                if "loss_action" in accum_loss_dict:
                    tb_writer.add_scalar("loss_action", accum_loss_dict["loss_action"], global_step)
                if "loss_gen" in accum_loss_dict:
                    tb_writer.add_scalar("loss_gen", accum_loss_dict["loss_gen"], global_step)

            if args.use_wandb:
                import wandb
                # Diffusion-Policy-compatible format: clean top-level metrics, no individual dim0..13 clutter
                wandb_payload = {
                    "train_loss": accum_loss,
                    "lr": current_lr,
                    "epoch": current_epoch,
                    "global_step": global_step,
                }
                if "loss_action" in accum_loss_dict:
                    wandb_payload["loss_action"] = accum_loss_dict["loss_action"]
                if "loss_gen" in accum_loss_dict:
                    wandb_payload["loss_gen"] = accum_loss_dict["loss_gen"]
                wandb.log(wandb_payload, step=global_step)

        # Validation Loss Evaluation
        if (global_step % args.eval_steps == 0 or global_step == 1) and len(val_dataset) > 0:
            val_loss, val_loss_dict, val_action_mse = evaluate(
                policy,
                val_dataloader,
                device,
                max_batches=args.eval_batches,
                is_distributed=is_distributed,
                world_size=world_size,
                current_step=global_step,
                action_mean=train_dataset.action_mean,
                action_std=train_dataset.action_std,
            )

            # Compute train_action_mse_error on current training batch
            train_action_mse = compute_batch_action_mse(
                policy,
                batch,
                action_mean=train_dataset.action_mean,
                action_std=train_dataset.action_std,
            )

            if is_main_process:
                current_epoch = round((global_step * world_size * args.batch_size) / len(train_dataset), 2)
                val_log_str = f" >>> EVAL [Step {global_step}] Val Loss: {val_loss:.4f} | Train Act MSE: {train_action_mse:.4f} | Val Act MSE: {val_action_mse:.4f}"
                if "loss_action" in val_loss_dict:
                    val_log_str += f" | Val Act: {val_loss_dict['loss_action']:.4f}"
                if "loss_gen" in val_loss_dict:
                    val_log_str += f" | Val Gen: {val_loss_dict['loss_gen']:.4f}"
                logging.info(val_log_str)

                if tb_writer:
                    tb_writer.add_scalar("val_loss", val_loss, global_step)
                    tb_writer.add_scalar("train_action_mse_error", train_action_mse, global_step)
                    tb_writer.add_scalar("val_action_mse_error", val_action_mse, global_step)

                if args.use_wandb:
                    import wandb
                    eval_payload = {
                        "val_loss": val_loss,
                        "train_action_mse_error": train_action_mse,
                        "val_action_mse_error": val_action_mse,
                        "epoch": current_epoch,
                        "global_step": global_step,
                    }
                    if "loss_action" in val_loss_dict:
                        eval_payload["val_loss_action"] = val_loss_dict["loss_action"]
                    if "loss_gen" in val_loss_dict:
                        eval_payload["val_loss_gen"] = val_loss_dict["loss_gen"]
                    wandb.log(eval_payload, step=global_step)

                # Check and save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logging.info(f"New BEST validation loss: {best_val_loss:.4f}! Saving best_model...")
                    save_checkpoint(policy, output_dir, global_step, train_dataset, is_best=True)

        # Regular Checkpointing
        if global_step % args.save_steps == 0 and is_main_process:
            save_checkpoint(policy, output_dir, global_step, train_dataset, is_best=False)

    # Final save
    if is_main_process:
        save_checkpoint(policy, output_dir, global_step, train_dataset, is_best=False)
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

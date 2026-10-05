"""End-to-end smoke test verifying data loading, forward, backward, optimizer step, and checkpointing for Full Fine-Tuning."""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "pelican_vla0.5_infer" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lerobot.policies.pelicanvla05 import PelicanVLA05Policy
from training.dataset import LeRobotPelicanDataset, collate_pelican_batch

os.environ["COSMOS_TOKENIZER_PATH"] = str(_ROOT / "pretrained_model" / "cosmos_tokenizer")
os.environ["QWEN3_VL_PATH"] = "Qwen/Qwen3-VL-4B-Instruct"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Pick GPU 1 or 2 or 3
gpu_id = 1 if torch.cuda.device_count() > 1 else 0
if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Running smoke test on device: {device} (Total free VRAM: {torch.cuda.mem_get_info(gpu_id)[0] // (1024**3)} GB)")

# 1. Dataset test
dataset_path = "/home/aitt/data/serving_brewed_coffee_lrb_annotated"
camera_map = {
    "cam_head": "image0",
    "cam_left_wrist": "image1",
    "cam_right_wrist": "image2",
}

print(f"\n1. Loading LeRobot dataset from {dataset_path}...")
dataset = LeRobotPelicanDataset(
    dataset_root=dataset_path,
    camera_map=camera_map,
    chunk_size=50,
    future_horizon=15,
)
print(f"Dataset successfully loaded: {len(dataset)} total frames.")

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_pelican_batch,
)

batch = next(iter(dataloader))
print("\nBatch loaded successfully from real dataset:")
print(f"  state shape:        {batch['observation.state'].shape}")
print(f"  action shape:       {batch['action'].shape}")
print(f"  image0 shape:       {batch['observation.images.image0'].shape}")
print(f"  pixel_values shape: {batch['observation.pixel_values'].shape}")
print(f"  input_ids shape:    {batch['observation.input_ids'].shape}")

# 2. Model test with Gradient Checkpointing in FULL FINE-TUNING mode
model_path = _ROOT / "pretrained_model" / "pelican_vla05"
print(f"\n2. Loading pretrained Pelican-VLA 0.5 model from {model_path}...")
policy = PelicanVLA05Policy.from_pretrained(str(model_path), strict=True)
policy.model.config.freeze_backbone = False
policy.model.gradient_checkpointing_enable()

# Configure Astribot 16-dim action/state
policy.config.output_features["action"].shape = [16]
policy.config.input_features["observation.state"].shape = [16]

policy.to(device)
policy.train()

trainable_params = [p for p in policy.parameters() if p.requires_grad]
print(f"Trainable parameters in FULL fine-tune mode: {sum(p.numel() for p in trainable_params):,}")

optimizer = torch.optim.AdamW(trainable_params, lr=2.5e-5)

# Move batch to GPU
batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# 3. Two-step training test
print("\n3. Executing Step 1: Forward + Loss Calculation + Backward + Optimizer Step...")
optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss1, loss_dict1 = policy(batch, current_step=0)
loss1.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
optimizer.step()

print(f"Step 1 Success! Total Loss: {loss1.item():.4f} (Grad Norm: {grad_norm:.4f})")
print("Loss breakdown:")
for k, v in loss_dict1.items():
    if isinstance(v, float) and "loss_action_dim" not in k:
        print(f"  {k}: {v:.4f}")

print("\nExecuting Step 2: Forward + Loss Calculation + Backward + Optimizer Step...")
optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss2, loss_dict2 = policy(batch, current_step=1)
loss2.backward()
torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
optimizer.step()
print(f"Step 2 Success! Total Loss: {loss2.item():.4f}")

print("\n" + "=" * 65)
print("ALL FULL FINE-TUNING SMOKE TESTS PASSED!")
print("Pelican-VLA 0.5 pipeline is fully configured and ready for training!")
print("=" * 65)

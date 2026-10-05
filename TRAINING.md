# Pelican-VLA 0.5 Training & Fine-Tuning Guide

This guide details how to train and fine-tune **Pelican-VLA 0.5** on robotic manipulation datasets (such as LeRobot v2.1/v3 formatted datasets).

---

## 1. Environment & Setup

A dedicated conda environment `pelican_vla` is pre-configured with CUDA 13.0, PyTorch 2.14, and Transformers 5.18.

```bash
# Activate environment
conda activate pelican_vla

# Dynamic library path for CXXABI and CUDA
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export COSMOS_TOKENIZER_PATH="/home/anhnb9/Documents/Pelican-VLA05/pretrained_model/cosmos_tokenizer"
export QWEN3_VL_PATH="Qwen/Qwen3-VL-4B-Instruct"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

---

## 2. Directory Structure

```text
Pelican-VLA05/
├── pretrained_model/
│   ├── pelican_vla05/               # Pelican-VLA 0.5 checkpoint (model.safetensors, config.json, stats.json)
│   └── cosmos_tokenizer/           # Cosmos Discrete VAE JIT models (encoder.jit, decoder.jit)
├── training/
│   ├── dataset.py                  # LeRobot v2.1/v3 temporal dataset loader & collate function
│   └── train.py                    # Multi-GPU (DDP) and single-GPU training engine
├── scripts/
│   └── train_astri_coffee.sh       # Executable launcher for training runs
├── smoke_test_train.py             # End-to-end verification script (forward, backward, optimizer, save)
└── checkpoints/                    # Saved fine-tuned checkpoints and split weights
```

---

## 3. Launching Training

### Single GPU Training (e.g., GPU 3)
```bash
NUM_GPUS=1 GPU_ID=3 BATCH_SIZE=2 bash scripts/train_astri_coffee.sh
```

### Multi-GPU Distributed Training (All 4 GPUs via `torchrun`)
```bash
NUM_GPUS=4 BATCH_SIZE=4 LR=2.5e-5 bash scripts/train_astri_coffee.sh
```

### Custom Dataset Training
To train on another dataset, provide custom arguments to `training/train.py`:
```bash
python training/train.py \
    --dataset_path "/path/to/lerobot/dataset" \
    --pretrained_model_path "./pretrained_model/pelican_vla05" \
    --output_dir "./checkpoints/custom_experiment" \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2.5e-5 \
    --max_steps 10000 \
    --save_steps 1000 \
    --freeze_backbone \
    --gradient_checkpointing
```

---

## 4. Key Training Features

- **Training Modes:**
  - **Head & Bottleneck Fine-Tuning (`--freeze_backbone`):** Trains only the Action Head MLP, Gen Head, Slot Bottleneck, and Task projection heads (~77.4M trainable parameters). Fast, memory-efficient, ideal for new robot embodiments.
  - **Full-Model Fine-Tuning (without `--freeze_backbone`):** Trains entire Qwen3-VL 4B backbone + heads with AdamW and cosine schedule.
- **Dataset Temporal Alignment:**
  - Automatically loads 3 temporal frames: $t-1$ (previous), $t$ (current), and $t_{\text{future}}$ (future for visual generation prediction loss).
  - Automatically handles action chunking ($K=50$ steps) and robot state vector normalization.
- **Checkpoint Outputs:**
  - Saved in Hugging Face / LeRobot format containing `model.safetensors` as well as modular split files: `backbone.safetensors`, `action_head.safetensors`, `gen_head.safetensors`, and `task_head.safetensors`.

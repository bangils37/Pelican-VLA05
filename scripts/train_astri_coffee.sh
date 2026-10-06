#!/usr/bin/env bash
set -e

# Pelican-VLA 0.5 Training Launcher for Astribot Coffee Dataset (Full Fine-Tuning)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONDA_ENV_PYTHON="/home/anhnb9/miniconda3/envs/pelican_vla/bin/python"

# Environment Variables
export COSMOS_TOKENIZER_PATH="$PROJECT_ROOT/pretrained_model/cosmos_tokenizer"
export QWEN3_VL_PATH="Qwen/Qwen3-VL-4B-Instruct"
export LD_LIBRARY_PATH="/home/anhnb9/miniconda3/envs/pelican_vla/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Configuration
DATASET_PATH="/home/aitt/data/serving_brewed_coffee_lrb_annotated"
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/pelican_astri_coffee_full_ft"

# Pretrained model / Checkpoint to continue from
if [ -d "$OUTPUT_DIR/best_model" ]; then
    DEFAULT_PRETRAINED="$OUTPUT_DIR/best_model"
else
    DEFAULT_PRETRAINED="$PROJECT_ROOT/pretrained_model/pelican_vla05"
fi
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"$DEFAULT_PRETRAINED"}

# GPU Isolation (Default to GPUs 1,2,3 to preserve GPU 0 for Desktop/Display)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3"}

# If NUM_GPUS is not set, count number of comma-separated GPUs in CUDA_VISIBLE_DEVICES
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^$' | wc -l)
fi

# Training Hyperparameters
GPU_ID=${GPU_ID:-0}              # Used when NUM_GPUS=1
BATCH_SIZE=${BATCH_SIZE:-4}      # Per GPU batch size
NUM_WORKERS=${NUM_WORKERS:-2}    # DataLoader workers per GPU (memory-safe)
MAX_VIDEO_READERS=${MAX_VIDEO_READERS:-6} # Max open VideoReaders per worker
MAX_STEPS=${MAX_STEPS:-300000}    # Total training steps (default 30k)
LR=${LR:-2.5e-5}                 # Peak learning rate
WARMUP_STEPS=${WARMUP_STEPS:-500}
SAVE_STEPS=${SAVE_STEPS:-250}   # Save checkpoint every N steps
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1} # Keep only last K regular checkpoints to avoid filling disk
LOG_STEPS=${LOG_STEPS:-10}

# Validation Hyperparameters
VAL_RATIO=${VAL_RATIO:-0.05}     # 5% held-out episodes for validation (~20 episodes)
EVAL_STEPS=${EVAL_STEPS:-250}    # Evaluate val loss every 250 steps
EVAL_BATCHES=${EVAL_BATCHES:-50} # Batches to evaluate per validation run

# Weights & Biases (W&B) Logging
USE_WANDB=${USE_WANDB:-1}        # 1 to enable W&B, 0 to disable
WANDB_PROJECT=${WANDB_PROJECT:-"astribot_making_coffee"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"pelican_vla05_continue_$(date +%Y%m%d_%H%M%S)"}

WANDB_ARGS=""
if [ "$USE_WANDB" -eq 1 ]; then
    WANDB_ARGS="--use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name $WANDB_RUN_NAME"
fi

echo "=========================================================="
echo "Starting Pelican-VLA 0.5 FULL FINE-TUNING (Continue/Resume)"
echo "Project Root:         $PROJECT_ROOT"
echo "Dataset Path:         $DATASET_PATH"
echo "Model Weights:        $PRETRAINED_MODEL"
echo "Output Directory:     $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs:       $NUM_GPUS"
echo "Workers / GPU:        $NUM_WORKERS (Max Video Readers: $MAX_VIDEO_READERS)"
echo "Batch Size / GPU:     $BATCH_SIZE"
echo "Max Steps:            $MAX_STEPS"
echo "Save Steps:           $SAVE_STEPS (Keep max $SAVE_TOTAL_LIMIT checkpoints)"
echo "Learning Rate:        $LR"
echo "Validation Split:     ${VAL_RATIO} (Eval every ${EVAL_STEPS} steps, ${EVAL_BATCHES} batches)"
echo "W&B Tracking:         Enabled ($WANDB_PROJECT / $WANDB_RUN_NAME)"
echo "=========================================================="

mkdir -p "$OUTPUT_DIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running Multi-GPU Distributed Full Fine-Tuning via torchrun ($NUM_GPUS GPUs)..."
    /home/anhnb9/miniconda3/envs/pelican_vla/bin/torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        "$PROJECT_ROOT/training/train.py" \
        --dataset_path "$DATASET_PATH" \
        --pretrained_model_path "$PRETRAINED_MODEL" \
        --cosmos_tokenizer_path "$COSMOS_TOKENIZER_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --max_video_readers "$MAX_VIDEO_READERS" \
        --learning_rate "$LR" \
        --warmup_steps "$WARMUP_STEPS" \
        --max_steps "$MAX_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --save_total_limit "$SAVE_TOTAL_LIMIT" \
        --log_steps "$LOG_STEPS" \
        --val_ratio "$VAL_RATIO" \
        --eval_steps "$EVAL_STEPS" \
        --eval_batches "$EVAL_BATCHES" \
        $WANDB_ARGS
else
    echo "Running Single-GPU Full Fine-Tuning on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID "$CONDA_ENV_PYTHON" "$PROJECT_ROOT/training/train.py" \
        --dataset_path "$DATASET_PATH" \
        --pretrained_model_path "$PRETRAINED_MODEL" \
        --cosmos_tokenizer_path "$COSMOS_TOKENIZER_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --max_video_readers "$MAX_VIDEO_READERS" \
        --learning_rate "$LR" \
        --warmup_steps "$WARMUP_STEPS" \
        --max_steps "$MAX_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --save_total_limit "$SAVE_TOTAL_LIMIT" \
        --log_steps "$LOG_STEPS" \
        --val_ratio "$VAL_RATIO" \
        --eval_steps "$EVAL_STEPS" \
        --eval_batches "$EVAL_BATCHES" \
        $WANDB_ARGS
fi

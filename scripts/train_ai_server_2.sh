#!/usr/bin/env bash
# Launch from /nvme with the isolated training environment. Usage: bash ... [smoke|train]
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=${1:-train}
case "$MODE" in smoke|train) ;; *) echo 'Expected smoke or train' >&2; exit 2;; esac
export CONDA_ENV_PYTHON=${CONDA_ENV_PYTHON:-/nvme/anhnb9/envs/pelican_training/bin/python}
export HF_HOME=/nvme/anhnb9/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NUM_GPUS=2 MASTER_PORT=${MASTER_PORT:-29651}
export DATASET_PATH=/nvme/anhnb9/datasets/serving_brewed_coffee_clean_374_episodes
export PRETRAINED_MODEL="$PROJECT_ROOT/pretrained_model/pelican_vla05"
export OUTPUT_DIR=${OUTPUT_DIR:-/nvme/anhnb9/runs/coffee_aug_20261020_2gpu/$MODE}
export START_STEP=0 USE_WANDB=0
export BATCH_SIZE=${BATCH_SIZE:-4} NUM_WORKERS=${NUM_WORKERS:-2} MAX_VIDEO_READERS=6
export INSTRUCTION_AUGMENTATION=1 AUGMENTATION_PROB=0.5 USE_SUBTASKS_PROB=0.5
export EVAL_INSTRUCTION_LEVEL=high_level SUBTASK_BOUNDARY_POLICY=keep SEED=42
export LR=2.5e-5 VAL_RATIO=0.05 SAVE_TOTAL_LIMIT=2
export OMP_NUM_THREADS=2
if [ "$MODE" = smoke ]; then
    export MAX_STEPS=2 WARMUP_STEPS=1 SAVE_STEPS=2 EVAL_STEPS=2 EVAL_BATCHES=2 LOG_STEPS=1
else
    export MAX_STEPS=${MAX_STEPS:-100000} WARMUP_STEPS=500 SAVE_STEPS=250 EVAL_STEPS=250 EVAL_BATCHES=50 LOG_STEPS=10
fi
if [ -e "$OUTPUT_DIR/instruction_augmentation.json" ]; then
    echo "Run directory already initialized: $OUTPUT_DIR. Use a new OUTPUT_DIR to avoid overwriting a run." >&2
    exit 2
fi
exec bash "$PROJECT_ROOT/scripts/train_astri_coffee.sh"

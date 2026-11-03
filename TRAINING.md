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

## Dynamic coffee instruction augmentation

The coffee launcher now defaults to
`/home/aitt/data/serving_brewed_coffee_clean_374_episodes`, enables instruction
augmentation, and uses the separate output directory `checkpoints/pelican_coffee_augmented`.
It does not reuse the previous W&B run ID. Override `DATASET_PATH`, `OUTPUT_DIR`,
and `WANDB_RUN_ID` explicitly when needed. Launching the script starts training;
the smoke command below only reads data and processes a few samples.

```bash
# Validate real images, tokenization, splits and two-worker sampling without a policy model.
/home/anhnb9/miniconda3/envs/pelican_vla/bin/python scripts/smoke_instruction_augmentation.py

# Train with the curated pool and 50% canonical instructions within each instruction level.
AUGMENTATION_PROB=0.5 USE_SUBTASKS_PROB=0.5 bash scripts/train_astri_coffee.sh

# Baseline ablation: use a separate output directory to avoid resuming an augmented run.
INSTRUCTION_AUGMENTATION=0 OUTPUT_DIR="$PWD/checkpoints/coffee_baseline" bash scripts/train_astri_coffee.sh
```

Direct `training/train.py` usage keeps augmentation opt-in via
`--instruction_augmentation`. Options:

| CLI | Launcher variable | Default | Meaning |
| --- | --- | --- | --- |
| `--instruction_pool` | `INSTRUCTION_POOL` | bundled JSON | Reviewed pool, loaded once at initialization |
| `--augmentation_prob` | `AUGMENTATION_PROB` | 0.5 | Probability of a noncanonical paraphrase after selecting the instruction level |
| `--use_subtasks_prob` | `USE_SUBTASKS_PROB` | 0.5 | Training probability of using the current subtask when available |
| `--eval_instruction_level` | `EVAL_INSTRUCTION_LEVEL` | high_level | Fixed canonical high-level instruction, or canonical subtask with high-level fallback |
| `--subtask_boundary_policy` | `SUBTASK_BOUNDARY_POLICY` | keep | Keep the starting subtask; `high_level` uses the high-level instruction whenever any label in the action chunk differs |
| `--seed` | `SEED` | 42 | Episode split, sampler and worker seed |

The bundled pool has 7 high-level and 53 subtask instructions. It preserves a shot
of coffee, lifting/placing, the original return location, and the other side of the
tray without inventing a direction or slot. Semantic equivalence still requires
review against demonstrations for any new candidates. This does not augment
visual observations or robot trajectories.

Validation is now deterministic in its **instruction selection**, including with
augmentation disabled. It does not promise deterministic model loss. `all` also
uses canonical instructions. Compare runs with the same validation mode; older
runs randomly mixed high-level/subtask text. The optional boundary policy does not
change action targets. Keep it identical between baseline and augmented ablations.

Each output directory and regular/best checkpoint contains the exact normalized
`coffee_instruction_pool.json` (when enabled) and `instruction_augmentation.json`
with its SHA-256, configuration, dataset path, and episode IDs. To continue using
that pool, pass `--instruction_pool /path/to/checkpoint/coffee_instruction_pool.json`
and restore the recorded settings. Configuration is not automatically restored
from policy checkpoints. Seeds reproduce sampling with the same worker count,
rank layout and execution order; exact mid-epoch RNG/optimizer resume is not
implemented by this feature.

Pool schema errors, mismatched dataset labels, unknown label IDs and instructions
that exceed the tokenizer's configured limit fail explicitly. The runtime module
is self-contained inside `training/`, with no dependency on the external workspace
and no API calls. To update the bundled pool after reviewing external candidates,
copy the reviewed JSON into `training/coffee_instruction_pool.json`, or select it
with `--instruction_pool`. Commit that pool with the experiment code.

```bash
python -m unittest discover -s tests -p test_instruction_augmentation.py -v
```

### Regression and debug suite

Run all augmentation checks with one command (CPU only, no training job or API calls):

```bash
/home/anhnb9/miniconda3/envs/pelican_vla/bin/python scripts/test_instruction_augmentation.py \
  --real-data \
  --external-src /home/anhnb9/Documents/augment_coffee_data/src \
  --report-dir /tmp/pelican-augmentation-tests
```

Omit `--real-data` to run without the real dataset or cached tokenizer. Omit
`--external-src` to run the repository independently of the standalone generator.
Every stage writes its own log and the runner writes `results.json`; exit status
is nonzero on any failure or timeout. `--timeout` sets the per-stage deadline
(default 180 seconds). On timeout, the runner kills the stage's process group,
including its DataLoader workers. Run without `-O`.

Coverage:

- `tests/test_instruction_augmentation.py`: 10 sampler tests covering fixed eval,
  canonical fallback, joint probability distribution, seed reproducibility,
  schema rejection, control characters and load/save integrity.
- `tests/test_instruction_dataset.py`: 12 integration tests with actual synthetic
  Parquet files and the actual dataset class. Only the image/token processor is
  replaced, so no pretrained weights are needed. Covers sparse episode IDs,
  train/val partition, v2 empty shared caches, missing/invalid labels, token limits,
  all state/action targets, internal subtask transitions and episode-end padding,
  regular/best checkpoint saving and retention, stale snapshot removal, and
  repeated epochs using both `spawn` and `fork` where available.
- External `src/test_generation.py`: 8 tests for candidate preservation, failed
  generation, parser validation, HTTP payload/header construction, malformed
  responses, missing keys and redacted errors. Network calls are mocked.
- `scripts/smoke_instruction_augmentation.py`: real videos from all three cameras,
  every subtask, actual cached tokenizer, boundary fallback without action changes,
  all three batches of a five-sample two-worker loader repeated with the same seed,
  canonical validation repeatability and snapshot hash.
- CLI import/help and shell launcher syntax are checked separately.

These tests validate data plumbing and instruction contracts. They do not measure
policy success, semantic equivalence of arbitrary new paraphrases, distributed GPU
training, or live Gemma API availability. Split validation now rejects invalid
ratios and configurations that leave no training episodes; use `val_ratio=0` for
an explicit unsplit dataset.

### ai-server-2 deployment

The server-specific entry point is `scripts/train_ai_server_2.sh smoke|train`.
It selects physical GPUs 0 and 1, batch size 4 per GPU, seed 42, 50% subtask
selection and 50% augmentation. The smoke mode performs two full training steps,
validation and checkpoint saving. Train mode defaults to 100,000 steps with
validation/checkpointing every 250 steps. W&B is disabled; TensorBoard logs are
written under the output directory. Existing initialized output directories are
rejected; restarting is an explicit choice, not an automatic resume.

Server paths:

- Code: `/nvme/anhnb9/projects/Pelican-VLA05` (branch `feat/dynamic-instruction-augmentation`).
- Standalone generator: `/nvme/anhnb9/projects/augment_coffee_data/src`.
- Dataset: `/nvme/anhnb9/datasets/serving_brewed_coffee_clean_374_episodes`.
- Policy and Cosmos weights: `pretrained_model/` under the code directory.
- Environment: `/nvme/anhnb9/envs/pelican_training/bin/python`.
- Hugging Face cache: `/nvme/anhnb9/cache/huggingface` (offline mode).
- Preflight reports: `/nvme/anhnb9/runs/coffee_augmentation_preflight`.
- Run outputs: `/nvme/anhnb9/runs/coffee_aug_20261020_2gpu/{smoke,train}`.

The training environment reads the existing inference environment's packages via
`pelican_runtime.pth`, with decord/TensorBoard dependencies installed in its own
site-packages. It does not modify the inference environment. This is an overlay,
not a fully independent package installation; changes to the underlying inference
environment can affect future runs.

Code was synchronized with rsync from tracked and nonignored source files,
including uncommitted augmentation work. `.git`, existing checkpoints and run logs
were not copied. The destination repository's history is retained. The source
manifest records hashes and the source base commit; the base commit alone does
not describe uncommitted changes. Dataset synchronization dereferences symlinks
(`rsync -L`) so videos are present locally on the server. SHA-256 manifests under
the preflight directory record dataset and weight integrity.

GPU sharing was explicitly authorized for this run; the existing jobs belonging
to other users must not be stopped. Launch only one smoke/train process at a time.
The generic launcher now accepts `CONDA_ENV_PYTHON`, `MASTER_PORT` and
`QWEN3_VL_PATH`, and uses that interpreter's `torch.distributed.run` for DDP.

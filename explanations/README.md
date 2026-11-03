# Explanations & Visualizations Directory

Thư mục này dùng để lưu trữ các kết quả phân tích giải thích mô hình (model explanations), attention heatmaps, và các biểu đồ visualize phục vụ việc debug các lượt inference theo từng phiên (`session_id`).

## Cấu trúc thư mục

```
explanations/
└── <session_id>/                          # Ví dụ: 20261015T162257+0700_797af814
    ├── cycle_000015_rollout_step0_dim0/   # Thư mục xuất ra từ scripts/explain_session.py
    │   ├── cam_head.png                   # Attention heatmap overlay trên ảnh camera đầu
    │   ├── cam_head.npz                   # Attention tensor + RGB + token maps
    │   ├── cam_left_wrist.png             # Heatmap camera cổ tay trái
    │   ├── cam_left_wrist.npz
    │   ├── cam_right_wrist.png            # Heatmap camera cổ tay phải
    │   ├── cam_right_wrist.npz
    │   └── metadata.json                  # Replay error, method, tolerance, cycle info
    └── ...
```

## Lệnh tạo visualization & giải thích cho một cycle

```bash
/home/anhnb9/miniconda3/envs/pelican_vla/bin/python scripts/explain_session.py \
  logs/sessions/20261015T162257+0700_797af814 \
  --cycle 15 \
  --method rollout
```
Kết quả sẽ tự động lưu vào `explanations/20261015T162257+0700_797af814/` (đồng thời có symlink liên kết hai chiều từ thư mục session tương ứng).

## Compare checkpoints on a real training sample

`scripts/explain_training_sample.py` runs both checkpoints on the same sample
loaded through `LeRobotPelicanDataset`. It checks training episode membership
against the checkpoint manifests, all three video streams, normalization stats,
and a complete 50-frame action chunk inside subtask 2. Input images use the
training loader's direct resize and actual temporal frames, not the generic
attention adapter's repeated-frame/letterbox path. The high-level canonical
instruction and inference seed are fixed for comparison.

```bash
/home/anhnb9/miniconda3/envs/pelican_vla/bin/python scripts/explain_training_sample.py \
  --device cuda:1 \
  --output explanations/coffee_aug_training_sample_new
```

The completed run is `coffee_aug_20261022_training_sample/`: episode 1, frame 842,
global row 2523, dataset train index 842. Open `comparison.ipynb` for embedded
attention and action plots. Raw per-camera heatmaps/token maps, normalized and
original-unit predicted/ground-truth actions, sample provenance and metrics are
saved as NPZ/JSON. The script refuses to overwrite an existing output directory.

`attention_comparison.png` uses shared color limits between the two checkpoints
within each camera. Rollout explains the action token at step 0 through language
decoder image tokens in the last denoising call. It is not dimension-specific,
causal attribution or full temporal/vision-encoder attribution. The action plot
shows an open-loop prediction of 50 steps; a single training sample cannot rank
checkpoints by held-out performance or real robot success.

## Attention video for both checkpoints

```bash
/home/anhnb9/miniconda3/envs/pelican_vla/bin/python scripts/render_training_attention_video.py \
  --device cuda:1 \
  --sample-report explanations/coffee_aug_20261022_training_sample/sample.json \
  --output explanations/coffee_aug_20261022_training_sample/videos
```

This covers the entire training episode referenced by the sample report. Default
stride 6 samples a 30 FPS episode at 5 FPS without interpolating attention. Each
checkpoint is loaded once. Both use identical RGB/state/instruction inputs and a
fixed inference seed. At episode boundaries, temporal inputs and action targets
follow the training dataset's clamping/padding behavior.

Outputs: `comparison.mp4` (RGB + both models), `best_model.mp4`,
`checkpoint-076250.mp4`, a preview image, raw RGB/heat/token-map NPZ archives, and
`manifest.json`. The encoder produces H.264/yuv420p MP4 with faststart and checks
all decoded frames. Color limits are fixed per camera across all timestamps and
both checkpoints, avoiding per-frame contrast changes. A video's last sampled
frame can extend playback by less than one sampling interval beyond the original
episode duration. This is offline decoder attention, not live robot execution or
causal attribution. Existing output directories are never overwritten.

## Compare the old coffee checkpoint on the same episode

```bash
/home/anhnb9/miniconda3/envs/pelican_vla/bin/python scripts/render_legacy_attention_comparison.py \
  --device cuda:1
```

This uses `checkpoints/pelican_astri_coffee_full_ft/best_model`, referenced by
session `20261015T162257+0700_797af814`, on dataset episode 1. It does not replay
the recorded robot session. The script verifies the RGB frames against the cached
two-checkpoint run and uses the same instruction and seed. Raw robot states are
normalized with the legacy checkpoint's own saved statistics. Its original
training split is unknown because no split manifest was saved.

Outputs are in `20261015T162257+0700_797af814/dataset_episode_001_comparison/`:
`legacy_best.mp4`, `comparison_three_checkpoints.mp4`, raw legacy attention,
a preview and provenance manifest. Color limits are recomputed across all three
models and all timestamps per camera. Both videos are decoded in full to verify
frame counts. The default output must not already exist; use `--output` for a
new run.

## Replay the recorded robot session through all three checkpoints

```bash
/home/anhnb9/miniconda3/envs/pelican_vla/bin/python scripts/render_session_checkpoint_comparison.py \
  logs/sessions/20261015T162257+0700_797af814 \
  --output explanations/20261015T162257+0700_797af814/session_three_checkpoints \
  --device cuda:1
```

This loads every non-warmup cycle's saved image history, raw state, task and
CPU/CUDA RNG. The legacy model uses the recorded normalization statistics;
both newer models use their own checkpoint statistics. Every legacy action chunk
is replayed and checked against recorded actions using the normalized tolerance
from `explain_session.py`. A mismatch aborts video creation.

Outputs include a four-row comparison (RGB, legacy, augmented best at 38750,
augmented at 76250), three separate model videos, raw attention/token maps,
input images, preview and manifest. Shared per-camera color limits are fixed
across all cycles and models. Playback follows cycle timestamps at 10 FPS,
holding each cycle image/heatmap until the next cycle; the last cycle is held
for the median cycle interval. All MP4 frames are decoded for validation.
This is an offline comparison on observations collected by the old policy,
not a robot rollout performed by the new policies. Existing output directories
are rejected.

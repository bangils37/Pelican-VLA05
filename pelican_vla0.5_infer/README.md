# PelicanVLA05

A compact, inference-focused release of the PelicanVLA05 vision-language-action
policy: a Qwen3-VL backbone with a flow-matching action head, a frozen NVIDIA
Cosmos image-tokenizer branch, and optional bottleneck tokens.

Training scripts and dataset builders are not included. The model definition
retains all parameterized modules, losses, and generation heads needed for
checkpoint compatibility. For attribution and third-party licenses, see
[`NOTICE`](NOTICE) and [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## Install

```bash
pip install -r requirements.txt
# or, to install as a package:
pip install -e .
```

Point `QWEN3_VL_PATH` at the Qwen3-VL weights used to initialise the backbone
(local dir or HF hub id):

```bash
export QWEN3_VL_PATH=/path/to/Qwen3-VL-4B-Instruct
```

Cosmos Tokenizer assets are separate model files. By default the pinned
`nvidia/Cosmos-0.1-Tokenizer-CI8x8` revision is downloaded on first use. For an
offline installation, prepare both `encoder.jit` and `decoder.jit` in one
directory and set:

```bash
export COSMOS_TOKENIZER_PATH=/path/to/Cosmos-0.1-Tokenizer-CI8x8
```

## Quick start

```python
import numpy as np
from pelicanvla05_infer import PelicanVLA05Inference, load_stats_json

# Normalization stats saved alongside the checkpoint (stats.json).
state_stats, action_stats = load_stats_json(
    "/path/to/pretrained_model/stats.json",
    state_keys=["observation.state.arm.position", "observation.state.effector.position"],
    action_keys=["action.arm.position", "action.effector.position"],
    robot_type="ur_5e_singleArm",   # omit if stats.json is not nested per robot
)

engine = PelicanVLA05Inference(
    model_path="/path/to/pretrained_model",
    camera_map={"front": "image0", "wrist": "image1", "side": "image2"},
    state_stats=state_stats,
    action_stats=action_stats,
    action_dim=7,
    delta_mask=[True, True, True, True, True, True, False],  # gripper is absolute
)

action_chunk = engine.infer(
    images={"front": rgb0, "wrist": rgb1, "side": rgb2},  # HxWxC uint8 RGB
    state=[j1, j2, j3, j4, j5, j6, gripper],
    task="pick up the cup",
)
# action_chunk: (chunk_size, action_dim) absolute targets
engine.reset()  # clear temporal buffers between episodes
```

Smoke-test the whole pipeline (random images, no robot needed):

```bash
python pelicanvla05_infer.py --model_path /path/to/pretrained_model --action_dim 7
```

## Astribot S1 deployment

The Astribot entrypoint uses the same ROS 2 topics, 16-D vector order and
hardware safety guards as the sibling Diffusion Policy runtime. The model input
order is:

```text
[left_arm_j0..j6, right_arm_j0..j6, left_gripper, right_gripper]
```

ROS 2 Humble and the local `astribot_msgs` overlay use Python 3.10. The existing
`pelican_vla` training environment on this machine uses Python 3.12 and cannot
load Humble's `rclpy` native extension. Install the Pelican dependencies into
the Python 3.10 Diffusion Policy runtime once:

```bash
/home/anhnb9/Documents/diffusion_policy/env/bin/python -m pip install \
  -r pelican_vla0.5_infer/requirements.txt
```

The launcher defaults to that Python 3.10 runtime; override it with
`PELICAN_PYTHON=/path/to/python3.10` if needed. Build and source
`astribot_msgs`, then run a command-free smoke test first:

```bash
scripts/run_astribot_inference.sh \
  checkpoints/pelican_astri_coffee_full_ft/best_model \
  --max-cycles 5
```

For the MuJoCo Astribot topics, add:

```bash
--ros-config pelican_vla0.5_infer/configs/astribot_s1_sim_ros2.yaml
```

Only after checking camera/state freshness, inference latency, raw arm jumps,
and gripper units should command publication be enabled:

```bash
scripts/run_astribot_inference.sh \
  checkpoints/pelican_astri_coffee_full_ft/best_model \
  --execute --prepare-ready-pose --execute-steps 8
```

Without `--execute` the process never publishes commands. Execute mode adds S1
joint limits, a 0.15-rad per-policy-step arm limit, a 20-unit gripper limit, a
250 Hz velocity-limited arm stream, stale-sensor rejection, tracking-error
abort, a 1-second inference-latency limit, and a final hold command. If the
checkpoint is too slow, `--inference-steps N` can reduce the flow-matching
iterations, but policy quality must be revalidated in dry-run/simulation. The
physical and simulator ROS mappings are in
`configs/astribot_s1_ros2.yaml` and `configs/astribot_s1_sim_ros2.yaml`.

The current checkpoint was trained with temporal image offsets `[-15, 0, 15]`.
Because a live controller cannot access the future `+15` frame, online
inference uses the newest available frame for that slot and prints a startup
warning. This train/deployment mismatch should be included in robot validation;
future training runs intended for online control should use causal offsets.

## Checkpoint layout

`from_pretrained` accepts a directory containing either a single
`model.safetensors` or the split files `backbone.safetensors` + `heads.safetensors`
(+ optional `action_head.safetensors` / `gen_head.safetensors`), together with
`config.json` and `stats.json`. `PelicanVLA05Inference` uses strict checkpoint loading
by default so mismatched or stale parameter names fail immediately.

## Model interface

```python
from lerobot.policies.pelicanvla05 import PelicanVLA05Policy
policy = PelicanVLA05Policy.from_pretrained(model_path)   # loads full or split weights
actions, _ = policy.predict_action_chunk(batch)      # batch built by the processor
```

## License

Apache-2.0. See [`LICENSE`](LICENSE). Third-party components (LeRobot, openpi,
and Hugging Face Transformers) are attributed in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

This license covers the source code in this repository. A checkpoint is a
separate distribution and must carry its own license and model card. Qwen3-VL
weights are Apache-2.0; NVIDIA Cosmos tokenizer weights use the NVIDIA Open
Model License.

## Session recording, HOME and offline explanations

Astribot inference automatically starts a uniquely named session under
`~/Desktop/pelican_astribot_logs/sessions/`. Override with `--record-dir`.
Recording begins after checkpoint loading, before waiting for live observations.
No robot command is published unless `--execute` is supplied.

Every session contains:

- `session.json`: configuration, git revision, checkpoint file sizes/mtimes,
  normalization stats, camera layout, tokenizer paths, library versions,
  outcome, writer errors and accepted/dropped record counts.
- `console.log`, `events.jsonl`: runtime console output, full-body measured joint
  positions plus source names/mode/velocity/acceleration/torque with source stamps,
  raw/limited policy commands and tracking
  diagnostics, and actual published commands (including 250 Hz arm streaming,
  READY/HOME and hold). Monotonic timestamps align these records.
- `videos/<camera>.mp4`: inference-resolution camera videos, default 30 FPS.
  `video_frame` events map encoded frame indices to actual sampling times;
  video timing is nominal FPS and is not authoritative when frames drop.
- `cycles/cycle_XXXXXX.npz` + `.json`: lossless RGB frames selected from history for each
  inference, state history/current state, raw absolute action chunk, CPU/CUDA
  RNG states, task, input arrival/source timestamps, full-body state snapshot
  and inference latency. Cycle `-1` is warm-up and is not executed.

The writer uses a **separate CPU process**, with bounded independent queues for
trace, events and video. Inference never waits for disk/video encoding. This
cannot guarantee zero overhead or lossless recording under arbitrary disk/CPU
load: queue/RNG bookkeeping still costs time. Overload drops records and reports
counts; `complete=false` explicitly marks incomplete sessions. A writer failure
raises at the next cycle and follows the existing hold/cleanup path. Flushing
happens after stopping commands, with a bounded timeout. Process kill/power loss
can leave incomplete files. `--no-record` disables videos only, not trace logging.
Selected history frames are deduplicated within a cycle and stored uncompressed;
unused intermediate camera frames are in video, not the lossless cycle archive.
Index/length fields preserve the temporal selection exactly.
Plan disk capacity accordingly. Do not mutate checkpoints during recording or
before replay (weights themselves are not copied into each session).

### HOME behavior

- `--execute` sessions return to HOME by default after `--max-cycles`,
  `--max-duration SECONDS`, or Enter in an interactive terminal.
- `--no-return-home` disables this behavior. `--home-duration` defaults to 4 s.
- Ctrl-C, exceptions, stale sensors and tracking aborts **hold** instead of
  initiating HOME motion. ROS shutdown also skips HOME. HOME freshness checks
  cover every moved body part, including head and torso.
- The HOME target closes both grippers, matching the existing DP HOME pose.
  Choose `--no-return-home` for tasks whose final object placement/gripper state
  must be preserved. HOME publishes an interpolated target; this is not a
  measured convergence guarantee.

Standalone HOME/READY (no checkpoint or cameras needed):

```bash
scripts/move_astribot_home.sh --pose home                 # diagnostics only
scripts/move_astribot_home.sh --pose home --execute       # confirms before moving
scripts/move_astribot_home.sh --pose ready --execute
```

### Explain the actual recorded input

Run offline in a Pelican environment, from the repository root:

```bash
python scripts/explain_session.py /path/to/session --cycle 0 --method rollout
python scripts/explain_session.py /path/to/session --cycle 0 \
  --method gradient_rollout --action-step 0 --action-dim 14
```

The tool reconstructs the inference batch using the recorded real history and
RNG state, checks replay actions against the saved chunk, and refuses to produce
an explanation if replay differs beyond tolerance. Errors are compared in
training-standard-deviation units (not mixed radians/gripper units); the default
absolute normalized tolerance is four compute-dtype epsilons, floored at 1e-5
(0.03125 for BF16). Use `--replay-atol` to require a tighter tolerance. Reports
include raw/normalized error, threshold and whether replay was bitwise equal.
Identical input tensors and RNG do not guarantee bitwise floating-point results
across processes; treat accepted nonzero error as approximate numerical replay. It saves PNG overlays, raw
heat/token maps and replay error under `session/explanations/`. It checks
checkpoint size/mtime identity, not cryptographic identity; keep the same model,
processor assets and software environment. CUDA recordings require CUDA replay.
Attention capture/backprop never runs inside the robot control loop.

The decoder attention maps still cover the Qwen image-token path, not full
Cosmos/vision-tower attribution. `gradient_rollout` remains a conditional
final-Euler-step method; full Chefer/LRP is not implemented.

### Repeatable regression tests and debugging

Run from the repository root with the inference environment's Python:

```bash
python scripts/test_pelican.py
```

This runs runtime, recorder/HOME, corruption/failure-injection, and attention
suites in separate processes, including plain function tests that unittest
otherwise misses. It sets the attention module path itself. Logs and
`report.json` go to a unique directory under `/tmp/pelican-tests`; any failed
suite or timeout returns a nonzero exit code. Use `--output DIR` to retain them
elsewhere. Tests do not command hardware.

For checkpoint + ROS integration, source ROS Humble and the Astribot message
overlay first, and set `QWEN3_VL_PATH`, `COSMOS_TOKENIZER_PATH`, and
`CUDA_VISIBLE_DEVICES` for the deployment environment:

```bash
python scripts/test_pelican_ros.py \
  --model-path checkpoints/pelican_astri_coffee_full_ft/best_model
```

The harness creates its own simulator, localhost-only ROS domain 197 and a
unique `/pelican_test_<uuid>` prefix for **all** camera, state and command topics.
It refuses production topic configurations. It checks missing-head rejection,
standalone HOME, dry-run without commands, normal execute with HOME and execute
with HOME disabled. Every generated session is audited; MP4s are fully decoded.
The simulator follows commands immediately, so this verifies software plumbing,
not physical tracking, collision clearance or robot calibration. Logs, sessions,
configuration and JSON results remain under `/tmp/pelican-ros-tests`.
Do not run the older server-local `dry_run_mock.sh` on a hardware ROS domain.

Audit an existing session without loading a checkpoint or connecting to ROS:

```bash
python scripts/validate_session.py /path/to/session --expect-home yes --expect-commands yes
```

The auditor checks manifest/writer counts, dropped records, event JSON,
trace pairs, finite action/state tensors, sparse frame indices, RNG buffers and
video frame counts. `--skip-video` skips expensive decoding. Exit code 1 means
invalid/incomplete data; inspect the printed errors. A valid archive still does
not prove model replay equality; use `scripts/explain_session.py` for that check.

Measure recording overhead with a real checkpoint and replay inputs:

```bash
python scripts/benchmark_pelican_recorder.py \
  --model-path checkpoints/pelican_astri_coffee_full_ft/best_model \
  --dataset ../diffusion_policy/data/replay/astribot_s1/extracted/coffee_full.zarr \
  --pairs 20 --output /tmp/pelican-benchmarks
```

The benchmark warms the model, alternates baseline/recording in ABBA order,
records three camera streams plus synthetic joint events, and reports individual
samples, p50/p95/p99, median overhead and recorder completeness. CPU submission
latency is reported separately. This is a controlled replay workload, not a
worst-case latency guarantee. Avoid concurrent GPU jobs when comparing results.

When debugging, start with the failing suite's `.log` or integration case log,
then `session.json` (`complete`, `dropped`, `writer.error`) and `events.jsonl`.
The recorder now marks sessions incomplete if a queue payload was accepted but
lost during asynchronous serialization. Corrupt sparse history and nonfinite
replay tolerances are rejected instead of producing misleading explanations.

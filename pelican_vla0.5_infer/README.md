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

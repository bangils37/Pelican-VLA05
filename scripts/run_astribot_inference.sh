#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="${1:-$PROJECT_ROOT/checkpoints/pelican_astri_coffee_full_ft/best_model}"
if [[ $# -gt 0 ]]; then shift; fi

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ASTRIBOT_WS="${ASTRIBOT_WS:-$PROJECT_ROOT/../vm_astribot}"
# ROS 2 Humble on Ubuntu 22.04 is built for Python 3.10.  Do not default to the
# training env on this machine (Python 3.12), because rclpy's native extension
# cannot be loaded there. The Diffusion Policy runtime is already Python 3.10.
DEFAULT_RUNTIME_PYTHON="$PROJECT_ROOT/../diffusion_policy/env/bin/python"
PELICAN_PYTHON="${PELICAN_PYTHON:-$DEFAULT_RUNTIME_PYTHON}"

if [[ ! -f "$ROS_SETUP" ]]; then
    echo "ROS 2 setup not found: $ROS_SETUP" >&2
    exit 1
fi
if [[ ! -f "$ASTRIBOT_WS/install/setup.bash" ]]; then
    echo "Astribot overlay not found: $ASTRIBOT_WS/install/setup.bash" >&2
    exit 1
fi
if [[ ! -x "$PELICAN_PYTHON" ]]; then
    echo "Python 3.10 runtime not found: $PELICAN_PYTHON" >&2
    echo "Set PELICAN_PYTHON to a Python 3.10 environment compatible with ROS 2 Humble." >&2
    exit 1
fi

set +u
# shellcheck source=/dev/null
source "$ROS_SETUP"
# shellcheck source=/dev/null
source "$ASTRIBOT_WS/install/setup.bash"
if [[ -f "$PROJECT_ROOT/../diffusion_policy/cyclone_dds_setup.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/../diffusion_policy/cyclone_dds_setup.sh"
fi
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-25}"
export COSMOS_TOKENIZER_PATH="${COSMOS_TOKENIZER_PATH:-$PROJECT_ROOT/pretrained_model/cosmos_tokenizer}"

if ! "$PELICAN_PYTHON" -c 'import rclpy, astribot_msgs, torch, transformers, draccus, cv2, yaml' 2>/dev/null; then
    echo "The selected Python cannot import all ROS/Pelican dependencies: $PELICAN_PYTHON" >&2
    echo "Install once with:" >&2
    echo "  $PELICAN_PYTHON -m pip install -r $PROJECT_ROOT/pelican_vla0.5_infer/requirements.txt" >&2
    exit 1
fi

exec "$PELICAN_PYTHON" "$PROJECT_ROOT/infer_astribot.py" \
    --model-path "$MODEL_PATH" "$@"

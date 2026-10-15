#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="${1:-$PROJECT_ROOT/checkpoints/pelican_astri_coffee_full_ft/best_model}"
CYCLES="${2:-3}"

echo "=========================================================="
echo "Pelican-VLA05 Astribot S1 - Safe Local Dry-Run (No Robot)"
echo "Target GPU: GPU 1 (RTX Pro 6000)"
echo "Model: $MODEL_PATH"
echo "Cycles: $CYCLES"
echo "=========================================================="

# 1. Source ROS and Astribot overlay
set +u
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash
# shellcheck source=/dev/null
source "$PROJECT_ROOT/../vm_astribot/install/setup.bash"
set -u
export ROS_DOMAIN_ID=25

PYTHON_BIN="$PROJECT_ROOT/../diffusion_policy/env/bin/python"

# 2. Start mock sensors in background
echo "[1/3] Khởi động mock publisher (3 camera + 6 state topics)..."
"$PYTHON_BIN" "$PROJECT_ROOT/tests/publish_mock_astribot.py" &
MOCK_PID=$!

cleanup() {
    echo ""
    echo "[3/3] Đang dọn dẹp tiến trình mock publisher (PID: $MOCK_PID)..."
    kill "$MOCK_PID" 2>/dev/null || true
    wait "$MOCK_PID" 2>/dev/null || true
    echo "Hoàn tất!"
}
trap cleanup EXIT

sleep 2

# 3. Run dry-run on GPU 1
echo "[2/3] Chạy suy luận Dry-Run (KHÔNG gửi command tới robot)..."
CUDA_VISIBLE_DEVICES=1 "$PROJECT_ROOT/scripts/run_astribot_inference.sh" \
    "$MODEL_PATH" \
    --device cuda:0 \
    --task "serving brewed coffee" \
    --max-cycles "$CYCLES"

from pathlib import Path
import sys
from collections import deque

import numpy as np
import cv2


INFER_DIR = Path(__file__).resolve().parents[1] / "pelican_vla0.5_infer"
sys.path.insert(0, str(INFER_DIR))

from astribot_runtime import (  # noqa: E402
    AstribotActionLimiter,
    TrackingCriticalGuard,
    compressed_image_to_rgb,
    pack_vector,
    rate_limit_arm,
    tracking_diagnostics,
    unpack_vector,
)
from basevla_infer import PelicanVLA05Inference  # noqa: E402


def _parts():
    return {
        "left_arm": np.arange(7, dtype=np.float32),
        "right_arm": np.arange(10, 17, dtype=np.float32),
        "left_gripper": [20.0],
        "right_gripper": [80.0],
    }


def test_astribot_vector_layout_round_trip():
    packed = pack_vector(_parts())
    unpacked = unpack_vector(packed)
    for key, expected in _parts().items():
        np.testing.assert_array_equal(unpacked[key], expected)


def test_action_limiter_caps_joint_and_gripper_steps():
    reference = np.zeros(16, dtype=np.float32)
    limited = unpack_vector(AstribotActionLimiter(0.1, 20.0).limit(np.full(16, 1000.0), reference))
    np.testing.assert_allclose(limited["left_arm"], 0.1)
    np.testing.assert_allclose(limited["right_arm"], 0.1)
    np.testing.assert_allclose(limited["left_gripper"], 20.0)
    np.testing.assert_allclose(limited["right_gripper"], 20.0)


def test_action_limiter_uses_asymmetric_shoulder_limits_and_rejects_nan():
    reference = np.zeros(16, dtype=np.float32)
    action = reference.copy()
    action[2] = 9.0
    action[9] = -9.0
    limiter = AstribotActionLimiter(10.0, 100.0)
    limited = unpack_vector(limiter.limit(action, reference))
    np.testing.assert_allclose(limited["left_arm"][2], 1.51)
    np.testing.assert_allclose(limited["right_arm"][2], -1.51)
    action[0] = np.nan
    try:
        limiter.limit(action, reference)
    except ValueError as error:
        assert "NaN" in str(error)
    else:
        raise AssertionError("NaN action was accepted")


def test_rate_limit_and_tracking_guard():
    result = rate_limit_arm(np.zeros(7), [1, -1, 0.01, -0.01, 0, 2, -2], 0.032)
    np.testing.assert_allclose(result, [0.032, -0.032, 0.01, -0.01, 0, 0.032, -0.032])

    command = np.zeros(16)
    measured = command.copy()
    measured[3] = -0.2
    diagnostics = tracking_diagnostics(command, measured)
    assert diagnostics.severity == "warning"
    assert diagnostics.worst_joint == 3

    guard = TrackingCriticalGuard(3)
    assert not guard.observe("critical")
    assert not guard.observe("critical")
    assert guard.observe("critical")


def test_external_camera_history_uses_control_frame_offsets():
    engine = PelicanVLA05Inference.__new__(PelicanVLA05Inference)
    engine.camera_map = {"cam_head": "image0"}
    engine.image_delta_indices = [-15, 0, 15]
    engine.image_resolution = (2, 2)
    engine._buffers = {"cam_head": deque(maxlen=32)}
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(16)]
    # Both a list (ROS snapshot) and a stacked ndarray (recorded replay) are
    # valid Sequences and must follow the same temporal indexing.
    engine.set_image_history({"cam_head": np.stack(frames)})
    stack, present = engine._temporal_stack("cam_head")
    assert present
    np.testing.assert_allclose(stack[:, 0, 0, 0].numpy() * 255, [0, 15, 15])


def test_compressed_camera_is_converted_from_bgr_to_rgb():
    bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    bgr[:] = [3, 17, 241]
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    rgb = compressed_image_to_rgb(encoded.tobytes())
    assert rgb.shape == (4, 5, 3)
    assert rgb[0, 0].tolist() == [241, 17, 3]


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} runtime tests passed")

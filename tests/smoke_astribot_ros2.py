"""ROS ABI smoke test; run with the ROS 2 Humble Python 3.10 runtime."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from astribot_msgs.msg import RobotJointState
from sensor_msgs.msg import CompressedImage, Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pelican_vla0.5_infer"))

from astribot_ros2 import AstribotS1Node, image_message_to_rgb  # noqa: E402
from astribot_runtime import S1_VMO_READY_POSE  # noqa: E402


def main() -> None:
    config_path = ROOT / "pelican_vla0.5_infer/configs/astribot_s1_ros2.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rclpy.init()
    node = AstribotS1Node(
        config,
        {"cam_head": (224, 224), "cam_left_wrist": (224, 224), "cam_right_wrist": (224, 224)},
        history_depth=16,
        state_sample_hz=30.0,
    )
    try:
        # Never let this test publish onto configured robot topics, even when a
        # physical bridge happens to be live on the same ROS domain.
        published = []

        class RecordingPublisher:
            def publish(self, message) -> None:
                published.append(message)

        node._command_publishers = {
            part: RecordingPublisher() for part in node._command_publishers
        }

        bgr = np.zeros((360, 640, 3), dtype=np.uint8)
        bgr[:] = [3, 17, 241]
        ok, encoded = cv2.imencode(".png", bgr)
        assert ok
        for key in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            message = CompressedImage()
            message.format = "png"
            message.data = encoded.tobytes()
            node._on_image(message, key, "compressed")

        for part, dof in (
            ("torso", 4),
            ("left_arm", 7),
            ("right_arm", 7),
            ("left_gripper", 1),
            ("right_gripper", 1),
            ("head", 2),
        ):
            message = RobotJointState()
            message.position = [0.0] * dof
            node._on_joint(message, part)

        node._sample()
        ready, reason = node.readiness()
        assert ready, reason
        images, states = node.snapshot()
        assert states[-1].shape == (16,)
        assert all(history[-1].shape == (224, 224, 3) for history in images.values())
        assert images["cam_head"][-1][0, 0].tolist() == [241, 17, 3]

        raw_message = Image()
        raw_message.height = 1
        raw_message.width = 1
        raw_message.encoding = "bgr8"
        raw_message.step = 3
        raw_message.data = bytes([3, 17, 241])
        assert image_message_to_rgb(raw_message)[0, 0].tolist() == [241, 17, 3]

        node.set_action_target(np.zeros(16))
        node._stream_command()
        node.publish_hold()
        node.move_to_pose(S1_VMO_READY_POSE, 0.02, 100.0, 1.0)
        assert published
        node.clear_observation_history()
        assert not node.readiness()[0]
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print("ROS bridge smoke test passed")


if __name__ == "__main__":
    main()

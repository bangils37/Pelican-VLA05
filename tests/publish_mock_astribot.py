#!/usr/bin/env python3
"""Publish synthetic Astribot cameras and joint states for local ROS dry-run."""

from __future__ import annotations

import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from astribot_msgs.msg import RobotJointState

CAMERA_TOPICS = (
    "/astribot_camera/head_rgbd/color_compress/compressed",
    "/astribot_camera/left_wrist_rgbd/color_compress/compressed",
    "/astribot_camera/right_wrist_rgbd/color_compress/compressed",
)

STATE_TOPICS = {
    "torso": "/astribot_torso/joint_space_states",
    "left_arm": "/astribot_arm_left/joint_space_states",
    "right_arm": "/astribot_arm_right/joint_space_states",
    "left_gripper": "/astribot_gripper_left/joint_space_states",
    "right_gripper": "/astribot_gripper_right/joint_space_states",
    "head": "/astribot_head/joint_space_states",
}

READY_POSITIONS = {
    "torso": [0.59620858, -1.19085960, 0.59082105, -0.00832367],
    "left_arm": [0.16365501, -0.02352135, -1.42463252, 1.65502828, -0.34839234, 0.12443571, 0.12673429],
    "right_arm": [-0.15411713, -0.02150648, 1.41121377, 1.64720447, 0.34210999, 0.12159182, -0.11975973],
    "left_gripper": [0.0],
    "right_gripper": [0.0],
    "head": [-0.00284645, 0.88701215],
}


class MockAstribotPublisher(Node):
    def __init__(self) -> None:
        super().__init__("mock_astribot_publisher")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        self._cam_pubs = [
            self.create_publisher(CompressedImage, topic, qos) for topic in CAMERA_TOPICS
        ]
        self._state_pubs = {
            part: self.create_publisher(RobotJointState, topic, qos)
            for part, topic in STATE_TOPICS.items()
        }

        # Pre-encode dummy JPEG frame
        height, width = 224, 224
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert ok
        self._payload = encoded.tobytes()

        self._timer = self.create_timer(1.0 / 30.0, self._publish)
        self.get_logger().info("Mock Astribot publisher running (30 Hz)")

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for pub in self._cam_pubs:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = self._payload
            pub.publish(msg)

        for part, pub in self._state_pubs.items():
            msg = RobotJointState()
            msg.header.stamp = stamp
            msg.position = READY_POSITIONS[part]
            pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = MockAstribotPublisher()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

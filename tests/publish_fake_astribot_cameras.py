#!/usr/bin/env python3
"""Publish synthetic Astribot camera frames for a safe local dry-run.

This helper deliberately creates only the three camera publishers below.  It
does not publish, subscribe to, or otherwise touch any robot command topic.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


CAMERA_TOPICS = (
    "/astribot_camera/head_rgbd/color_compress/compressed",
    "/astribot_camera/left_wrist_rgbd/color_compress/compressed",
    "/astribot_camera/right_wrist_rgbd/color_compress/compressed",
)


class SyntheticCameraPublisher(Node):
    def __init__(self) -> None:
        super().__init__("pelican_synthetic_camera_test")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publishers = [
            self.create_publisher(CompressedImage, topic, qos) for topic in CAMERA_TOPICS
        ]

        height, width = 480, 640
        x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        self._frames = []
        for index in range(len(CAMERA_TOPICS)):
            frame = np.empty((height, width, 3), dtype=np.uint8)
            frame[..., 0] = x
            frame[..., 1] = y
            frame[..., 2] = 60 + index * 80
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("failed to encode synthetic JPEG")
            self._frames.append(encoded.tobytes())

        self._timer = self.create_timer(1.0 / 30.0, self._publish)
        self.get_logger().info("Publishing synthetic images on camera topics only")

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for publisher, payload in zip(self._publishers, self._frames):
            message = CompressedImage()
            message.header.stamp = stamp
            message.header.frame_id = "synthetic_test_camera"
            message.format = "jpeg"
            message.data = payload
            publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = SyntheticCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # SIGINT may already have shut down the default context in Humble.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

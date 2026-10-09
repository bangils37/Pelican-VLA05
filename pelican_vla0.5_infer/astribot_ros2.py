"""ROS 2 bridge for Pelican-VLA05 inference on Astribot S1."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Mapping, Sequence

import numpy as np
import rclpy
from astribot_msgs.msg import RobotJointController, RobotJointState
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from astribot_runtime import (
    PART_DOF,
    PART_KEYS,
    compressed_image_to_rgb,
    pack_vector,
    rate_limit_arm,
    resize_rgb,
    unpack_vector,
)


PART_JOINT_NAMES = {
    "torso": [f"astribot_torso_joint_{index}" for index in range(1, 5)],
    "left_arm": [f"astribot_arm_left_joint_{index}" for index in range(1, 8)],
    "right_arm": [f"astribot_arm_right_joint_{index}" for index in range(1, 8)],
    "left_gripper": ["astribot_gripper_joint"],
    "right_gripper": ["astribot_gripper_joint"],
    "head": ["astribot_head_pan_joint", "astribot_head_tilt_joint"],
}


def image_message_to_rgb(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    channels = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
    }.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported ROS image encoding: {message.encoding}")
    minimum_step = int(message.width) * channels
    if int(message.step) < minimum_step:
        raise ValueError(f"invalid Image.step={message.step}, expected >= {minimum_step}")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected = int(message.height) * int(message.step)
    if raw.size < expected:
        raise ValueError(f"image has {raw.size} bytes, expected >= {expected}")
    image = raw[:expected].reshape(int(message.height), int(message.step))
    image = image[:, :minimum_step].reshape(int(message.height), int(message.width), channels)
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    elif encoding in ("bgr8", "8uc3"):
        image = image[:, :, ::-1]
    elif encoding == "bgra8":
        image = image[:, :, [2, 1, 0]]
    elif encoding in ("rgba8", "8uc4"):
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


def _camera_spec(value) -> tuple[str, str]:
    if isinstance(value, str):
        return value, "compressed" if value.endswith("/compressed") else "raw"
    topic = str(value["topic"])
    message_type = str(value.get("message_type", "raw")).lower()
    if message_type not in ("raw", "compressed"):
        raise ValueError(f"invalid camera message type for {topic}: {message_type}")
    return topic, message_type


class AstribotS1Node(Node):
    """Collect synchronized inputs and stream absolute arm/gripper targets."""

    def __init__(
        self,
        ros_config: Mapping,
        camera_shapes: Mapping[str, tuple[int, int]],
        history_depth: int,
        state_sample_hz: float,
    ) -> None:
        super().__init__("pelican_vla05_astribot_s1")
        self._lock = threading.RLock()
        self._camera_shapes = dict(camera_shapes)
        self._required_parts = tuple(ros_config.get("required_state_parts", PART_KEYS))
        self._state_parts = tuple(dict.fromkeys((*PART_KEYS, *self._required_parts)))
        unknown = sorted(set(self._state_parts) - set(PART_DOF))
        if unknown:
            raise ValueError(f"unknown state parts: {unknown}")
        self._latest_images: dict[str, np.ndarray] = {}
        self._image_times = {key: None for key in self._camera_shapes}
        self._image_history = {
            key: deque(maxlen=history_depth) for key in self._camera_shapes
        }
        self._parts: dict[str, np.ndarray] = {}
        self._part_times: dict[str, float] = {}
        self._state_history = deque(maxlen=history_depth)
        self._history_not_before = 0.0
        self._decode_errors: dict[str, str] = {}
        self._target: dict[str, np.ndarray] | None = None
        self._smooth_arms: dict[str, np.ndarray] = {}
        self._last_grippers: dict[str, np.ndarray] = {}
        self._publish_joint_names = bool(ros_config.get("publish_joint_names", False))

        stream = dict(ros_config.get("command_stream", {}))
        self.command_publish_hz = float(stream.get("publish_hz", 250.0))
        self.arm_velocity_limit = float(stream.get("arm_velocity_limit", 8.0))
        if min(self.command_publish_hz, self.arm_velocity_limit, state_sample_hz) <= 0:
            raise ValueError("ROS rates and velocity limit must be positive")
        self._arm_max_step = self.arm_velocity_limit / self.command_publish_hz
        sensor_group = ReentrantCallbackGroup()
        command_group = MutuallyExclusiveCallbackGroup()
        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=15,
        )
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        camera_topics = dict(ros_config["camera_topics"])
        missing = sorted(set(self._camera_shapes) - set(camera_topics))
        if missing:
            raise ValueError(f"ROS config is missing cameras: {missing}")
        self._camera_subscriptions = []
        for key in self._camera_shapes:
            topic, message_type = _camera_spec(camera_topics[key])
            message_cls = CompressedImage if message_type == "compressed" else Image
            self._camera_subscriptions.append(
                self.create_subscription(
                    message_cls,
                    topic,
                    lambda message, key=key, kind=message_type: self._on_image(
                        message, key, kind
                    ),
                    qos_sub,
                    callback_group=sensor_group,
                )
            )

        state_topics = dict(ros_config["state_topics"])
        command_topics = dict(ros_config["command_topics"])
        if set(self._state_parts) - set(state_topics):
            raise ValueError("ROS config does not cover all required state topics")
        if set(PART_KEYS) - set(command_topics):
            raise ValueError("ROS config does not cover all policy command topics")
        self._state_subscriptions = [
            self.create_subscription(
                RobotJointState,
                state_topics[part],
                lambda message, part=part: self._on_joint(message, part),
                qos_sub,
                callback_group=sensor_group,
            )
            for part in self._state_parts
        ]
        self._command_publishers = {
            part: self.create_publisher(RobotJointController, topic, qos_pub)
            for part, topic in command_topics.items()
            if part in PART_DOF
        }
        self._state_timer = self.create_timer(
            1.0 / state_sample_hz, self._sample, callback_group=sensor_group
        )
        self._command_timer = self.create_timer(
            1.0 / self.command_publish_hz, self._stream_command, callback_group=command_group
        )

    def _on_image(self, message, key: str, kind: str) -> None:
        try:
            rgb = (
                compressed_image_to_rgb(message.data)
                if kind == "compressed"
                else image_message_to_rgb(message)
            )
            rgb = resize_rgb(rgb, self._camera_shapes[key])
        except Exception as error:
            with self._lock:
                self._decode_errors[key] = str(error)
            return
        with self._lock:
            self._latest_images[key] = rgb
            self._image_times[key] = time.monotonic()
            self._decode_errors.pop(key, None)

    def _on_joint(self, message: RobotJointState, part: str) -> None:
        positions = np.asarray(message.position, dtype=np.float32)
        dof = PART_DOF[part]
        if positions.size < dof:
            self.get_logger().error(f"{part} has {positions.size} positions, expected {dof}")
            return
        if dof == 7 and message.name:
            by_name = dict(zip(message.name, positions))
            names = PART_JOINT_NAMES[part]
            if all(name in by_name for name in names):
                positions = np.asarray([by_name[name] for name in names], dtype=np.float32)
        with self._lock:
            self._parts[part] = positions[:dof].copy()
            self._part_times[part] = time.monotonic()

    def _sample(self) -> None:
        with self._lock:
            if not all(part in self._parts for part in self._state_parts):
                return
            if not all(key in self._latest_images for key in self._camera_shapes):
                return
            if any(
                self._part_times[part] < self._history_not_before
                for part in self._state_parts
            ) or any(
                self._image_times[key] is None
                or self._image_times[key] < self._history_not_before
                for key in self._camera_shapes
            ):
                return
            for key in self._camera_shapes:
                self._image_history[key].append(self._latest_images[key].copy())
            self._state_history.append(pack_vector(self._parts))

    def readiness(self) -> tuple[bool, str]:
        with self._lock:
            missing_images = sorted(set(self._camera_shapes) - set(self._latest_images))
            missing_parts = sorted(set(self._required_parts) - set(self._parts))
            errors = dict(self._decode_errors)
            sampled = bool(self._state_history)
        reasons = []
        if missing_images:
            reasons.append(f"missing images: {missing_images}")
        if missing_parts:
            reasons.append(f"missing joint states: {missing_parts}")
        if not missing_parts and not sampled:
            reasons.append("waiting for synchronized sample")
        if errors:
            reasons.append(f"image errors: {errors}")
        return not reasons, "; ".join(reasons) if reasons else "ready"

    def snapshot(self) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
        with self._lock:
            return (
                {key: list(history) for key, history in self._image_history.items()},
                list(self._state_history),
            )

    def input_ages(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            ages = {
                f"camera:{key}": np.inf if stamp is None else now - stamp
                for key, stamp in self._image_times.items()
            }
            ages.update(
                {
                    f"state:{part}": (
                        np.inf
                        if part not in self._part_times
                        else now - self._part_times[part]
                    )
                    for part in self._required_parts
                }
            )
        return ages

    def latest_state(self) -> np.ndarray:
        with self._lock:
            if not all(part in self._parts for part in PART_KEYS):
                raise RuntimeError("policy joint state is not ready")
            return pack_vector(self._parts)

    def clear_observation_history(self) -> None:
        """Drop transition frames and require fresh callbacks for every input."""
        with self._lock:
            for history in self._image_history.values():
                history.clear()
            self._state_history.clear()
            self._history_not_before = time.monotonic()

    def _publish_part(self, part: str, positions: Sequence[float]) -> None:
        message = RobotJointController()
        message.header.stamp = self.get_clock().now().to_msg()
        message.mode = 1
        if self._publish_joint_names:
            message.name = list(PART_JOINT_NAMES[part])
        message.command = [float(value) for value in positions]
        self._command_publishers[part].publish(message)

    def move_to_pose(
        self,
        pose: Mapping[str, Sequence[float]],
        duration_s: float,
        rate_hz: float,
        max_state_age: float,
    ) -> None:
        with self._lock:
            missing = sorted(set(pose) - set(self._parts))
            if missing:
                raise RuntimeError(f"ready-pose states are missing: {missing}")
            starts = {part: self._parts[part].astype(np.float64).copy() for part in pose}
        steps = max(1, round(duration_s * rate_hz))
        deadline = time.monotonic()
        for index in range(steps):
            stale = {
                key: age
                for key, age in self.input_ages().items()
                if age > max_state_age
            }
            if stale:
                raise RuntimeError(f"inputs became stale during ready pose: {stale}")
            alpha = 0.5 * (1.0 - np.cos(np.pi * (index + 1) / steps))
            current = {
                part: starts[part] + alpha * (np.asarray(target) - starts[part])
                for part, target in pose.items()
            }
            for part, target in current.items():
                self._publish_part(part, target)
            if all(part in current for part in PART_KEYS):
                with self._lock:
                    self._smooth_arms = {
                        arm: current[arm].copy() for arm in ("left_arm", "right_arm")
                    }
                    self._last_grippers = {
                        gripper: current[gripper].copy()
                        for gripper in ("left_gripper", "right_gripper")
                    }
            deadline += 1.0 / rate_hz
            time.sleep(max(0.0, deadline - time.monotonic()))
        self.set_action_target(pack_vector(pose))

    def set_action_target(self, action: Sequence[float]) -> None:
        parts = unpack_vector(action)
        with self._lock:
            if not self._smooth_arms:
                self._smooth_arms = {
                    arm: self._parts[arm].astype(np.float64).copy()
                    for arm in ("left_arm", "right_arm")
                }
            self._target = parts

    def _stream_command(self) -> None:
        with self._lock:
            if self._target is None:
                return
            arm_commands = {}
            for arm in ("left_arm", "right_arm"):
                smooth = rate_limit_arm(
                    self._smooth_arms[arm], self._target[arm], self._arm_max_step
                )
                self._smooth_arms[arm] = smooth
                arm_commands[arm] = smooth.copy()
            gripper_commands = {}
            for gripper in ("left_gripper", "right_gripper"):
                target = self._target[gripper].copy()
                if gripper not in self._last_grippers or not np.allclose(
                    target, self._last_grippers[gripper]
                ):
                    self._last_grippers[gripper] = target
                    gripper_commands[gripper] = target
        for arm, command in arm_commands.items():
            self._publish_part(arm, command)
        for gripper, command in gripper_commands.items():
            self._publish_part(gripper, command)

    def publish_hold(self) -> None:
        with self._lock:
            if not self._smooth_arms:
                return
            held = {
                "left_arm": self._smooth_arms["left_arm"].copy(),
                "right_arm": self._smooth_arms["right_arm"].copy(),
                "left_gripper": self._last_grippers.get(
                    "left_gripper", self._parts["left_gripper"]
                ).copy(),
                "right_gripper": self._last_grippers.get(
                    "right_gripper", self._parts["right_gripper"]
                ).copy(),
            }
            self._target = held
        for part, command in held.items():
            self._publish_part(part, command)

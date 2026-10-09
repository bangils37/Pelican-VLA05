"""ROS-independent Astribot S1 packing, image, and command-safety helpers.

The constants and command guards mirror the proven runtime from the sibling
``diffusion_policy`` repository.  Keeping this module ROS-independent makes it
possible to validate the dangerous parts of deployment on a workstation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


ACTION_LAYOUT = "arms_then_grippers"
PART_KEYS = ("left_arm", "right_arm", "left_gripper", "right_gripper")
PART_DOF = {
    "torso": 4,
    "left_arm": 7,
    "right_arm": 7,
    "left_gripper": 1,
    "right_gripper": 1,
    "head": 2,
}

S1_VMO_READY_POSE = {
    "torso": np.asarray([0.59620858, -1.19085960, 0.59082105, -0.00832367]),
    "left_arm": np.asarray(
        [0.16365501, -0.02352135, -1.42463252, 1.65502828, -0.34839234, 0.12443571, 0.12673429]
    ),
    "right_arm": np.asarray(
        [-0.15411713, -0.02150648, 1.41121377, 1.64720447, 0.34210999, 0.12159182, -0.11975973]
    ),
    "left_gripper": np.asarray([0.0]),
    "right_gripper": np.asarray([0.0]),
    "head": np.asarray([-0.00284645, 0.88701215]),
}

S1_ARM_LIMITS = {
    "left_arm": (
        np.asarray([-3.08, -1.51, -3.08, -0.0, -2.37, -0.78, -1.55]),
        np.asarray([3.08, 0.62, 1.51, 2.55, 2.37, 0.78, 1.55]),
    ),
    "right_arm": (
        np.asarray([-3.08, -1.51, -1.51, 0.0, -2.37, -0.78, -1.55]),
        np.asarray([3.08, 0.62, 3.08, 2.55, 2.37, 0.78, 1.55]),
    ),
}


def unpack_vector(vector: Sequence[float]) -> dict[str, np.ndarray]:
    """Split the dataset's 16-D ``[left arm, right arm, left/right grip]`` vector."""
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (16,):
        raise ValueError(f"Astribot vector must have shape (16,), got {value.shape}")
    return {
        "left_arm": value[0:7].copy(),
        "right_arm": value[7:14].copy(),
        "left_gripper": value[14:15].copy(),
        "right_gripper": value[15:16].copy(),
    }


def pack_vector(parts: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Pack named Astribot parts in the exact order used during Pelican training."""
    return np.concatenate(
        (
            np.asarray(parts["left_arm"], dtype=np.float32).reshape(7),
            np.asarray(parts["right_arm"], dtype=np.float32).reshape(7),
            np.asarray(parts["left_gripper"], dtype=np.float32).reshape(1),
            np.asarray(parts["right_gripper"], dtype=np.float32).reshape(1),
        )
    )


class AstribotActionLimiter:
    """Clamp absolute Pelican targets to S1 joint, gripper, and step limits."""

    def __init__(
        self,
        max_joint_delta: float = 0.15,
        max_gripper_delta: float = 20.0,
        enforce_joint_limits: bool = True,
    ) -> None:
        if max_joint_delta <= 0 or max_gripper_delta <= 0:
            raise ValueError("action delta limits must be positive")
        self.max_joint_delta = max_joint_delta
        self.max_gripper_delta = max_gripper_delta
        self.enforce_joint_limits = enforce_joint_limits

    def limit(self, action: Sequence[float], reference: Sequence[float]) -> np.ndarray:
        target = unpack_vector(action)
        current = unpack_vector(reference)
        if any(not np.isfinite(value).all() for value in (*target.values(), *current.values())):
            raise ValueError("action/reference contains NaN or infinity")
        for arm in ("left_arm", "right_arm"):
            lower, upper = S1_ARM_LIMITS[arm]
            if self.enforce_joint_limits:
                target[arm] = np.clip(target[arm], lower, upper)
            delta = np.clip(
                target[arm] - current[arm], -self.max_joint_delta, self.max_joint_delta
            )
            target[arm] = current[arm] + delta
            if self.enforce_joint_limits:
                target[arm] = np.clip(target[arm], lower, upper)
        for gripper in ("left_gripper", "right_gripper"):
            delta = np.clip(
                target[gripper] - current[gripper],
                -self.max_gripper_delta,
                self.max_gripper_delta,
            )
            target[gripper] = np.clip(current[gripper] + delta, 0.0, 100.0)
        return pack_vector(target)


def rate_limit_arm(current: Sequence[float], target: Sequence[float], max_step: float) -> np.ndarray:
    current_np = np.asarray(current, dtype=np.float64)
    target_np = np.asarray(target, dtype=np.float64)
    if current_np.shape != (7,) or target_np.shape != (7,):
        raise ValueError("arm current/target must have shape (7,)")
    if max_step <= 0 or not np.isfinite(current_np).all() or not np.isfinite(target_np).all():
        raise ValueError("arm values must be finite and max_step positive")
    return current_np + np.clip(target_np - current_np, -max_step, max_step)


@dataclass(frozen=True)
class TrackingDiagnostics:
    maximum: float
    rms: float
    worst_joint: int
    severity: str


def tracking_diagnostics(
    previous_command: Sequence[float],
    measured_state: Sequence[float],
    warn_threshold: float = 0.15,
    critical_threshold: float = 0.30,
) -> TrackingDiagnostics:
    if warn_threshold <= 0 or critical_threshold <= warn_threshold:
        raise ValueError("tracking thresholds must satisfy 0 < warn < critical")
    previous = unpack_vector(previous_command)
    measured = unpack_vector(measured_state)
    error = np.concatenate((previous["left_arm"], previous["right_arm"])) - np.concatenate(
        (measured["left_arm"], measured["right_arm"])
    )
    absolute = np.abs(error)
    maximum = float(absolute.max())
    severity = (
        "critical"
        if maximum > critical_threshold
        else "warning"
        if maximum > warn_threshold
        else "ok"
    )
    return TrackingDiagnostics(
        maximum=maximum,
        rms=float(np.sqrt(np.mean(np.square(error)))),
        worst_joint=int(np.argmax(absolute)),
        severity=severity,
    )


@dataclass
class TrackingCriticalGuard:
    limit: int = 3
    consecutive: int = 0

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("tracking critical limit must be >= 0")

    def observe(self, severity: str) -> bool:
        if severity not in ("ok", "warning", "critical"):
            raise ValueError(f"unknown tracking severity: {severity}")
        self.consecutive = self.consecutive + 1 if severity == "critical" else 0
        return self.limit > 0 and self.consecutive >= self.limit


def compressed_image_to_rgb(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV could not decode compressed camera image")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def resize_rgb(image: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != output_hw:
        image = cv2.resize(image, output_hw[::-1], interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)

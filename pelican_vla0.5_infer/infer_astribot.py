#!/usr/bin/env python3
"""Run a Pelican-VLA05 checkpoint on Astribot S1 through ROS 2.

Dry-run is the default.  Robot commands are published only with ``--execute``.
The ROS transport and safety behavior follow Diffusion Policy's Astribot
inference runtime, while checkpoint loading and prediction use Pelican-VLA05.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from astribot_runtime import (
    S1_VMO_READY_POSE,
    AstribotActionLimiter,
    TrackingCriticalGuard,
    tracking_diagnostics,
    unpack_vector,
)
from basevla_infer import PelicanVLA05Inference


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "astribot_s1_ros2.yaml"
DEFAULT_MODEL = HERE.parent / "checkpoints" / "pelican_astri_coffee_full_ft" / "best_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pelican-VLA05 inference on Astribot S1")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--stats-path", default=None, help="default: MODEL_PATH/stats.json")
    parser.add_argument("--ros-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--qwen3-vl-path", default=os.environ.get("QWEN3_VL_PATH"))
    parser.add_argument(
        "--cosmos-tokenizer-path", default=os.environ.get("COSMOS_TOKENIZER_PATH")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default="serving brewed coffee")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--execute-steps", type=int, default=8)
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="override checkpoint flow-matching steps; validate policy quality after changing",
    )
    parser.add_argument(
        "--max-inference-latency",
        type=float,
        default=1.0,
        help="refuse to publish a chunk inferred more slowly than this many seconds",
    )
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--max-sensor-age", type=float, default=1.0)
    parser.add_argument("--max-joint-delta", type=float, default=0.15)
    parser.add_argument("--max-gripper-delta", type=float, default=20.0)
    parser.add_argument("--tracking-warn-threshold", type=float, default=0.15)
    parser.add_argument("--tracking-critical-threshold", type=float, default=0.30)
    parser.add_argument("--tracking-critical-limit", type=int, default=3)
    parser.add_argument("--execute", action="store_true", help="publish commands to the robot")
    parser.add_argument("--yes", action="store_true", help="skip execute-mode confirmation")
    parser.add_argument("--prepare-ready-pose", action="store_true")
    parser.add_argument("--ready-pose-duration", type=float, default=3.0)
    parser.add_argument("--ready-pose-rate", type=float, default=100.0)
    parser.add_argument("--disable-joint-limits", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_stats(path: Path) -> tuple[dict, dict]:
    with path.open(encoding="utf-8") as file:
        stats = json.load(file)
    # Checkpoints produced by this repository use state/action.  Also accept the
    # original LeRobot key for easier deployment from dataset metadata.
    state = stats.get("state", stats.get("observation.state"))
    action = stats.get("action")
    if state is None or action is None:
        raise ValueError(f"{path} must contain state (or observation.state) and action stats")
    result = []
    for label, value in (("state", state), ("action", action)):
        mean = np.asarray(value["mean"], dtype=np.float32)
        std = np.asarray(value["std"], dtype=np.float32)
        if mean.shape != (16,) or std.shape != (16,):
            raise ValueError(f"{label} stats must be 16-D, got {mean.shape}/{std.shape}")
        safe_std = std.copy()
        safe_std[safe_std < 1e-4] = 1.0
        result.append({"mean": mean, "std": safe_std})
    return result[0], result[1]


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "frequency": args.frequency,
        "ready-timeout": args.ready_timeout,
        "max-sensor-age": args.max_sensor_age,
        "max-joint-delta": args.max_joint_delta,
        "max-gripper-delta": args.max_gripper_delta,
        "max-inference-latency": args.max_inference_latency,
        "ready-pose-duration": args.ready_pose_duration,
        "ready-pose-rate": args.ready_pose_rate,
    }
    invalid = [key for key, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"options must be positive: {invalid}")
    if args.execute_steps < 1:
        raise ValueError("--execute-steps must be >= 1")
    if args.inference_steps is not None and args.inference_steps < 1:
        raise ValueError("--inference-steps must be >= 1")
    if args.max_cycles is not None and args.max_cycles < 1:
        raise ValueError("--max-cycles must be >= 1")
    if args.tracking_critical_limit < 0:
        raise ValueError("--tracking-critical-limit must be >= 0")
    if (
        args.tracking_warn_threshold <= 0
        or args.tracking_critical_threshold <= args.tracking_warn_threshold
    ):
        raise ValueError("tracking thresholds must satisfy 0 < warn < critical")
    if args.prepare_ready_pose and not args.execute:
        raise ValueError("--prepare-ready-pose requires --execute")


def _wait_ready(node, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        ready, reason = node.readiness()
        if ready:
            return
        if reason != previous:
            print(f"Waiting for Astribot observations: {reason}", flush=True)
            previous = reason
        time.sleep(0.1)
    raise TimeoutError(
        f"Astribot inputs were not ready after {timeout:.1f}s: {node.readiness()[1]}"
    )


def _assert_fresh(node, max_age: float) -> None:
    stale = {key: age for key, age in node.input_ages().items() if age > max_age}
    if stale:
        summary = ", ".join(f"{key}={age:.2f}s" for key, age in stale.items())
        raise RuntimeError(f"stale Astribot input(s): {summary}")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    model_path = Path(args.model_path).expanduser().resolve()
    stats_path = (
        Path(args.stats_path).expanduser().resolve()
        if args.stats_path
        else model_path / "stats.json"
    )
    config_path = Path(args.ros_config).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"stats file does not exist: {stats_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"ROS config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        ros_config = yaml.safe_load(file)
    state_stats, action_stats = _load_stats(stats_path)
    if args.cosmos_tokenizer_path:
        os.environ["COSMOS_TOKENIZER_PATH"] = args.cosmos_tokenizer_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    camera_map = {
        "cam_head": "image0",
        "cam_left_wrist": "image1",
        "cam_right_wrist": "image2",
    }
    print(f"Loading Pelican-VLA05 checkpoint: {model_path}", flush=True)
    engine = PelicanVLA05Inference(
        model_path=model_path,
        camera_map=camera_map,
        state_stats=state_stats,
        action_stats=action_stats,
        action_dim=16,
        qwen3_vl_path=args.qwen3_vl_path,
        device=args.device,
        # The Astribot dataset contains absolute joint targets, not deltas.
        delta_mask=None,
    )
    if args.execute_steps > engine.policy.config.chunk_size:
        raise ValueError(
            f"--execute-steps={args.execute_steps} exceeds chunk_size={engine.policy.config.chunk_size}"
        )
    if args.inference_steps is not None:
        engine.policy.config.num_inference_steps = args.inference_steps
        engine.policy.model.config.num_inference_steps = args.inference_steps
    history_depth = 1 + max(0, -min(engine.image_delta_indices))
    camera_shapes = {key: tuple(engine.image_resolution) for key in camera_map}
    positive_offsets = [offset for offset in engine.image_delta_indices if offset > 0]
    if positive_offsets:
        print(
            "WARNING: checkpoint was trained with future image offset(s) "
            f"{positive_offsets}; online inference substitutes the newest available frame.",
            flush=True,
        )

    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from astribot_ros2 import AstribotS1Node
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("astribot_msgs"):
            raise RuntimeError(
                "astribot_msgs is unavailable; build/source the vm_astribot ROS 2 workspace first"
            ) from error
        raise

    rclpy.init()
    node = AstribotS1Node(
        ros_config=ros_config,
        camera_shapes=camera_shapes,
        history_depth=history_depth,
        state_sample_hz=args.frequency,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name="astribot-ros2", daemon=True)
    spin_thread.start()
    limiter = AstribotActionLimiter(
        max_joint_delta=args.max_joint_delta,
        max_gripper_delta=args.max_gripper_delta,
        enforce_joint_limits=not args.disable_joint_limits,
    )
    guard = TrackingCriticalGuard(args.tracking_critical_limit)
    commands_started = False
    previous_command = None

    def infer_once() -> tuple[np.ndarray, float]:
        _assert_fresh(node, args.max_sensor_age)
        images, states = node.snapshot()
        if not states:
            raise RuntimeError("state history is empty")
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)
        start = time.monotonic()
        actions = engine.infer({}, states[-1], args.task, image_history=images)
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)
        elapsed = time.monotonic() - start
        if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
            raise RuntimeError(f"model returned invalid action chunk: shape={actions.shape}")
        return actions, elapsed

    try:
        _wait_ready(node, args.ready_timeout)
        mode = "EXECUTE" if args.execute else "DRY-RUN"
        print(
            f"Ready: mode={mode}, task={args.task!r}, cameras={list(camera_map)}, "
            f"temporal_offsets={engine.image_delta_indices}, execute_steps={args.execute_steps}, "
            f"inference_steps={engine.policy.config.num_inference_steps}",
            flush=True,
        )
        if not args.execute:
            print("No robot commands will be published. Add --execute after dry-run validation.")
        elif not args.yes:
            answer = input("Enter 's' to start publishing Astribot commands: ").strip().lower()
            if answer != "s":
                print("Aborted before publishing commands.")
                return
        if args.prepare_ready_pose:
            commands_started = True
            node.move_to_pose(
                S1_VMO_READY_POSE,
                duration_s=args.ready_pose_duration,
                rate_hz=args.ready_pose_rate,
                max_state_age=args.max_sensor_age,
            )
            node.clear_observation_history()
            _wait_ready(node, args.ready_timeout)
            print("Astribot reached the VMO ready pose; fresh history started.", flush=True)
        if not args.no_warmup:
            _, latency = infer_once()
            print(f"Warm-up inference discarded ({latency:.3f}s)", flush=True)

        cycle = 0
        dt = 1.0 / args.frequency
        while rclpy.ok():
            actions, latency = infer_once()
            _assert_fresh(node, args.max_sensor_age)
            reference = node.latest_state()
            first = unpack_vector(actions[0])
            arm_jump = max(
                np.max(np.abs(first["left_arm"] - unpack_vector(reference)["left_arm"])),
                np.max(np.abs(first["right_arm"] - unpack_vector(reference)["right_arm"])),
            )
            print(
                f"cycle={cycle:04d} inference={latency:.3f}s raw_arm_jump={arm_jump:.3f}rad "
                f"grippers=[{first['left_gripper'][0]:.1f}, {first['right_gripper'][0]:.1f}]",
                flush=True,
            )
            if latency > args.max_inference_latency:
                message = (
                    f"inference latency {latency:.3f}s exceeds safety limit "
                    f"{args.max_inference_latency:.3f}s"
                )
                if args.execute:
                    raise RuntimeError(f"{message}; refusing to publish stale actions")
                print(f"WARNING: {message}", flush=True)
            if args.execute:
                deadline = time.monotonic()
                for raw_action in actions[: args.execute_steps]:
                    _assert_fresh(node, args.max_sensor_age)
                    measured = node.latest_state()
                    if previous_command is None:
                        previous_command = measured.copy()
                    diagnostics = tracking_diagnostics(
                        previous_command,
                        measured,
                        args.tracking_warn_threshold,
                        args.tracking_critical_threshold,
                    )
                    if diagnostics.severity != "ok":
                        print(
                            f"{diagnostics.severity.upper()} tracking: max={diagnostics.maximum:.3f}rad "
                            f"rms={diagnostics.rms:.3f} joint={diagnostics.worst_joint}",
                            flush=True,
                        )
                    if guard.observe(diagnostics.severity):
                        raise RuntimeError(
                            f"automatic stop after {guard.consecutive} consecutive critical tracking samples"
                        )
                    command = limiter.limit(raw_action, reference)
                    node.set_action_target(command)
                    commands_started = True
                    previous_command = command.copy()
                    reference = command
                    deadline += dt
                    time.sleep(max(0.0, deadline - time.monotonic()))
            else:
                time.sleep(args.execute_steps * dt)
            cycle += 1
            if args.max_cycles is not None and cycle >= args.max_cycles:
                break
    except KeyboardInterrupt:
        print("Interrupted; stopping inference.", flush=True)
    finally:
        if commands_started:
            try:
                node.publish_hold()
            except Exception as error:
                print(f"Warning: could not publish final hold command: {error}", flush=True)
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()

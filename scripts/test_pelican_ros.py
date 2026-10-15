#!/usr/bin/env python3
"""Isolated ROS + checkpoint integration test. Source ROS/Astribot first.

All camera, state AND command topics use a unique /pelican_test_<id> prefix.
Only simulator feedback is used. No hardware topics are published.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def assert_isolated(config):
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') != '197':
        raise RuntimeError('Simulator requires localhost-only ROS domain 197')
    topics = [v['topic'] for v in config['camera_topics'].values()]
    topics += list(config['state_topics'].values()) + list(config['command_topics'].values())
    if not all(t.startswith('/pelican_test_') for t in topics):
        raise RuntimeError('Refusing non-test topics')


def simulate(config_path, ready_path):
    import yaml
    import numpy as np
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from astribot_msgs.msg import RobotJointState, RobotJointController
    sys.path.insert(0, str(ROOT / 'pelican_vla0.5_infer'))
    from astribot_runtime import S1_VMO_READY_POSE
    config = yaml.safe_load(Path(config_path).read_text())
    assert_isolated(config)
    rclpy.init()
    node = Node('pelican_test_simulator')
    positions = {k: v.tolist() for k, v in S1_VMO_READY_POSE.items()}
    cameras = [node.create_publisher(CompressedImage, v['topic'], qos_profile_sensor_data)
               for v in config['camera_topics'].values()]
    states = {k: node.create_publisher(RobotJointState, v, qos_profile_sensor_data)
              for k, v in config['state_topics'].items()}
    def update(message, part):
        positions[part] = list(message.command)
    subscriptions = [node.create_subscription(RobotJointController, topic,
        lambda msg, part=part: update(msg, part), 10) for part, topic in config['command_topics'].items()]
    ok, image = cv2.imencode('.jpg', np.full((224, 224, 3), 100, np.uint8))
    assert ok
    def publish():
        stamp = node.get_clock().now().to_msg()
        for pub in cameras:
            msg = CompressedImage(); msg.header.stamp = stamp; msg.format = 'jpeg'; msg.data = image.tobytes()
            pub.publish(msg)
        for part, pub in states.items():
            if part == 'head' and Path(ready_path).with_name('omit_head').exists():
                continue
            msg = RobotJointState(); msg.header.stamp = stamp; msg.position = positions[part]
            pub.publish(msg)
        Path(ready_path).touch()
    timer = node.create_timer(1 / 30, publish)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/tmp/pelican-ros-tests'))
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--simulate', nargs=2, metavar=('CONFIG', 'READY'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.simulate:
        simulate(*args.simulate); return 0
    if args.model_path is None: parser.error('--model-path is required')
    import yaml
    from validate_session import validate
    from test_pelican import run_logged
    run = args.output.resolve() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run.mkdir(parents=True)
    config = yaml.safe_load((ROOT / 'pelican_vla0.5_infer/configs/astribot_s1_ros2.yaml').read_text())
    namespace = '/pelican_test_' + uuid.uuid4().hex[:12]
    for cam in config['camera_topics'].values(): cam['topic'] = namespace + cam['topic']
    for key in ('state_topics', 'command_topics'):
        config[key] = {part: namespace + topic for part, topic in config[key].items()}
    config_path = run / 'ros.yaml'; config_path.write_text(yaml.safe_dump(config))
    env = dict(os.environ, ROS_LOCALHOST_ONLY='1', ROS_DOMAIN_ID='197', RMW_IMPLEMENTATION='rmw_fastrtps_cpp',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key in ('FASTRTPS_DEFAULT_PROFILES_FILE', 'CYCLONEDDS_URI'): env.pop(key, None)
    results = []
    simulator = None
    def command(name, argv, expected_error=None):
        with (run / f'{name}.log').open('w') as log:
            try:
                code = run_logged([sys.executable, *map(str, argv)], env=env, log=log, timeout=args.timeout)
                if expected_error is None:
                    if code: raise RuntimeError(f'{name} exited {code}; see {log.name}')
                else:
                    log.flush()
                    if code == 0 or expected_error not in Path(log.name).read_text():
                        raise RuntimeError(f'{name}: expected failure {expected_error}; see {log.name}')
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f'{name} timeout; see {log.name}') from exc
    try:
        with (run / 'simulator.log').open('w') as log:
            simulator = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--simulate',
                str(config_path), str(run / 'ready')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        while not (run / 'ready').exists():
            if simulator.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Simulator did not start; see simulator.log')
            time.sleep(.1)
        (run / 'omit_head').touch()
        command('home_missing_head', ['pelican_vla0.5_infer/move_astribot_home.py', '--ros-config', config_path,
            '--execute', '--yes', '--duration', '.2', '--timeout', '2'], expected_error='TimeoutError')
        (run / 'omit_head').unlink()
        results.append({'case': 'home_missing_head', 'ok': True})
        command('standalone_home', ['pelican_vla0.5_infer/move_astribot_home.py', '--ros-config', config_path,
            '--execute', '--yes', '--duration', '.2', '--timeout', '10'])
        results.append({'case': 'standalone_home', 'ok': True})
        for name, execute, home in [('dry_run', False, False), ('execute_home', True, True),
                                     ('execute_no_home', True, False)]:
            destination = run / name
            argv = ['infer_astribot.py', '--model-path', args.model_path.resolve(), '--ros-config', config_path,
                '--max-cycles', '1', '--record-dir', destination, '--ready-timeout', '15',
                '--max-inference-latency', '10', '--execute-steps', '1', '--home-duration', '.2']
            if execute: argv += ['--execute', '--yes']
            if execute and not home: argv += ['--no-return-home']
            command(name, argv)
            sessions = list(destination.glob('*/session.json'))
            if len(sessions) != 1: raise RuntimeError(f'{name}: expected exactly one session')
            result = validate(sessions[0].parent, expect_home=home, expect_commands=execute)
            result['case'] = name; results.append(result)
            print(f'{name}: {"PASS" if result["ok"] else "FAIL"}', flush=True)
            if not result['ok']: raise RuntimeError(str(result['errors']))
    except Exception as exc:
        results.append({'ok': False, 'error': str(exc)})
        print(str(exc), file=sys.stderr)
    finally:
        if simulator is not None and simulator.poll() is None:
            simulator.send_signal(signal.SIGINT)
            try: simulator.wait(timeout=10)
            except subprocess.TimeoutExpired:
                simulator.kill(); simulator.wait()
        (run / 'report.json').write_text(json.dumps(results, indent=2))
        print(f'Report: {run}')
    return int(any(not r['ok'] for r in results))


if __name__ == '__main__':
    sys.exit(main())

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'pelican_vla0.5_infer')]
from session_recorder import SessionRecorder, decode_history, encode_history
from validate_session import validate


class ValidationTests(unittest.TestCase):
    def test_corrupt_sparse_history_rejected(self):
        good = encode_history({'cam': [np.zeros((4, 4, 3), np.uint8)] * 4}, [-3, 0])
        for indices in ([0], [0, 4], [-1, 3], [3, 0], [0, 0], [0., 3.], []):
            with self.subTest(indices=indices):
                broken = copy.deepcopy(good)
                broken['history_indices_cam'] = np.asarray(indices)
                with self.assertRaises(ValueError):
                    decode_history(broken, ['cam'])
        for length in (0, -1, 33, 2.5):
            broken = copy.deepcopy(good)
            broken['history_length_cam'] = np.asarray(length)
            with self.assertRaises(ValueError):
                decode_history(broken, ['cam'])

    def test_invalid_tolerance_fails_before_model_loading(self):
        for value in ('nan', 'inf', '-1'):
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/explain_session.py'),
                                     '/does/not/exist', f'--replay-atol={value}'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('finite and nonnegative', result.stderr)

    def test_invalid_fps(self):
        for fps in (float('nan'), float('inf'), 0, -1):
            with self.assertRaises(ValueError):
                SessionRecorder('/does/not/exist', {}, fps=fps)

    def test_audit_detects_disk_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = SessionRecorder(tmp, {'camera_map': {'cam': 'image0'}}, video=False)
            rec.event('published_command', command=np.zeros(7))
            rec.trace('cycle_000000', dict(state=np.zeros(16), states=np.zeros((1,16)),
                actions=np.ones((2,16)), rng_cpu=np.zeros(100, np.uint8), rng_cuda=np.zeros(0,np.uint8),
                **encode_history({'cam': [np.zeros((4,4,3),np.uint8)]}, [0])), {'task': 'test'})
            rec.close('completed')
            self.assertTrue(validate(rec.path, expect_commands=True, expect_home=False)['ok'])
            self.assertFalse(validate(rec.path, expect_home=True)['ok'])
            with (rec.path / 'events.jsonl').open('a') as stream:
                stream.write('{truncated\n')
            self.assertFalse(validate(rec.path)['ok'])
            (rec.path / 'cycles/cycle_000000.npz').write_bytes(b'corrupt')
            result = validate(rec.path)
            self.assertTrue(any('cycle_000000.npz' in e for e in result['errors']))

    def test_lost_queue_payload_marks_session_incomplete(self):
        # Queue feeder serialization runs after put_nowait succeeds. A payload
        # can be accepted but never reach the worker without a worker exception.
        with tempfile.TemporaryDirectory() as tmp:
            rec = SessionRecorder(tmp, {}, video=False)
            rec.event('unpicklable', callback=lambda: None)
            rec.close('completed')
            meta = json.loads((rec.path / 'session.json').read_text())
            self.assertFalse(meta['complete'])
            self.assertEqual(meta['accepted']['event'], 1)
            self.assertEqual(meta['writer']['written'].get('event', 0), 0)

class IsolationTests(unittest.TestCase):
    def test_runner_enforces_timeout(self):
        import os
        from test_pelican import run_logged
        with tempfile.TemporaryFile(mode='w') as log:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_logged([sys.executable, '-c', 'import time; time.sleep(60)'],
                           log=log, timeout=.1, env=dict(os.environ))


    def test_simulator_refuses_robot_topics_and_wrong_domain(self):
        from unittest.mock import patch
        from test_pelican_ros import assert_isolated
        config = {'camera_topics': {'cam': {'topic': '/pelican_test_unit/cam'}},
                  'state_topics': {'arm': '/pelican_test_unit/state'},
                  'command_topics': {'arm': '/pelican_test_unit/command'}}
        with patch.dict('os.environ', {'ROS_LOCALHOST_ONLY': '1', 'ROS_DOMAIN_ID': '197'}):
            assert_isolated(config)
            for section in ('state_topics', 'command_topics'):
                bad = copy.deepcopy(config); bad[section]['arm'] = '/astribot_arm_left/joint_space_command'
                with self.assertRaises(RuntimeError): assert_isolated(bad)
            bad = copy.deepcopy(config); bad['camera_topics']['cam']['topic'] = '/astribot_camera/image'
            with self.assertRaises(RuntimeError): assert_isolated(bad)
        with patch.dict('os.environ', {'ROS_LOCALHOST_ONLY': '1', 'ROS_DOMAIN_ID': '25'}):
            with self.assertRaises(RuntimeError): assert_isolated(config)


if __name__ == '__main__':
    unittest.main()

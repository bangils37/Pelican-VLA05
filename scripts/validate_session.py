#!/usr/bin/env python3
"""Audit a closed inference session without loading the model or contacting ROS."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pelican_vla0.5_infer'))
from session_recorder import decode_history
from astribot_runtime import S1_HOME_POSE


def validate(path, *, expect_home=None, expect_commands=None, check_video=True):
    path = Path(path)
    errors = []
    counts = Counter()
    home_commands = {}

    def require(condition, message):
        if not condition:
            errors.append(message)

    meta = json.loads((path / 'session.json').read_text())
    writer = json.loads((path / 'writer.json').read_text())
    require(meta.get('complete') is True, 'Session marked incomplete')
    require(writer.get('closed') is True and not writer.get('error'), 'Writer failed or not closed')
    require(not meta.get('flush_timed_out'), 'Flush timed out')
    require(not any(meta.get('dropped', {}).values()), 'Dropped records')
    require(writer == meta.get('writer'), 'Writer summary differs from manifest')
    for kind in ('trace', 'event', 'video'):
        require(meta.get('accepted', {}).get(kind, 0) == writer.get('written', {}).get(kind, 0),
                f'Accepted/written mismatch: {kind}')
    with (path / 'events.jsonl').open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                kind = event['kind']
                require(np.isfinite(event['monotonic']), f'Invalid timestamp at event {number}')
                counts['video' if kind == 'video_frame' else 'event'] += 1
                if kind == 'published_command':
                    counts['commands'] += 1
                    require(np.isfinite(event['command']).all(), f'Nonfinite command at event {number}')
                    if event.get('phase') == 'RETURN_HOME':
                        counts['home'] += 1
                        home_commands[event['part']] = event['command']
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f'Invalid event {number}: {exc}')
    for kind in ('event', 'video'):
        require(counts[kind] == writer.get('written', {}).get(kind, 0), f'File count mismatch: {kind}')
    archives = sorted((path / 'cycles').glob('*.npz'))
    require(len(archives) == writer.get('written', {}).get('trace', 0), 'Trace count mismatch')
    require({p.stem for p in archives} == {p.stem for p in (path / 'cycles').glob('*.json')},
            'Unpaired cycle JSON/NPZ')
    for archive in archives:
        try:
            cycle = json.loads(archive.with_suffix('.json').read_text())
            require(bool(cycle.get('task')), f'{archive.name}: missing task')
            with np.load(archive, allow_pickle=False) as data:
                require(data['state'].shape == (16,), f'{archive.name}: state shape')
                require(data['states'].ndim == 2 and data['states'].shape[1] == 16 and len(data['states']) > 0,
                        f'{archive.name}: state history shape')
                require(data['actions'].ndim == 2 and data['actions'].shape[1] == 16,
                        f'{archive.name}: action shape')
                for key in ('state', 'states', 'actions'):
                    require(np.isfinite(data[key]).all(), f'{archive.name}: nonfinite {key}')
                for key in ('rng_cpu', 'rng_cuda'):
                    require(data[key].ndim == 1 and data[key].dtype == np.uint8,
                            f'{archive.name}: invalid {key}')
                require(data['rng_cpu'].size > 0, f'{archive.name}: missing CPU RNG')
                decode_history(data, meta['camera_map'])
                for cam in meta['camera_map']:
                    frames = data[f'history_{cam}']
                    require(frames.ndim == 4 and frames.shape[-1] == 3 and frames.dtype == np.uint8,
                            f'{archive.name}: invalid RGB frames {cam}')
        except Exception as exc:
            errors.append(f'{archive.name}: {type(exc).__name__}: {exc}')
    if counts['home']:
        for part, target in S1_HOME_POSE.items():
            actual = np.asarray(home_commands.get(part, []))
            require(actual.shape == target.shape and np.allclose(actual, target, atol=1e-6, rtol=0),
                    f'HOME final target missing/incorrect: {part}')
    if expect_commands is not None:
        require(bool(counts['commands']) == expect_commands, 'Unexpected command presence/absence')
    if expect_home is not None:
        require(bool(counts['home']) == expect_home, 'Unexpected RETURN_HOME presence/absence')
    if check_video and counts['video']:
        import cv2
        for cam in meta['camera_map']:
            capture = cv2.VideoCapture(str(path / 'videos' / f'{cam}.mp4'))
            decoded = 0
            try:
                while capture.read()[0]:
                    decoded += 1
            finally:
                capture.release()
            require(decoded == counts['video'], f'{cam}: decoded {decoded}/{counts["video"]} frames')
    return {'session': str(path), 'ok': not errors, 'errors': errors, 'counts': dict(counts),
            'cycles': len(archives)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--expect-home', choices=['yes', 'no'])
    parser.add_argument('--expect-commands', choices=['yes', 'no'])
    parser.add_argument('--skip-video', action='store_true')
    args = parser.parse_args()
    try:
        result = validate(args.session, expect_home=None if args.expect_home is None else args.expect_home == 'yes',
                          expect_commands=None if args.expect_commands is None else args.expect_commands == 'yes',
                          check_video=not args.skip_video)
    except Exception as exc:
        result = {'ok': False, 'errors': [f'{type(exc).__name__}: {exc}']}
    print(json.dumps(result, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())

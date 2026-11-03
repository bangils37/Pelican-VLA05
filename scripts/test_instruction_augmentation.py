#!/usr/bin/env python
"""Run augmentation regression suites with per-stage logs and a JSON result report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-data', action='store_true', help='Also decode actual dataset videos with the cached Qwen tokenizer')
    parser.add_argument('--dataset-path', default='/home/aitt/data/serving_brewed_coffee_clean_374_episodes')
    parser.add_argument('--qwen-path', default='Qwen/Qwen3-VL-4B-Instruct')
    parser.add_argument('--external-src', type=Path, help='Also test the standalone augmenter and mocked Gemma client')
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=180, help='Timeout in seconds per stage')
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error('Run without -O; real-data assertions must remain enabled')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    report_dir = args.report_dir or Path(tempfile.mkdtemp(prefix='pelican-augmentation-tests-'))
    report_dir.mkdir(parents=True, exist_ok=True)
    stages = [
        ('sampler', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_instruction_augmentation.py', '-v']),
        ('dataset', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_instruction_dataset.py', '-v']),
        ('cli', [sys.executable, 'training/train.py', '--help']),
        ('launcher', ['bash', '-n', 'scripts/train_astri_coffee.sh']),
    ]
    if args.external_src:
        if not args.external_src.is_dir():
            parser.error('--external-src must be an existing source directory')
        stages.append(('generator', [sys.executable, '-m', 'unittest', 'discover', '-s', str(args.external_src.resolve()), '-p', 'test_*.py', '-v']))
    if args.real_data:
        stages.append(('real_data', [sys.executable, 'scripts/smoke_instruction_augmentation.py',
                                    '--dataset_path', args.dataset_path, '--qwen3_vl_path', args.qwen_path]))
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1')
    results = []
    for name, command in stages:
        print(f'Running {name}...', flush=True)
        start = time.monotonic()
        log_path = report_dir/f'{name}.log'
        with log_path.open('w') as log:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=args.timeout)
                    status = 'passed' if code == 0 else 'failed'
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    # Kill the whole group so DataLoader workers cannot outlive a failed stage.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    log.write(f'\nStage timed out after {args.timeout} seconds\n')
                    status = 'timeout'
            except OSError as error:
                log.write(str(error))
                status = 'failed'
        result = dict(stage=name, status=status, seconds=round(time.monotonic()-start, 2), log=str(log_path.resolve()))
        results.append(result)
        (report_dir/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(f'  {status}: {log_path}', flush=True)
    print(f'Report: {(report_dir/"results.json").resolve()}', flush=True)
    return int(any(item['status'] != 'passed' for item in results))


if __name__ == '__main__':
    raise SystemExit(main())

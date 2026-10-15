#!/usr/bin/env python3
"""Run regression suites in isolated subprocesses; retain logs and JSON summary."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run_logged(argv, *, log, timeout, env):
    """Timeout kills the entire test process group, including recorder workers."""
    process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/pelican-tests'))
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    output = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True)
    # The runtime file uses plain functions, not unittest: discovery alone misses it.
    suites = {'runtime': ['tests/test_astribot_runtime.py'],
              'session_and_home': ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'],
              'attention': ['-m', 'unittest', 'discover', '-s', 'attention_vis/tests', '-v']}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "attention_vis/src") + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    for name, argv in suites.items():
        started = time.monotonic()
        with (output / f'{name}.log').open('w') as log:
            try:
                code = run_logged([sys.executable, *argv], env=env, log=log, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                code = 124
        result = {'suite': name, 'returncode': code, 'seconds': time.monotonic() - started}
        results.append(result)
        print(f'{name}: {"PASS" if code == 0 else "FAIL"} ({result["seconds"]:.1f}s)', flush=True)
        if code:
            print((output / f'{name}.log').read_text()[-8000:])
    (output / 'report.json').write_text(json.dumps({'python': sys.executable, 'results': results}, indent=2))
    print(f'Report: {output}')
    return int(any(r['returncode'] for r in results))


if __name__ == '__main__':
    sys.exit(main())

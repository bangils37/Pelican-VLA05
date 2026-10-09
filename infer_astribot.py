#!/usr/bin/env python3
"""Repository-level entrypoint for Astribot S1 Pelican inference."""

from pathlib import Path
import runpy
import sys


INFER_DIR = Path(__file__).resolve().parent / "pelican_vla0.5_infer"
sys.path.insert(0, str(INFER_DIR))
runpy.run_path(str(INFER_DIR / "infer_astribot.py"), run_name="__main__")

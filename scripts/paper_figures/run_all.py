#!/usr/bin/env python
"""Regenerate every 6.2% paper output in dependency order.

    python run_all.py --config config_6_2pct.json [--skip-sensitivity]

Order: preprocess -> segment_3d -> slicewise -> kam_baseline -> comparison
       -> fig_3d_cells -> sensitivity.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    "run_preprocess.py",
    "run_segment_3d.py",
    "run_slicewise.py",
    "run_kam_baseline.py",
    "run_comparison.py",
    "fig_3d_cells.py",
    "run_sensitivity.py",
    "run_refinement_comparison.py",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()

    for stage in STAGES:
        if args.skip_sensitivity and stage == "run_sensitivity.py":
            continue
        cmd = [sys.executable, str(HERE / stage), "--config", args.config]
        if args.out_root:
            cmd += ["--out-root", args.out_root]
        print(f"\n=== {stage} ===")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

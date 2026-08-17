#!/usr/bin/env python3
"""Wait for the oracle search, then run analysis and all tests sequentially."""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "oracle_results"


def run(command: list[str]) -> dict:
    started = dt.datetime.now()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return {"command": command, "returncode": completed.returncode,
            "started": started.isoformat(), "finished": dt.datetime.now().isoformat()}


def main() -> int:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MALLOC_ARENA_MAX"] = "2"
    OUT.mkdir(parents=True, exist_ok=True)
    # Blocking acquisition means no analysis or tests overlap a segmentation.
    with (OUT / "oracle_search.lock").open("a+") as lock:
        print("waiting for oracle search lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print("search complete; starting sequential finalization", flush=True)
        records = [
            run([sys.executable, str(HERE / "oracle_analyze.py"),
                 "--out-dir", str(OUT)]),
            run([sys.executable, "-m", "pytest", "-q"]),
            run([sys.executable, "-m", "pytest",
                 str(HERE / "test_synthetic_3d_benchmark.py"), "-q"]),
        ]
    report = {"passed": all(r["returncode"] == 0 for r in records),
              "steps": records, "finished": dt.datetime.now().isoformat()}
    (OUT / "finalize_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

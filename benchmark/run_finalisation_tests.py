#!/usr/bin/env python3
"""Run final repository and benchmark validation sequentially and record it."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "continuation_results"

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"


def run(command: list[str]) -> dict:
    started = dt.datetime.now().astimezone()
    completed = subprocess.run(command, cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"command": command, "returncode": completed.returncode,
            "started": started.isoformat(),
            "finished": dt.datetime.now().astimezone().isoformat(),
            "output": completed.stdout}


def main() -> int:
    steps = [
        run([sys.executable, "-m", "pytest", "-q"]),
        run([sys.executable, "-m", "pytest",
             "benchmark/test_synthetic_3d_benchmark.py", "-q"]),
    ]
    report_path = OUT / "finalisation_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["tests"] = steps
    report["passed"] = bool(report.get("audit", {}).get("passed")) and all(
        step["returncode"] == 0 for step in steps)
    report["finished"] = dt.datetime.now().astimezone().isoformat()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [f"Finalisation finished: {report['finished']}",
             f"Overall passed: {report['passed']}"]
    for step in steps:
        lines.extend(["", "$ " + " ".join(step["command"]),
                      f"return code: {step['returncode']}", step["output"].rstrip()])
    (OUT / "finalisation.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

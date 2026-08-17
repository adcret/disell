#!/usr/bin/env python3
"""Complete only missing repeated-seed diagnostics for selected primary solutions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oracle_core as oc
import oracle_runner
import oracle_select as osel
from oracle_store import Store


def main() -> int:
    primary = HERE / "oracle_results"
    selected = json.loads((primary / "selected_solutions.json").read_text())
    store = Store(primary / "evaluations.jsonl")
    tasks = []
    for solution in ("balanced", "min_count"):
        row = selected[solution]
        params = {
            name: int(row[name]) if name == "min_cell_size" else float(row[name])
            for name in osel.PARAMETER_NAMES
        }
        key = oc.Config(**params).key()
        for seed in range(1, 21):
            if not store.has(key, seed):
                tasks.append((params, seed))
    print(f"missing diagnostic trials: {len(tasks)}", flush=True)
    for row in oracle_runner.evaluate_batch(
            tasks, primary / "cache", processes=1, progress_every=1,
            label="primary diagnostic "):
        row["stage"] = "diagnostic_repeat"
        store.append([row])  # append() flushes and fsyncs every yielded trial
        if row.get("status") != "ok":
            raise RuntimeError(f"diagnostic trial failed: {row}")
    print("primary diagnostic repeats complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

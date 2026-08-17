#!/usr/bin/env python3
"""Required one-trial profile and 20-trial memory-stability gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_core as oc
import oracle_runner
from oracle_store import Store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "oracle_results")
    parser.add_argument("--memory-limit-gib", type=float, default=None)
    args = parser.parse_args()
    cache = args.out_dir / "cache"
    workspace = oc.load_workspace(cache)
    rng = np.random.default_rng(20260812)
    tasks = []
    for i in range(21):
        config = oc.canonical(oc.Config(
            local_threshold_deg=float(rng.uniform(.007, .014)),
            global_threshold_deg=(-1.0 if i % 4 == 0 else float(rng.uniform(.08, .8))),
            footprint_tolerance=float(rng.uniform(.12, .28)),
            footprint_radius_um=float(rng.uniform(.8, 1.7)),
            min_cell_size=int(rng.integers(20, 221)),
            kam_radius_um=float(rng.uniform(.8, 1.7))), workspace)
        tasks.append((config.as_dict(), i))
    oracle_runner.precompute_kam([p["kam_radius_um"] for p, _ in tasks], cache, 1)
    store = Store(args.out_dir / "smoke_evaluations.jsonl")
    swap_start = int(psutil.swap_memory().used)
    pending = [(p, s) for p, s in tasks if not store.has(oc.Config(**p).key(), s)]
    for row in oracle_runner.evaluate_batch(
            pending, cache, 1, progress_every=1, label="smoke ",
            memory_limit_gib=args.memory_limit_gib):
        row["stage"] = "profile" if row["random_seed"] == 0 else "smoke"
        store.append([row])
        if row.get("status") != "ok":
            raise SystemExit(f"smoke gate failed: {row.get('error')}")
    rows = [r for r in store if r.get("status") == "ok"]
    peaks = np.asarray([r["peak_rss_bytes"] for r in rows], dtype=float)
    swap_growth = max(0, int(psutil.swap_memory().used) - swap_start)
    # Fresh processes may vary with parameter difficulty, but must not trend
    # upward systematically.  Allow 10% of the median across the 20-run gate.
    tail = peaks[1:]
    slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) > 1 else 0.0
    stable = len(rows) >= 21 and swap_growth <= 256 * 1024**2 and (
        slope <= 0.10 * float(np.median(tail)) / max(len(tail), 1))
    report = {
        "passed": bool(stable), "completed": len(rows),
        "peak_rss_max_bytes": int(peaks.max(initial=0)),
        "peak_rss_median_bytes": int(np.median(peaks)) if len(peaks) else 0,
        "rss_trend_bytes_per_trial": slope, "swap_growth_bytes": swap_growth,
    }
    (args.out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if stable else 2


if __name__ == "__main__":
    raise SystemExit(main())

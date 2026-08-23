#!/usr/bin/env python3
"""How the merge step should be gated: by size, and by what misorientation.

The merge folds a small region into its closest neighbour.  Two things gate it:
the region must be smaller than ``merge_size_voxels``, and its misorientation
from that neighbour must be below a threshold.  This sweeps both.

The threshold is ``merge_factor`` times the misorientation that neighbouring
*large* regions in this volume actually show -- about 0.37 deg at 6.2 % strain,
and measured from the segmentation itself rather than supplied.  That reference
is what makes the rule transferable: the absolute scale changes with strain, but
"a fraction of what a real cell boundary looks like here" does not.

Range.  A factor of 1 puts the threshold at a typical real boundary, so [0, 1]
spans everything from no merging to merging across gaps as wide as the genuine
ones.  The sweep samples it densely and continues a little past 1 to show the
collapse.

Usage::

    python merge_sweep.py factor --phantom 6p2 --workers 6
    python merge_sweep.py grid   --workers 6
    python merge_sweep.py report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "runs" / "merge_sweep"

TRUTH = {"2p4": 1481, "3p5": 1622, "4p6": 1924, "6p2": 2525, "primary": 2534}

#: 1000 points across [0, 1] -- factor 1 is a threshold as wide as a real cell
#: boundary -- continuing a little past it to show the collapse.
FACTORS = np.unique(np.concatenate([
    np.linspace(0.0, 1.0, 1001),
    np.round(np.linspace(1.0, 2.0, 41), 4),
]))

SIZES = (5, 10, 20, 40, 60, 100, 150, 250, 400, 700)

STATE: dict = {}


def cell_status(truth, prediction, tau=0.9):
    """Per true cell: 1 recovered, 2 fused with a neighbour, 0 neither."""

    import numpy as np
    import strict_recovery as sr

    rows, cols, counts, tsz, psz = sr.contingency(truth, prediction)
    recovered = np.unique(rows[(counts / psz[cols] >= tau) &
                               (counts / tsz[rows] >= tau)])
    substantial = (counts >= sr.SUBSTANTIAL_VOXELS) & (
        counts >= sr.SUBSTANTIAL_FRACTION * tsz[rows])
    per_pred = np.bincount(cols[substantial], minlength=psz.size)
    fused = np.setdiff1d(np.unique(rows[substantial & (per_pred[cols] >= 2)]),
                         recovered)
    out = np.zeros(tsz.size, np.uint8)
    out[recovered] = 1
    out[fused] = 2
    out[0] = 0
    return out


def setup(phantom_key: str):
    """Segment once; every sweep point then differs only in the merge."""

    import capped_search as cs
    import phantom_lab as lab

    ph = (lab.load_phantom() if phantom_key == "primary"
          else lab.load_strain_phantom(phantom_key))
    params = dict(cs.DEFAULTS)
    params.pop("merge_size_voxels"); params.pop("merge_threshold_deg")
    params["min_cell_size"] = 3
    params["local_threshold_deg"] = cs.DEFAULT_LOCAL_THRESHOLD_DEG["flood fill + merge"]
    result = lab.segment(ph, **params, seed=0)
    import numpy as np

    before = cell_status(ph.labels, result.labels)
    live = np.flatnonzero(np.bincount(ph.labels.ravel(), minlength=before.size))
    live = live[live > 0]
    STATE.update({
        "labels": result.labels, "field": ph.field, "truth": ph.labels,
        "local": params["local_threshold_deg"], "key": phantom_key,
        "before": result.n_cells, "status_before": before[live], "live": live,
    })


def init_worker(phantom_key: str):
    setup(phantom_key)


def evaluate(task):
    import numpy as np

    import merge_cells as mc
    import strict_recovery as sr

    factor, size = float(task[0]), int(task[1])
    percentile = float(task[2]) if len(task) > 2 else None
    merged, diagnostics = mc.merge_small_cells(
        STATE["labels"], STATE["field"], merge_size_voxels=size,
        merge_mode="relative", merge_factor=factor,
        spread_factor=None if percentile is None else 1.0,
        spread_percentile=80.0 if percentile is None else percentile,
        local_threshold_deg=STATE["local"], return_diagnostics=True)
    strict = sr.strict_recovery(STATE["truth"], merged, tau=0.9)
    after = cell_status(STATE["truth"], merged)[STATE["live"]]
    before = STATE["status_before"]
    gained = int(((before != 1) & (after == 1)).sum())
    lost = int(((before == 1) & (after != 1)).sum())
    return {
        "spread_percentile": percentile,
        "gained": gained, "lost": lost, "net": gained - lost,
        "blocked_by_spread": diagnostics["blocked_by_spread"],
        "phantom": STATE["key"], "merge_factor": factor, "merge_size_voxels": size,
        "threshold_deg": diagnostics["merge_threshold_deg"],
        "reference_deg": diagnostics["reference_misorientation_deg"],
        "merges": diagnostics["merges"],
        "cells_before": STATE["before"], "cells_after": strict["n_cells_pred"],
        "recovered_at_90": strict["recovered_at_tau"],
        "recovery_rate": strict["recovery_rate_at_tau"],
        "fused_true_cells": strict["fused_true_cells"],
        "contamination": strict["contamination"],
    }


def run(phantom_key: str, tasks, workers: int):
    import time
    from concurrent.futures import ProcessPoolExecutor

    import capped_search as cs

    OUT.mkdir(parents=True, exist_ok=True)
    store = OUT / f"{phantom_key}.jsonl"
    seen = {(round(r["merge_factor"], 6), r["merge_size_voxels"],
             r.get("spread_percentile")) for r in cs.read_rows(store)}
    todo = [t for t in tasks
            if (round(float(t[0]), 6), int(t[1]),
                float(t[2]) if len(t) > 2 else None) not in seen]
    if not todo:
        print(f"  {phantom_key}: complete")
        return
    print(f"  {phantom_key}: {len(todo)} of {len(tasks)} points", flush=True)
    started, done, buffer = time.perf_counter(), 0, []
    with cs.exclusive_store(store), ProcessPoolExecutor(
            max_workers=workers, initializer=init_worker,
            initargs=(phantom_key,)) as pool:
        for row in cs.as_they_complete(pool, evaluate, todo,
                                       cs.QUEUE_DEPTH * workers):
            buffer.append(row); done += 1
            if len(buffer) >= 300:
                cs.append_rows(store, buffer); buffer = []
            if done % max(1, len(todo) // 10) == 0:
                elapsed = time.perf_counter() - started
                print(f"    {done}/{len(todo)}  {elapsed/60:.1f} min "
                      f"({(len(todo)-done)*elapsed/max(done,1)/60:.1f} left)",
                      flush=True)
        if buffer:
            cs.append_rows(store, buffer); buffer = []


def report():
    import capped_search as cs
    import pandas as pd

    frames = []
    for path in sorted(OUT.glob("*.jsonl")):
        rows = cs.read_rows(path)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        print("nothing swept yet")
        return None
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(OUT / "merge_sweep.csv", index=False)

    for key, group in frame.groupby("phantom"):
        n = TRUTH[key]
        baseline = group[group.merge_factor == 0].recovered_at_90.max()
        best = group.loc[group.recovered_at_90.idxmax()]
        print(f"\n{key}: {len(group)} points, {n} true cells")
        print(f"  no merge          {100 * baseline / n:5.2f}%")
        print(f"  best              {100 * best.recovered_at_90 / n:5.2f}%  "
              f"at factor {best.merge_factor:g} (threshold "
              f"{best.threshold_deg:.4f} deg), size {int(best.merge_size_voxels)}"
              f"  -> +{100 * (best.recovered_at_90 - baseline) / n:.2f} points")
        per_size = group.loc[group.groupby("merge_size_voxels")
                             .recovered_at_90.idxmax()]
        print("  best per size cutoff:")
        for _, row in per_size.iterrows():
            print(f"    size {int(row.merge_size_voxels):>4}  "
                  f"{100 * row.recovered_at_90 / n:5.2f}%  "
                  f"factor {row.merge_factor:<8g} thr {row.threshold_deg:.4f}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["factor", "grid", "zoom", "report"])
    parser.add_argument("--phantom", default="6p2")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.stage == "factor":
        tasks = [(f, s) for s in SIZES for f in FACTORS]
        print(f"merge sweep: {len(FACTORS)} factors x {len(SIZES)} sizes "
              f"= {len(tasks)} points")
        run(args.phantom, tasks, args.workers)
    elif args.stage == "zoom":
        # The broad sweep put the optimum near factor 0.15 and size 40-60.  The
        # spread gate changes the picture -- it is what stops a merge absorbing
        # a real cell -- so the zoom sweeps it alongside, and scores by cells
        # gained against cells lost rather than by recovery alone.
        # Factor 0 is the no-merge baseline.  Without it the gain a merge buys
        # cannot be stated for a phantom the broad sweep never covered, which
        # is why three of the four read as nan in the report.
        factors = np.round(
            np.concatenate([[0.0], np.arange(0.05, 0.2501, 0.005)]), 4)
        sizes = (40, 45, 50, 55, 60, 65, 70)
        percentiles = (50.0, 65.0, 80.0, 95.0)
        tasks = [(fa, sz, pc) for pc in percentiles for sz in sizes for fa in factors]
        print(f"zoom: {len(factors)} factors x {len(sizes)} sizes x "
              f"{len(percentiles)} spread percentiles = {len(tasks)} per phantom")
        for key in ("6p2", "4p6", "3p5", "2p4"):
            run(key, tasks, args.workers)
    elif args.stage == "grid":
        # every strain, on the region the single-phantom sweep identified
        coarse = np.round(np.linspace(0.0, 1.0, 101), 4)
        tasks = [(f, s) for s in SIZES for f in coarse]
        for key in ("2p4", "3p5", "4p6", "6p2"):
            run(key, tasks, args.workers)
    else:
        report()


if __name__ == "__main__":
    main()

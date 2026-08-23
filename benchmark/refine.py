#!/usr/bin/env python3
"""Refined parameter sweeps for both arms, on every phantom.

The first search established where each method's optimum lies; this one
resolves it.  Two things motivated it.

*KAM* was searched on a percentile grid of 2.5-point steps starting at 20, and
its optimum sat at 22.5 -- close enough to the edge, and coarse enough in step,
that the reported ceiling could be an artefact of the grid.  The refined sweep
runs percentiles from 4 to 60 in steps of 0.5, adds smaller minimum sizes, and
restores kernel radii below the 0.9 um floor that the flood fill needed for its
out-of-plane support.  That floor was never appropriate for KAM: it is not a
flood-fill neighbourhood, and the experimental work it is modelled on uses
0.5 um.  (Measured outcome: smaller kernels are worse, so the floor did not in
fact bias the comparison -- but that had to be shown rather than assumed.)

*Flood fill* is refined around each strain's own optimum rather than on the
shared coarse grid, so the per-strain trend is not limited by grid resolution.

Both stores are keyed by configuration hash and resumable.

Usage::

    python refine.py kam        --workers 12
    python refine.py flood      --workers 12
    python refine.py merged     --workers 12
    python refine.py report
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

OUT = HERE / "runs" / "refined"

#: Every phantom the comparison is reported on.
PHANTOMS = {
    "primary": HERE / "phantoms" / "primary_6p2_consistent.npz",
    "2p4": HERE / "phantoms" / "strain_2p4_r0.npz",
    "3p5": HERE / "phantoms" / "strain_3p5_r0.npz",
    "4p6": HERE / "phantoms" / "strain_4p6_r0.npz",
    "6p2": HERE / "phantoms" / "strain_6p2_r0.npz",
}

# ------------------------------------------------------------------ KAM grid

#: Half-point steps, and starting well below the previous grid's floor of 20.
KAM_PERCENTILES = tuple(np.round(np.arange(4.0, 60.01, 0.5), 2))
#: The KAM arm is searched over its own range, 0.4-2.5 um, rather than the
#: 0.9-2.0 um window the flood fill needed.  That window was never appropriate
#: here: the lower bound existed to give the flood-fill neighbourhood
#: out-of-plane support, and the upper bound was set by the same argument.  One
#: representative radius per distinct rasterised footprint.
KAM_RADII_UM = (0.4, 0.57, 0.8, 0.895, 1.0, 1.08, 1.135, 1.15, 1.2, 1.265,
                1.285, 1.345, 1.445, 1.51, 1.565, 1.615, 1.7, 1.79, 1.89,
                1.97, 2.05, 2.155, 2.265, 2.335, 2.4, 2.475)
#: Percentile and radius are resolved in full above, because they are the axes
#: that move the result.  These two are bracketed rather than swept: the
#: parameter-economy measurement put their cost at 0.0016 and 0.0003
#: respectively, so a finer grid here would buy nothing and cost hours.
KAM_MIN_SIZES = (1, 3, 10)
KAM_CONNECTIVITY = (1, 2)

# ----------------------------------------------------------- flood-fill grid

#: Refined about each phantom's own coarse optimum.
FLOOD_RADII_UM = (1.0, 1.08, 1.135, 1.15, 1.2, 1.265, 1.285, 1.345, 1.445)
FLOOD_TOLERANCES = (0.02, 0.04, 0.07, 0.10, 0.15, 0.20)
FLOOD_MIN_SIZES = (3, 5, 8, 12, 20, 30, 45)
#: Finer than the coarse grid's factor-1.45 steps.
FLOOD_LOCAL_DEG = tuple(np.round(np.logspace(np.log10(0.006), np.log10(0.032), 16), 6))
FLOOD_GLOBAL_DEG = (-1.0,)
FLOOD_KAM_RADIUS_UM = 1.2

WORK = None


def init_worker(path: str):
    global WORK
    import capped_search as cs

    WORK = cs.Workspace(Path(path))


def score(labels):
    import capped_search as cs

    return cs.score_cheap(WORK.labels, labels)


def kam_task(task):
    import numpy as np

    import pipelines

    row = {"arm": "KAM threshold", **task}
    try:
        labels, markers, threshold = pipelines.run_kam_threshold(
            WORK.kam(task["kam_radius_um"]), WORK.mask,
            percentile=float(task["percentile"]),
            min_cell_size=int(task["min_cell_size"]),
            connectivity=int(task["connectivity"]),
            watershed_connectivity=1,
        )
        row["kam_threshold_deg"] = float(threshold)
        row.update(score(np.asarray(labels, dtype=np.int32)))
        row["status"] = "ok"
    except Exception as error:                      # noqa: BLE001
        row.update({"status": "invalid", "error": type(error).__name__})
    return row


#: The merge rule applied when the flood fill is scored as deployed.  The
#: merge zoom (``merge_sweep.py``) searched factor x size x spread percentile on
#: all four strain phantoms; this single setting lands within 0.002-0.004 of
#: every phantom's own optimum, so the arm is scored under one global rule
#: rather than one tuned per volume.  Relative mode is what makes that
#: possible: the threshold is a fraction of the misorientation neighbouring
#: large regions actually show in *this* volume, so it rescales with strain
#: on its own.
MERGE_RULE = {"merge_mode": "relative", "merge_factor": 0.15,
              "merge_size_voxels": 40, "spread_factor": 1.0,
              "spread_percentile": 95.0}


def merged_task(task):
    """The flood fill scored after its merge step -- the arm as deployed.

    Scoring the raw partition measures an intermediate product.  It also used
    to be scored against a rule that disqualified any surplus above 1.05x the
    true count, which forced the arm's threshold high because every
    over-segmenting value was rejected before the merge could act on it.  That
    rule is retired; this stage is what replaces it.
    """

    import capped_search as cs
    import merge_cells as mc

    row = {"arm": "flood fill + merge", **task, **MERGE_RULE}
    try:
        markers = cs.flood_markers(WORK, task)
        if int(markers.max()) == 0:
            row.update({"status": "invalid", "error": "NoAcceptedMarkers"})
            return row
        labels = cs.watershed(WORK, markers, task["kam_radius_um"])
        merged, diagnostics = mc.merge_small_cells(
            labels, WORK.field,
            merge_size_voxels=MERGE_RULE["merge_size_voxels"],
            merge_mode=MERGE_RULE["merge_mode"],
            merge_factor=MERGE_RULE["merge_factor"],
            spread_factor=MERGE_RULE["spread_factor"],
            spread_percentile=MERGE_RULE["spread_percentile"],
            local_threshold_deg=float(task["local_threshold_deg"]),
            return_diagnostics=True)
        row["merges"] = diagnostics["merges"]
        row["merge_threshold_deg"] = diagnostics["merge_threshold_deg"]
        row["reference_deg"] = diagnostics["reference_misorientation_deg"]
        row.update(score(merged))
        row["status"] = "ok"
    except Exception as error:                      # noqa: BLE001
        row.update({"status": "invalid", "error": type(error).__name__})
    return row


def flood_task(task):
    import capped_search as cs

    row = {"arm": "flood fill", **task}
    try:
        markers = cs.flood_markers(WORK, task)
        if int(markers.max()) == 0:
            row.update({"status": "invalid", "error": "NoAcceptedMarkers"})
            return row
        labels = cs.watershed(WORK, markers, task["kam_radius_um"])
        row.update(score(labels))
        row["status"] = "ok"
    except Exception as error:                      # noqa: BLE001
        row.update({"status": "invalid", "error": type(error).__name__})
    return row


def kam_grid():
    return [{"percentile": float(p), "kam_radius_um": float(r),
             "min_cell_size": int(m), "connectivity": int(c)}
            for p in KAM_PERCENTILES for r in KAM_RADII_UM
            for m in KAM_MIN_SIZES for c in KAM_CONNECTIVITY]


def flood_grid():
    return [{"footprint_radius_um": float(r), "footprint_tolerance": float(t),
             "local_threshold_deg": float(l), "global_threshold_deg": float(g),
             "min_cell_size": int(m), "kam_radius_um": FLOOD_KAM_RADIUS_UM,
             "seed": 0}
            for r in FLOOD_RADII_UM for t in FLOOD_TOLERANCES
            for l in FLOOD_LOCAL_DEG for g in FLOOD_GLOBAL_DEG
            for m in FLOOD_MIN_SIZES]


def key(row, names):
    import capped_search as cs

    return cs.config_hash({n: row.get(n) for n in names})


def run(name, grid, worker, names, workers):
    import time
    from concurrent.futures import ProcessPoolExecutor

    import capped_search as cs

    OUT.mkdir(parents=True, exist_ok=True)
    for label, path in PHANTOMS.items():
        store = OUT / f"{name}_{label}.jsonl"
        seen = {key(r, names) for r in cs.read_rows(store)}
        todo = [t for t in grid if key(t, names) not in seen]
        if not todo:
            print(f"  {label}: complete")
            continue
        print(f"  {label}: {len(todo)} of {len(grid)} to run", flush=True)
        started, done, buffer = time.perf_counter(), 0, []
        # The store lock is what keeps a second launch of the same stage from
        # redoing the work already in flight; the bounded queue is what keeps
        # peak memory set by the pool rather than by the length of the grid.
        with cs.exclusive_store(store), ProcessPoolExecutor(
                max_workers=workers, initializer=init_worker,
                initargs=(str(path),)) as pool:
            for row in cs.as_they_complete(pool, worker, todo,
                                           cs.QUEUE_DEPTH * workers):
                buffer.append(row); done += 1
                if len(buffer) >= 400:
                    cs.append_rows(store, buffer); buffer = []
                if done % max(1, len(todo) // 8) == 0:
                    elapsed = time.perf_counter() - started
                    print(f"    {done}/{len(todo)}  {elapsed/60:.1f} min "
                          f"({(len(todo)-done)*elapsed/max(done,1)/60:.1f} left)",
                          flush=True)
            if buffer:
                cs.append_rows(store, buffer); buffer = []


def report():
    import capped_search as cs
    import pandas as pd

    rows = []
    truth = {"primary": 2534, "2p4": 1481, "3p5": 1622, "4p6": 1924, "6p2": 2525}
    for label in PHANTOMS:
        for name, arm in (("kam", "KAM threshold"), ("flood", "flood fill"),
                          ("merged", "flood fill + merge")):
            store = OUT / f"{name}_{label}.jsonl"
            ok = [r for r in cs.read_rows(store) if r.get("status") == "ok"]
            if not ok:
                continue
            # Selection follows the study's policy, not a bare maximum.  The
            # two differ: with min_cell_size free down to 3 the highest
            # recovery on this grid is often a partition 1.1-1.6x the true
            # count, which the policy disqualifies precisely because no merge
            # step follows to remove the surplus.  Reporting the bare maximum
            # here would put a number in the paper that section 1 would not
            # have selected.  The unconstrained ceiling is carried alongside,
            # since the question this sweep exists to answer -- whether the
            # coarse grid resolved the optimum -- is about the ceiling.
            admissible = [r for r in ok if not cs._disqualified(r)]
            best = min(admissible or ok, key=cs.POLICIES[cs.STRICT_RECOVERY])
            recovered = best.get("recovered_at_90", 0)
            ceiling = max(r["recovered_at_90"] for r in ok)
            rows.append({
                "phantom": label, "arm": arm, "evaluated": len(ok),
                "admissible": len(admissible),
                "recovered": recovered, "rate": recovered / truth[label],
                "ceiling_rate": ceiling / truth[label],
                "contamination": round(best["contamination"], 4),
                "cells": best["n_cells_pred"],
                **{k: best.get(k) for k in
                   ("percentile", "kam_radius_um", "min_cell_size", "connectivity",
                    "footprint_radius_um", "footprint_tolerance",
                    "local_threshold_deg")},
            })
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "refined_optima.csv", index=False)
    (OUT / "refined_optima.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, default=float) + "\n")
    print(frame.to_string(index=False))
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["kam", "flood", "merged", "report"])
    # A KAM worker peaks near 0.55 GB (phantom, the 26-entry KAM cache and one
    # evaluation), so 12 fits comfortably on a 27 GB machine alongside a
    # desktop session.  It was 14 when the per-offset KAM buffer made a single
    # large-radius evaluation cost 4.8 GB, and that combination is what invoked
    # the OOM killer.
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    if args.stage == "kam":
        grid = kam_grid()
        print(f"refined KAM sweep: {len(grid)} configurations per phantom")
        run("kam", grid, kam_task,
            ("percentile", "kam_radius_um", "min_cell_size", "connectivity"),
            args.workers)
    elif args.stage == "merged":
        grid = flood_grid()
        print(f"refined merged sweep: {len(grid)} configurations per phantom")
        run("merged", grid, merged_task,
            ("footprint_radius_um", "footprint_tolerance", "local_threshold_deg",
             "global_threshold_deg", "min_cell_size", "kam_radius_um", "seed"),
            args.workers)
    elif args.stage == "flood":
        grid = flood_grid()
        print(f"refined flood-fill sweep: {len(grid)} configurations per phantom")
        run("flood", grid, flood_task,
            ("footprint_radius_um", "footprint_tolerance", "local_threshold_deg",
             "global_threshold_deg", "min_cell_size", "kam_radius_um", "seed"),
            args.workers)
    else:
        report()


if __name__ == "__main__":
    main()

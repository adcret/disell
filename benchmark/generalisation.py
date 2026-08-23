#!/usr/bin/env python3
"""Can these parameters be chosen without ground truth?

On experimental DFXM data there is no ground truth, so a method is only usable
if good parameters can be found without one.  Two properties decide that, and
neither is visible in a leaderboard of best scores.

``breadth``    How much of the searched parameter space performs near the
               method's own best.  A method whose accuracy collapses a step
               away from an exactly-tuned point cannot be tuned blind, however
               high that point scores.  Reported as the fraction of evaluated
               configurations reaching a given share of the arm's best
               recovery, which is directly comparable between arms because
               each is measured against its own optimum.

``transfer``   Whether parameters chosen on one microstructure still work on
               another.  The strain series gives four microstructures with
               known ground truth, so every optimum can be applied to every
               other strain and the loss measured.  This is the closest
               available proxy for "tuned on one experiment, applied to the
               next".

Usage::

    python generalisation.py breadth
    python generalisation.py transfer --workers 14
    python generalisation.py all --workers 14
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

OUT = HERE / "analysis"
PRIMARY = HERE / "runs/primary"
STRAIN_ROOT = HERE / "runs/strain"

#: Shares of an arm's own best recovery at which breadth is reported.
BREADTH_LEVELS = (0.95, 0.90, 0.80, 0.50)


# ------------------------------------------------------------------- breadth


def breadth(store: Path = PRIMARY) -> dict:
    """How much of each arm's searched space performs near its own best."""

    import capped_search as cs

    arms = {
        "flood fill": [r for r in cs.read_rows(store / "markers.jsonl")
                       if r.get("status") == "ok"],
        "KAM threshold": [r for r in cs.read_rows(store / "baseline.jsonl")
                          if r.get("status") == "ok"],
    }
    merge_rows = [r for r in cs.read_rows(store / "merge.jsonl")
                  if r.get("status") == "ok"]
    if merge_rows:
        arms["flood fill + merge"] = merge_rows

    report: dict = {
        "levels": list(BREADTH_LEVELS),
        "note": "each arm is measured against its own best, so the fractions "
                "compare tunability rather than accuracy.",
        "evaluation_scope": {
            "flood fill": {
                "source": "markers.jsonl",
                "selection": "exhaustive marker grid",
                "exhaustive": True,
            },
            "KAM threshold": {
                "source": "baseline.jsonl",
                "selection": "exhaustive baseline grid",
                "exhaustive": True,
            },
            "flood fill + merge": {
                "source": "merge.jsonl",
                "selection": "staged marker finalists x merge grid",
                "exhaustive": False,
            },
        },
        "arms": {},
    }
    for name, rows in arms.items():
        if not rows:
            continue
        recovered = np.array([r.get("recovered_at_90") or 0 for r in rows], float)
        best = float(recovered.max()) if recovered.size else 0.0
        if best <= 0:
            continue
        entry = {
            "n_configurations": int(recovered.size),
            "best_recovered_at_90": int(best),
            "median_recovered_at_90": float(np.median(recovered)),
        }
        for level in BREADTH_LEVELS:
            entry[f"fraction_within_{int(level * 100)}pct_of_best"] = float(
                np.mean(recovered >= level * best)
            )
        report["arms"][name] = entry
        print(f"  {name:20s} best {int(best):4d}; "
              + "  ".join(f">={int(l*100)}%: {entry[f'fraction_within_{int(l*100)}pct_of_best']:6.2%}"
                          for l in BREADTH_LEVELS))
    return report


def finalised_breadth(store: Path = PRIMARY) -> dict:
    """Describe breadth on the configurations that received full scoring.

    This is intentionally separate from :func:`breadth`: the cheap marker and
    baseline stores are exhaustive for their respective grids, while the final
    store is a finalist sample.  Combining them into one leaderboard obscures
    that distinction.
    """

    import capped_search as cs

    arms = {}
    for row in cs.read_rows(store / "final.jsonl"):
        if row.get("status") != "ok":
            continue
        arm = row.get("arm") or "flood fill"
        if cs._disqualified(row):
            continue
        arms.setdefault(arm, []).append(row)

    report = {"source": "final.jsonl", "exhaustive": False, "arms": {}}
    for name, rows in arms.items():
        recovered = np.array([r.get("recovered_at_90") or 0
                               for r in rows], dtype=float)
        if not recovered.size:
            continue
        best = float(recovered.max())
        entry = {
            "n_configurations": int(recovered.size),
            "best_recovered_at_90": int(best),
            "median_recovered_at_90": float(np.median(recovered)),
        }
        for level in BREADTH_LEVELS:
            entry[f"fraction_within_{int(level * 100)}pct_of_best"] = float(
                np.mean(recovered >= level * best)) if best else 0.0
        report["arms"][name] = entry
    return report


# ------------------------------------------------------------------ transfer


def strain_winners() -> dict:
    """Each strain's own best configuration, per arm."""

    import capped_search as cs

    out: dict[str, dict] = {}
    if not STRAIN_ROOT.exists():
        return out
    for directory in sorted(STRAIN_ROOT.iterdir()):
        final = directory / "final.jsonl"
        if not final.exists():
            continue
        rows = [r for r in cs.read_rows(final) if r.get("status") == "ok"]
        winners: dict[str, dict] = {}
        for row in rows:
            arm = row.get("arm") or "flood fill"
            if arm not in winners or cs.strict_recovery_key(row) < cs.strict_recovery_key(winners[arm]):
                winners[arm] = row
        if winners:
            out[directory.name] = winners
    return out


def _evaluate(task):
    """Apply one parameter set to one phantom and score it."""

    import capped_search as cs

    cs.init_worker(task["phantom"])
    params = dict(task["params"])
    arm = task["arm"]
    row = {"source_strain": task["source"], "target_strain": task["target"],
           "arm": arm}
    try:
        if arm == "KAM threshold":
            result = cs.task_baseline({**params})
        else:
            result = cs.task_finalise({**params, "arm": arm})
        row.update({k: result.get(k) for k in
                    ("n_cells_pred", "n_cells_true", "recovered_at_90",
                     "recovery_rate_at_90", "contamination", "fused_true_cells",
                     "ari", "status")})
    except Exception as error:                      # noqa: BLE001
        row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    return row


def transfer(workers: int = 14) -> dict:
    """Apply every strain's optimum to every strain, and measure the loss."""

    from concurrent.futures import ProcessPoolExecutor

    import strain_phantoms

    winners = strain_winners()
    if not winners:
        print("  no finished strain searches yet -- skipping transfer")
        return {}

    keys = [k for k in strain_phantoms.STRAIN_TREND if k in winners]
    tasks = []
    for source in keys:
        for arm, row in winners[source].items():
            names = (("percentile", "kam_radius_um", "min_cell_size", "connectivity")
                     if arm == "KAM threshold" else
                     ("local_threshold_deg", "global_threshold_deg",
                      "footprint_tolerance", "footprint_radius_um",
                      "min_cell_size", "kam_radius_um", "seed",
                      "merge_size_voxels", "merge_threshold_deg"))
            params = {k: row[k] for k in names if k in row}
            for target in keys:
                tasks.append({
                    "source": source, "target": target, "arm": arm,
                    "params": params,
                    "phantom": str(HERE / "phantoms" /
                                   f"strain_{target}_r0.npz"),
                })
    print(f"  {len(tasks)} transfer evaluations "
          f"({len(keys)} strains x {len(keys)} targets x arms)")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_evaluate, tasks))
    return {"rows": rows, "strains": keys}


def transfer_report(payload: dict) -> None:
    """Print the transfer matrix and the penalty for using foreign parameters."""

    import pandas as pd

    rows = [r for r in payload.get("rows", []) if r.get("status") == "ok"]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "transfer_rows.csv", index=False)
    for arm, group in frame.groupby("arm"):
        matrix = group.pivot_table(index="source_strain", columns="target_strain",
                                   values="recovery_rate_at_90")
        print(f"\n  recovery rate at 90 %, {arm} "
              "(rows: parameters from; columns: applied to)")
        print(matrix.round(3).to_string())
        own = np.array([matrix.loc[s, s] for s in matrix.index if s in matrix.columns])
        foreign = matrix.to_numpy().copy()
        np.fill_diagonal(foreign, np.nan)
        penalty = np.nanmean(own) - np.nanmean(foreign)
        print(f"  mean loss from using another strain's parameters: {penalty:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["breadth", "transfer", "all"])
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    if args.action in ("breadth", "all"):
        print("=== parameter-space breadth (can it be tuned blind?) ===")
        report["breadth"] = breadth()
        report["breadth_finalised"] = finalised_breadth()

    if args.action in ("transfer", "all"):
        print("\n=== cross-strain parameter transfer ===")
        payload = transfer(args.workers)
        if payload:
            transfer_report(payload)
            report["transfer"] = payload

    (OUT / "generalisation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(f"\nwritten: {OUT/'generalisation.json'}")


if __name__ == "__main__":
    main()

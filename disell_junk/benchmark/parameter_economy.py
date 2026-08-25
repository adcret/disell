#!/usr/bin/env python3
"""How many of these parameters are real knobs?

A method with seven parameters and a method with one are not equally usable on
data that has no ground truth, even at identical accuracy.  This measures how
many of the flood-fill arm's axes actually have to be chosen.

Two measurements, and the second is the one that counts:

``marginal``   Freeze one axis at a single global value, re-tune every other
               axis per strain, and record the loss in recovery rate at
               tau = 0.9.  Cheap, because it reuses the completed searches --
               but freezing each axis separately says nothing about freezing
               them together, so on its own it proves nothing.

``joint``      Freeze *every* axis at its default at once and sweep only the
               local threshold, against the full search.  This cannot be read
               off the completed searches: the staged funnel keeps only what
               leads at each stage, so the frozen combination is absent from
               ``final.jsonl`` entirely.  It has to be run --
               ``capped_search.py --frozen`` -- and ``run`` does that here.

Recovery *rate* rather than count, because the four strain phantoms hold
different numbers of true cells (1,481 to 2,525) and the raw counts are not
comparable across them.

Usage::

    python parameter_economy.py run      --workers 14   # the frozen searches
    python parameter_economy.py report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "analysis"
FULL_ROOT = HERE / "runs/strain"
FROZEN_ROOT = HERE / "runs/frozen"
PHANTOM_DIR = HERE / "phantoms"

STRAINS = ("2p4", "3p5", "4p6", "6p2")
ARMS = ("flood fill", "flood fill + merge")

#: The primary phantom is where sections 1-4 are measured, so the defaults have
#: to be priced there too and not only on the strain series.
PRIMARY_FULL = HERE / "runs/primary"
PRIMARY_FROZEN = HERE / "runs/frozen" / "primary"

#: The KAM arm's axes.  Its baseline store carries full scoring for every one
#: of its 9,108 configurations, so unlike the flood-fill arm -- whose marginal
#: table is read off the funnelled finalists -- this one is measured on the
#: complete grid.
KAM_AXES = ("percentile", "kam_radius_um", "min_cell_size", "connectivity")

#: Every axis the marker and merge stages expose.
AXES = ("footprint_radius_um", "footprint_tolerance", "local_threshold_deg",
        "global_threshold_deg", "min_cell_size", "kam_radius_um",
        "merge_size_voxels", "merge_threshold_deg")

#: The frozen run needs no kam stage -- the radius is one of the pinned axes --
#: and no baseline stage, which searches the KAM arm rather than this one.
FROZEN_STAGES = ("markers", "merge", "finalise")


def qualified(root: Path, strain: str, arm: str) -> list[dict]:
    """Scored rows for one arm that the policy is willing to accept."""

    import capped_search as cs

    rows = cs.read_rows(root / strain / "final.jsonl")
    return [r for r in rows
            if r.get("status") == "ok"
            and (r.get("arm") or "flood fill") == arm
            and not cs._disqualified(r)]


def best_rate(rows: list[dict]) -> float:
    return max(r["recovery_rate_at_90"] for r in rows)


# --------------------------------------------------------------- measurements

def marginal(arm: str) -> dict:
    """Cost of freezing each axis alone, with everything else re-tuned."""

    pool = {s: qualified(FULL_ROOT, s, arm) for s in STRAINS}
    if not all(pool.values()):
        return {}
    ceiling = {s: best_rate(pool[s]) for s in STRAINS}
    tuned = {}
    for s in STRAINS:
        import capped_search as cs
        tuned[s] = min(pool[s], key=cs.strict_recovery_key)

    out = {}
    for axis in AXES:
        values = sorted({r[axis] for s in STRAINS for r in pool[s]
                         if r.get(axis) is not None
                         and not (isinstance(r[axis], float) and np.isnan(r[axis]))})
        if len(values) <= 1:
            continue
        priced = {}
        for value in values:
            loss = {}
            for s in STRAINS:
                hit = [r["recovery_rate_at_90"] for r in pool[s] if r.get(axis) == value]
                # An axis value that no strain reached cannot be a global
                # default, so it is not a candidate -- not a zero-cost one.
                loss[s] = ceiling[s] - max(hit) if hit else None
            if all(v is not None for v in loss.values()):
                priced[value] = loss
        if not priced:
            continue
        chosen = min(priced, key=lambda v: float(np.mean(list(priced[v].values()))))
        out[axis] = {
            "values_searched": len(values),
            "tuned_per_strain": {s: tuned[s].get(axis) for s in STRAINS},
            "best_fixed": chosen,
            "mean_cost": float(np.mean(list(priced[chosen].values()))),
            "worst_cost": float(max(priced[chosen].values())),
            "per_strain_cost": {s: float(v) for s, v in priced[chosen].items()},
        }
    return out


def joint(arm: str) -> dict:
    """Cost of freezing every axis at once, sweeping only the threshold."""

    import capped_search as cs

    pairs = [(s, FULL_ROOT / s, FROZEN_ROOT / s) for s in STRAINS]
    pairs.append(("primary", PRIMARY_FULL, PRIMARY_FROZEN))

    rows = []
    for s, full_dir, frozen_dir in pairs:
        full = qualified(full_dir.parent, full_dir.name, arm)
        frozen = qualified(frozen_dir.parent, frozen_dir.name, arm)
        if not full or not frozen:
            continue
        best_full = min(full, key=cs.strict_recovery_key)
        best_frozen = min(frozen, key=cs.strict_recovery_key)
        rows.append({
            "strain_key": s,
            "full_search": best_full["recovery_rate_at_90"],
            "frozen_sweep": best_frozen["recovery_rate_at_90"],
            "cost": best_full["recovery_rate_at_90"] - best_frozen["recovery_rate_at_90"],
            "threshold": best_frozen["local_threshold_deg"],
            "configurations_full": len(full),
            "configurations_frozen": len(frozen),
        })
    if not rows:
        return {}

    # And the cost of not sweeping even the threshold: the best single value
    # across all four strains, under the same frozen defaults.
    pool = {s: qualified(FROZEN_ROOT, s, arm) for s in STRAINS}
    fixed = {}
    if all(pool.values()):
        ceiling = {s: best_rate(pool[s]) for s in STRAINS}
        priced = {}
        for value in sorted({r["local_threshold_deg"] for s in STRAINS for r in pool[s]}):
            loss = {}
            for s in STRAINS:
                hit = [r["recovery_rate_at_90"] for r in pool[s]
                       if r["local_threshold_deg"] == value]
                loss[s] = ceiling[s] - max(hit) if hit else None
            if all(v is not None for v in loss.values()):
                priced[value] = loss
        if priced:
            chosen = min(priced, key=lambda v: float(np.mean(list(priced[v].values()))))
            fixed = {
                "threshold": chosen,
                "mean_cost": float(np.mean(list(priced[chosen].values()))),
                "worst_cost": float(max(priced[chosen].values())),
            }

    return {
        "per_strain": rows,
        "mean_cost": float(np.mean([r["cost"] for r in rows])),
        "worst_cost": float(max(r["cost"] for r in rows)),
        "threshold_also_fixed": fixed,
    }


def marginal_kam() -> dict:
    """The same freezing question asked of the KAM arm, on its whole grid.

    Section 6 would be one-sided if only the flood-fill arm were asked how many
    of its parameters matter.  The KAM baseline store scores every
    configuration it evaluates, so this needs no new computation and, unlike
    the flood-fill table, is not restricted to finalists.
    """

    import capped_search as cs

    pool = {}
    for strain in STRAINS:
        rows = [r for r in cs.read_rows(FULL_ROOT / strain / "baseline.jsonl")
                if r.get("status") == "ok" and not cs._disqualified(r)]
        if rows:
            pool[strain] = rows
    if len(pool) < len(STRAINS):
        return {}

    ceiling = {s: max(r["recovery_rate_at_90"] for r in rows)
               for s, rows in pool.items()}
    out = {}
    for axis in KAM_AXES:
        values = sorted({r[axis] for rows in pool.values() for r in rows
                         if r.get(axis) is not None})
        if len(values) <= 1:
            continue
        priced = {}
        for value in values:
            loss = {}
            for s, rows in pool.items():
                hit = [r["recovery_rate_at_90"] for r in rows if r.get(axis) == value]
                loss[s] = ceiling[s] - max(hit) if hit else None
            if all(v is not None for v in loss.values()):
                priced[value] = loss
        if not priced:
            continue
        chosen = min(priced, key=lambda v: float(np.mean(list(priced[v].values()))))
        tuned = {}
        for s, rows in pool.items():
            tuned[s] = max(rows, key=lambda r: r["recovery_rate_at_90"]).get(axis)
        out[axis] = {
            "values_searched": len(values),
            "tuned_per_strain": tuned,
            "best_fixed": chosen,
            "mean_cost": float(np.mean(list(priced[chosen].values()))),
            "worst_cost": float(max(priced[chosen].values())),
            "configurations": sum(len(r) for r in pool.values()),
        }
    return out


# --------------------------------------------------------------------- report

def show(report: dict) -> None:
    for arm in list(ARMS) + ["KAM threshold"]:
        block = report.get(arm, {})
        print(f"\n=== {arm} ===")
        marg = block.get("marginal", {})
        if marg:
            print(f"  {'axis':24s} {'values':>7s} {'best fixed':>12s} "
                  f"{'mean cost':>10s} {'worst':>8s}")
            for axis, row in sorted(marg.items(), key=lambda kv: kv[1]["mean_cost"]):
                print(f"  {axis:24s} {row['values_searched']:>7d} "
                      f"{str(row['best_fixed']):>12s} {row['mean_cost']:>10.4f} "
                      f"{row['worst_cost']:>8.4f}")
        j = block.get("joint", {})
        if j:
            print(f"\n  frozen defaults, sweeping only the local threshold:")
            print(f"  {'strain':>8s} {'full':>8s} {'frozen':>8s} {'cost':>8s} {'thr*':>10s}")
            for row in j["per_strain"]:
                print(f"  {row['strain_key']:>8s} {row['full_search']:>8.4f} "
                      f"{row['frozen_sweep']:>8.4f} {row['cost']:>8.4f} "
                      f"{row['threshold']:>10.6f}")
            print(f"  mean cost {j['mean_cost']:.4f}, worst {j['worst_cost']:.4f}")
            if j.get("threshold_also_fixed"):
                f = j["threshold_also_fixed"]
                print(f"  fixing the threshold too ({f['threshold']:.6f}): "
                      f"mean {f['mean_cost']:.4f}, worst {f['worst_cost']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "report", "all"])
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    if args.action in ("run", "all"):
        for strain in STRAINS:
            phantom = PHANTOM_DIR / f"strain_{strain}_r0.npz"
            if not phantom.exists():
                raise SystemExit(f"missing phantom {phantom}")
            for stage in FROZEN_STAGES:
                print(f"\n=== frozen {strain}: {stage} ===", flush=True)
                subprocess.run(
                    [sys.executable, str(HERE / "capped_search.py"), stage,
                     "--frozen", "--workers", str(args.workers),
                     "--phantom", str(phantom),
                     "--out", str(FROZEN_ROOT / strain)],
                    check=True)

    if args.action in ("report", "all"):
        import capped_search as cs

        report = {arm: {"marginal": marginal(arm), "joint": joint(arm)}
                  for arm in ARMS}
        report["KAM threshold"] = {"marginal": marginal_kam()}
        report["defaults"] = {
            **cs.DEFAULTS,
            "min_cell_size": cs.DEFAULT_MIN_CELL_SIZE,
            "local_threshold_deg": cs.DEFAULT_LOCAL_THRESHOLD_DEG,
        }
        show(report)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "parameter_economy.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
        print(f"\nwritten: {OUT/'parameter_economy.json'}")


if __name__ == "__main__":
    main()

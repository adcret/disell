#!/usr/bin/env python3
"""Section 2's claim, repeated on independent volumes.

"The advantage is three-dimensional" is measured on one phantom, and the
2D+link baseline is a construction rather than a search -- so a single number
gives no sense of how much of it is the volume it happened to be measured on.
This repeats the comparison on independent realisations of the primary phantom
and reports the spread.

Parameters are the study defaults, not per-volume optima, matching the
protocol used for the replicates elsewhere.

Usage::

    python dimension_replicates.py run
    python dimension_replicates.py report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "analysis"
PHANTOM_DIR = HERE / "phantoms"
PRIMARY_NAME = "primary_6p2_consistent"
REPLICATES = (0, 1, 2, 3, 4, 5)
ARMS = ("flood fill", "KAM threshold")


def phantom_path(rep: int) -> Path:
    suffix = "" if rep == 0 else f"_r{rep}"
    return PHANTOM_DIR / f"{PRIMARY_NAME}{suffix}.npz"


def parameters():
    """Defaults for the flood fill; the KAM arm's own tuned settings."""

    import capped_search as cs
    import why_flood_fill as wf

    _, kam_params = wf.best_parameters()
    flood_params = {
        "local_threshold_deg": cs.DEFAULT_LOCAL_THRESHOLD_DEG["flood fill"],
        "global_threshold_deg": cs.DEFAULTS["global_threshold_deg"],
        "footprint_tolerance": cs.DEFAULTS["footprint_tolerance"],
        "footprint_radius_um": cs.DEFAULTS["footprint_radius_um"],
        "min_cell_size": cs.DEFAULT_MIN_CELL_SIZE["flood fill"],
        "kam_radius_um": cs.DEFAULTS["kam_radius_um"],
    }
    return flood_params, kam_params


def run() -> list[dict]:
    import why_flood_fill as wf

    flood_params, kam_params = parameters()
    records = []
    for rep in REPLICATES:
        path = phantom_path(rep)
        if not path.exists():
            print(f"skipping r{rep}: no phantom", flush=True)
            continue
        print(f"\n=== primary r{rep} ===", flush=True)
        payload = wf.dimension_analysis(flood_params, kam_params, phantom_path=path)
        for arm in ARMS:
            entry = payload.get(arm)
            if not entry:
                continue
            records.append({
                "replicate": rep, "arm": arm,
                "recovered_2d": entry["2D+link"]["recovered_at_90"],
                "recovered_3d": entry["3D"]["recovered_at_90"],
                "rate_2d": entry["2D+link"]["recovery_rate"],
                "rate_3d": entry["3D"]["recovery_rate"],
                "gain_cells": entry["gain_from_3d_cells"],
                "unlinked_2d_cells": entry["unlinked_2d_cells"],
            })
    return records


def summarise(records: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        if not rows:
            continue
        def stat(key):
            values = [r[key] for r in rows]
            return {"mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(min(values)), "max": float(max(values))}
        out[arm] = {"n": len(rows), "recovered_2d": stat("recovered_2d"),
                    "recovered_3d": stat("recovered_3d"),
                    "rate_2d": stat("rate_2d"), "rate_3d": stat("rate_3d"),
                    "gain_cells": stat("gain_cells")}
    return out


def show(report):
    for arm, e in (report.get("summary") or {}).items():
        print(f"\n=== {arm} (n = {e['n']}) ===")
        print(f"  2D+link recovered  {e['recovered_2d']['mean']:8.1f} "
              f"+- {e['recovered_2d']['sd']:.1f}")
        print(f"  3D recovered       {e['recovered_3d']['mean']:8.1f} "
              f"+- {e['recovered_3d']['sd']:.1f}")
        print(f"  gain from 3D       {e['gain_cells']['mean']:8.1f} "
              f"+- {e['gain_cells']['sd']:.1f} cells "
              f"(range {e['gain_cells']['min']:.0f}-{e['gain_cells']['max']:.0f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "report"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "dimension_replicates.json"

    if args.action == "run":
        records = run()
        report = {"records": records, "summary": summarise(records)}
        path.write_text(json.dumps(report, indent=2, sort_keys=True,
                                   default=float) + "\n")
    else:
        report = json.loads(path.read_text()) if path.exists() else {}
    show(report)
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()

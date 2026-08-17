#!/usr/bin/env python3
"""Local cell-count sensitivity around the validated count-first winner."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from count_first_analysis import PARAMETERS

HERE = Path(__file__).resolve().parent
SOURCE_AUDIT = HERE / "two_stage_oracle_count_results" / "implementation_audit.json"
OUT = HERE / "analysis" / "count_first_v1" / "local_oat"
STAGE = "count_first_local_oat_v1"
CENTRE_HASH = "874bacd8112537a30267aeda6ee7b800ca917081b4abde13a1d0aed0c17c825c"
CENTRE = {
    "local_threshold_deg": 0.006804162534344432,
    "global_threshold_deg": 0.8729926391777059,
    "footprint_tolerance": 0.04,
    "footprint_radius_um": 1.948897380571898,
    "min_cell_size": 116,
    "kam_radius_um": 1.1737337858305015,
}
BOUNDS = {
    "local_threshold_deg": (10 ** -2.7, 10 ** -0.8),
    "global_threshold_deg": (10 ** -2.1, 10 ** -0.1),
    "footprint_tolerance": (0.04, 0.96),
    "footprint_radius_um": (0.3, 4.0),
    "min_cell_size": (5, 250),
    "kam_radius_um": (0.3, 2.0),
}
LOG_PARAMETERS = {
    "local_threshold_deg",
    "global_threshold_deg",
    "footprint_radius_um",
    "kam_radius_um",
}
N_POINTS = 11


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def local_grid(name: str, centre: float) -> list:
    low, high = BOUNDS[name]
    span_low = max(low, 0.5 * float(centre))
    span_high = min(high, 2.0 * float(centre))
    if name == "min_cell_size":
        values = np.unique(np.rint(np.linspace(span_low, span_high, N_POINTS)).astype(int))
        values = np.unique(np.append(values, int(centre)))
        return [int(value) for value in values]
    values = (
        np.geomspace(span_low, span_high, N_POINTS)
        if name in LOG_PARAMETERS
        else np.linspace(span_low, span_high, N_POINTS)
    )
    if not np.any(np.isclose(values, centre, rtol=1e-10, atol=1e-12)):
        values = np.sort(np.append(values, centre))
    return [float(value) for value in values]


def signature(row: dict) -> tuple:
    return tuple(row[name] for name in PARAMETERS)


def prepare_design(workspace, tso) -> tuple[dict, dict, list[dict]]:
    centre = tso.canonical(dict(CENTRE), workspace)
    configurations = []
    seen = set()
    axes = {}
    for name in PARAMETERS:
        canonical_values = []
        for raw_value in local_grid(name, centre[name]):
            candidate = dict(centre)
            candidate[name] = int(raw_value) if name == "min_cell_size" else float(raw_value)
            candidate = tso.canonical(candidate, workspace)
            if candidate[name] not in canonical_values:
                canonical_values.append(candidate[name])
            key = signature(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidate["_sampling"] = {
                "experiment": STAGE,
                "oat_parameter": name,
                "centre_config_hash": CENTRE_HASH,
                "is_centre": key == signature(centre),
            }
            configurations.append(candidate)
        axes[name] = {"centre": centre[name], "canonical_values": canonical_values}
    design = {
        "experiment": STAGE,
        "centre_seed0_hash": CENTRE_HASH,
        "centre": centre,
        "bounds": {name: list(bounds) for name, bounds in BOUNDS.items()},
        "axes": axes,
        "n_unique_configurations": len(configurations),
        "response": "recovered cell count; target 360",
    }
    atomic(OUT / "design.json", design)
    return design, centre, configurations


def analyse(design: dict, centre: dict, rows: list[dict]) -> dict:
    import pandas as pd

    records = []
    axis_summary = []
    for name in PARAMETERS:
        points = []
        for row in rows:
            sampling = row.get("sampling_metadata") or {}
            if sampling.get("oat_parameter") != name and signature(row) != signature(centre):
                continue
            if row.get("status_category") != "ok":
                continue
            error = abs(int(row["n_cells_pred"]) - int(row["n_cells_true"]))
            points.append({
                "parameter": name,
                "value": row[name],
                "n_cells_pred": int(row["n_cells_pred"]),
                "absolute_count_error": error,
                "within_1_percent": error / int(row["n_cells_true"]) <= 0.01,
                "identity_f1": row.get("identity_f1"),
                "ari": row.get("ari"),
                "boundary_assd_um": row.get("boundary_assd_um"),
            })
        points.sort(key=lambda point: point["value"])
        unique = []
        for point in points:
            if not unique or not np.isclose(
                point["value"], unique[-1]["value"], rtol=1e-10, atol=1e-12
            ):
                unique.append(point)
        points = unique
        records.extend(points)
        exact_values = [point["value"] for point in points if point["absolute_count_error"] == 0]
        near_values = [point["value"] for point in points if point["within_1_percent"]]
        axis_summary.append({
            "parameter": name,
            "tested_points": len(points),
            "exact_count_points": len(exact_values),
            "exact_count_fraction": len(exact_values) / len(points),
            "exact_count_low": min(exact_values) if exact_values else None,
            "exact_count_high": max(exact_values) if exact_values else None,
            "within_1_percent_points": len(near_values),
            "within_1_percent_low": min(near_values) if near_values else None,
            "within_1_percent_high": max(near_values) if near_values else None,
            "maximum_absolute_count_error": max(point["absolute_count_error"] for point in points),
        })
    pd.DataFrame(records).to_csv(OUT / "profiles.csv", index=False)
    pd.DataFrame(axis_summary).to_csv(OUT / "axis_summary.csv", index=False)
    result = {
        "experiment": STAGE,
        "centre": centre,
        "target_cells": 360,
        "n_durable_rows": len(rows),
        "axis_summary": axis_summary,
        "interpretation": (
            "Exact-count fractions are one-at-a-time local robustness measures. "
            "They do not include parameter interactions."
        ),
    }
    atomic(OUT / "summary.json", result)
    return result


def main() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["MALLOC_ARENA_MAX"] = "2"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/disell-count-first-oat-mpl")
    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "implementation_audit.json"
    if not audit_path.exists():
        shutil.copyfile(SOURCE_AUDIT, audit_path)
    os.environ["DISELL_TWO_STAGE_OUT"] = str(OUT)
    import sys

    sys.path.insert(0, str(HERE))
    import oracle_core as oc
    import two_stage_oracle as tso

    tso.OUT = OUT
    tso.IDENTITY = json.loads(audit_path.read_text())
    workspace = oc.load_workspace(tso.CACHE)
    design, centre, configurations = prepare_design(workspace, tso)
    store = OUT / "evaluations.jsonl"
    tso.run_tasks(configurations, STAGE, (0,), 8, store)
    rows = tso.strict_jsonl(store)[1]
    result = analyse(design, centre, rows)
    print(json.dumps({
        "rows": result["n_durable_rows"],
        "exact_count_points": {
            row["parameter"]: row["exact_count_points"] for row in result["axis_summary"]
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

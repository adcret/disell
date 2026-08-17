#!/usr/bin/env python3
"""Evaluate saved cross-validated parameter rules without rerunning any search."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_core as oc
import oracle_runner
import oracle_select as osel

OUT = HERE / "continuation_results"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    with temporary.open("r+") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return value


def main() -> int:
    cv = pd.read_csv(OUT / "transferable_rule_cross_validation.csv")
    pivot = cv.pivot_table(
        index=["scheme", "held_out_group", "phantom_id"],
        columns="parameter", values="rule_value", aggfunc="first"
    ).reset_index()
    records = []
    for _, prediction in pivot.iterrows():
        scheme = str(prediction.scheme)
        phantom_id = str(prediction.phantom_id)
        checkpoint = OUT / "rule_evaluations" / scheme / f"{phantom_id}.json"
        if checkpoint.exists():
            records.append(json.loads(checkpoint.read_text()))
            continue
        path = OUT / "phantoms" / phantom_id
        workspace = oc.load_workspace(path / "cache")
        raw = {name: float(prediction[name]) for name in osel.PARAMETER_NAMES
               if name != "footprint_tolerance"}
        # This parameter was intentionally excluded from the fitted targets;
        # use the prespecified primary starting value, not any phantom label.
        raw["footprint_tolerance"] = 0.1837
        raw["min_cell_size"] = max(5, int(round(raw["min_cell_size"])))
        config = oc.canonical(oc.Config(**raw), workspace)
        result = next(iter(oracle_runner.evaluate_batch(
            [(config.as_dict(), 0)], path / "cache", processes=1,
            progress_every=0, label=f"{scheme} {phantom_id} "
        )))
        if result.get("status") != "ok":
            raise RuntimeError(f"rule evaluation failed: {result}")
        result.update(scheme=scheme, held_out_group=plain(prediction.held_out_group),
                      phantom_id=phantom_id, stage="rule_validation",
                      footprint_tolerance_source="fixed_primary_0.1837")
        atomic_json(checkpoint, plain(result))
        records.append(result)
        print(f"checkpointed {scheme} {phantom_id}", flush=True)
    pd.DataFrame(records).sort_values(["scheme", "phantom_id"]).to_csv(
        OUT / "transferable_rule_performance.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

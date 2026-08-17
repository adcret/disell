#!/usr/bin/env python3
"""Replay one uncheckpointed candidate and record raw non-finite diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_core as oc
import two_stage_oracle as tso


def scan(value, path="result"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from scan(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from scan(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        yield {
            "field": path,
            "value": "NaN" if np.isnan(value) else (
                "Infinity" if value > 0 else "-Infinity"
            ),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_index", type=int)
    args = parser.parse_args()
    tso.IDENTITY = tso.identity()
    tso.WORK = oc.load_workspace(tso.CACHE)
    sampled = tso.broad(8000, tso.WORK)[args.sample_index]
    parameters = {key: value for key, value in sampled.items()
                  if not key.startswith("_")}
    started = time.time()
    raw = tso._evaluate_task({
        "parameters": parameters, "seed": 0, "stage": "broad",
        "boundary": True,
    })
    report = {
        "sample_index": args.sample_index,
        "config_hash": raw.get("config_hash"),
        "config_key": raw.get("config_key"),
        "parameters": parameters,
        "stage": raw.get("stage"),
        "seed": raw.get("candidate_order_seed"),
        "worker_status": raw.get("status"),
        "elapsed_seconds": time.time() - started,
        "nonfinite": list(scan(raw)),
        "one_to_one_recovered_cells": raw.get("one_to_one_recovered_cells"),
        "predicted_cell_count": raw.get("n_cells_pred"),
        "accepted_marker_count": raw.get("accepted_marker_count"),
    }
    path = tso.OUT / f"nonfinite_reconstruction_{args.sample_index}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

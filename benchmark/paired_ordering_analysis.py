#!/usr/bin/env python3
"""Summarise the matched size-ordering ablation without rerunning segmentation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRIALS = (
    HERE / "continuation_results" / "algorithm_comparison" / "paired_trials.jsonl"
)
OUT = HERE / "analysis" / "paired_ordering_v1"
RANDOM = "random_order_multiseed_v1"
ORDERED = "size_prioritised_multiseed_v1"
PARAMETERS = (
    "footprint_radius_um",
    "footprint_tolerance",
    "global_threshold_deg",
    "kam_radius_um",
    "local_threshold_deg",
    "min_cell_size",
)


def _direction(delta: pd.Series, *, lower_is_better: bool) -> dict[str, int]:
    values = delta.to_numpy(dtype=float)
    equal = np.isclose(values, 0.0, rtol=0.0, atol=1e-12)
    better = values < 0 if lower_is_better else values > 0
    worse = values > 0 if lower_is_better else values < 0
    return {
        "ordered_better": int(np.sum(better & ~equal)),
        "equal": int(np.sum(equal)),
        "ordered_worse": int(np.sum(worse & ~equal)),
    }


def summarize(frame: pd.DataFrame, seeds=range(5)) -> tuple[dict, pd.DataFrame]:
    """Return a balanced paired summary and per-configuration effects."""
    data = frame.copy()
    data["identity_f1"] = (
        2 * data["one_to_one_recovered_cells"]
        / (data["n_true_cells"] + data["n_predicted_cells"])
    )
    data["orientation_f1_at_0p02deg"] = (
        2 * data["orientation_correct_cells_at_0p02deg"]
        / (data["n_true_cells"] + data["n_predicted_cells"])
    )
    cohort = data[
        data["candidate_order_seed"].isin(list(seeds))
        & data["status_category"].eq("ok")
    ].copy()
    metrics = (
        "absolute_cell_count_error",
        "identity_f1",
        "orientation_f1_at_0p02deg",
        "ari",
        "vi_total_bits",
        "boundary_assd_um",
    )
    paired = cohort.pivot(
        index="pairing_key", columns="algorithm_id", values=list(metrics)
    )
    required = [(metric, algorithm) for metric in metrics for algorithm in (RANDOM, ORDERED)]
    paired = paired.dropna(subset=required)
    complete_keys = set(paired.index)
    cohort = cohort[cohort["pairing_key"].isin(complete_keys)]

    pair_effects = {}
    for metric in metrics:
        random_values = paired[(metric, RANDOM)].astype(float)
        ordered_values = paired[(metric, ORDERED)].astype(float)
        delta = ordered_values - random_values
        pair_effects[metric] = {
            "random_median": float(random_values.median()),
            "ordered_median": float(ordered_values.median()),
            "median_paired_delta": float(delta.median()),
            **_direction(
                delta,
                lower_is_better=metric
                in {"absolute_cell_count_error", "vi_total_bits", "boundary_assd_um"},
            ),
        }

    aggregation = {
        "absolute_cell_count_error": lambda values: values.median(),
        "identity_f1": "median",
        "orientation_f1_at_0p02deg": "median",
        "ari": "median",
        "vi_total_bits": "median",
        "boundary_assd_um": "median",
    }
    by_config = (
        cohort.groupby(["algorithm_id", *PARAMETERS], dropna=False)
        .agg(aggregation)
        .reset_index()
    )
    wide = by_config.pivot(
        index=list(PARAMETERS),
        columns="algorithm_id",
        values=list(aggregation),
    ).dropna()
    effects = wide.index.to_frame(index=False)
    config_effects = {}
    for metric in aggregation:
        delta = wide[(metric, ORDERED)] - wide[(metric, RANDOM)]
        effects[f"random_median__{metric}"] = wide[(metric, RANDOM)].to_numpy()
        effects[f"ordered_median__{metric}"] = wide[(metric, ORDERED)].to_numpy()
        effects[f"ordered_minus_random__{metric}"] = delta.to_numpy()
        config_effects[metric] = {
            "median_delta": float(delta.median()),
            **_direction(
                delta,
                lower_is_better=metric
                in {"absolute_cell_count_error", "vi_total_bits", "boundary_assd_um"},
            ),
        }

    planned = data[data["candidate_order_seed"].isin(list(seeds))]
    planned_configurations = int(planned[list(PARAMETERS)].drop_duplicates().shape[0])
    summary = {
        "comparison": "matched size-ordering ablation on the synthetic phantom",
        "cohort": {
            "seed_orders": list(map(int, seeds)),
            "planned_configurations": planned_configurations,
            "usable_configurations": int(len(effects)),
            "excluded_configurations": planned_configurations - int(len(effects)),
            "complete_pairs": int(len(paired)),
            "exclusion_rule": "exclude settings without successful results for both algorithms",
        },
        "pair_level": pair_effects,
        "configuration_medians": config_effects,
        "interpretation": (
            "At identical parameters and seed order, size ordering usually reduces "
            "cell-count error and improves correct-cell identity. Exact-count rates "
            "at independently selected settings remain a separate selection result."
        ),
    }
    return summary, effects


def main() -> None:
    frame = pd.DataFrame(
        json.loads(line) for line in TRIALS.read_text().splitlines() if line.strip()
    )
    summary, effects = summarize(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paired_ordering_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    effects.to_csv(OUT / "paired_configuration_effects.csv", index=False)
    count = summary["configuration_medians"]["absolute_cell_count_error"]
    identity = summary["configuration_medians"]["identity_f1"]
    cohort = summary["cohort"]
    report = [
        "# Matched size-ordering ablation",
        "",
        f"The balanced cohort contains {cohort['complete_pairs']} paired runs: "
        f"{cohort['usable_configurations']} parameter settings with five identical "
        "seed orders per algorithm.",
        "",
        f"- Median absolute count error was lower with size ordering for "
        f"{count['ordered_better']}/{cohort['usable_configurations']} settings, "
        f"equal for {count['equal']}, and higher for {count['ordered_worse']}.",
        f"- Median correct-cell F1 was higher with size ordering for "
        f"{identity['ordered_better']}/{cohort['usable_configurations']} settings, "
        f"equal for {identity['equal']}, and lower for {identity['ordered_worse']}.",
        f"- {cohort['excluded_configurations']} planned settings were excluded because "
        "one or both algorithms produced no valid segmentation.",
        "",
        "This matched ablation isolates ordering. The exact-count frequencies at the "
        "separately selected method settings answer a different, selection-level question.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

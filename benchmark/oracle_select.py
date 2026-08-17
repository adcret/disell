#!/usr/bin/env python3
"""Aggregation over seed orders, and the four solutions the study reports.

ARI is not an adequate objective on its own.  It counts agreeing voxel *pairs*,
so it is dominated by the large cells: cutting a 300-voxel cell in two costs it
almost nothing, while the partition gains a spurious object.  The published
operating point is the direct evidence -- 472 predicted cells against 360 true
ones at ARI 0.888, with a rival at ARI 0.890 and 100 more cells.

So four solutions are reported rather than one:

``max_ari``       the maximum-ARI point, for comparability with the old result;
``min_vi``        the minimum total variation of information;
``min_count``     the minimum absolute cell-count error;
``balanced``      the Pareto-knee point.

The balanced rule is fixed in advance: retain everything within 0.005 mean ARI
of the maximum, then score the retained set on total VI, false splits, absolute
cell-count error, boundary F1, ASSD and seed-order stability, in that order of
priority.  Priority is expressed as decreasing weights on the min-max normalised
criteria rather than as a strict lexicographic order, because a strict order is
decided entirely by the first criterion and would reduce to ``min_vi``.  The
strict-priority winner is computed as well and reported alongside, so the choice
of rule is visible rather than assumed.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

PARAMETER_NAMES = (
    "local_threshold_deg",
    "global_threshold_deg",
    "footprint_tolerance",
    "footprint_radius_um",
    "min_cell_size",
    "kam_radius_um",
)

#: Metrics aggregated over seed orders.
AGGREGATED = (
    "ari",
    "vi_total_bits",
    "vi_split_bits",
    "vi_merge_bits",
    "n_cells_pred",
    "cell_count_error",
    "matched_iou_mean",
    "matched_dice_mean",
    "object_f1_at_0p5",
    "object_f1_at_0p75",
    "object_precision_at_0p5",
    "object_recall_at_0p5",
    "object_precision_at_0p75",
    "object_recall_at_0p75",
    "panoptic_quality_at_0p5",
    "panoptic_quality_at_0p75",
    "true_cells_split",
    "excess_fragments_total",
    "excess_fragments_per_true_cell",
    "pred_cells_merging",
    "true_cells_unrepresented",
    "boundary_assd_um",
    "boundary_f1_at_0p4um",
    "boundary_f1_at_0p8um",
    "boundary_precision_at_0p4um",
    "boundary_recall_at_0p4um",
    "interface_precision",
    "false_internal_boundary_fraction",
    "true_facet_recall_area_weighted",
    "marker_marker_count",
    "marker_cells_unseeded",
    "marker_cells_with_multiple_markers",
    "marker_markers_covering_multiple_cells",
    "marker_excess_markers_total",
    "marker_interface_precision",
    "marker_percolating_marker_volume_fraction",
    "cells_added_by_watershed",
    "neighbour_requirement_full",
    "footprint_n_voxels",
    "kam_footprint_n_voxels",
    "segment_seconds",
)

#: The balanced rule: (column, direction, weight).  ``direction`` is +1 when
#: smaller is better.  Weights decrease in the stated order of priority.
BALANCED_CRITERIA = (
    ("vi_total_bits_mean", +1, 0.30),
    ("excess_fragments_total_mean", +1, 0.22),
    ("abs_cell_count_error", +1, 0.18),
    ("boundary_f1_at_0p4um_mean", -1, 0.12),
    ("boundary_assd_um_mean", +1, 0.10),
    ("ari_std", +1, 0.08),
)
ARI_BAND = 0.005


def aggregate(rows: Sequence[dict[str, Any]]):
    """Mean and standard deviation of every metric, per configuration.

    Returns a DataFrame indexed by ``config_key`` with ``<metric>_mean`` and
    ``<metric>_std`` columns, plus the parameters and the number of seed orders.
    """

    import pandas as pd

    frame = pd.DataFrame([row for row in rows if row.get("status") == "ok"])
    if frame.empty:
        return frame
    present = [name for name in AGGREGATED if name in frame.columns]
    grouped = frame.groupby("config_key")
    out = grouped[present].agg(["mean", "std"])
    out.columns = [f"{metric}_{statistic}" for metric, statistic in out.columns]
    for name in present:
        out[f"{name}_std"] = out[f"{name}_std"].fillna(0.0)
    parameters = grouped[list(PARAMETER_NAMES)].first()
    out = parameters.join(out)
    out["n_seeds"] = grouped.size()
    out["abs_cell_count_error"] = out["cell_count_error_mean"].abs()
    out["global_threshold_active"] = out["global_threshold_deg"] > 0
    return out.reset_index()


def _normalised(values: np.ndarray, direction: int) -> np.ndarray:
    """Min-max normalise so that 0 is best and 1 is worst within the set."""

    values = np.asarray(values, dtype=float) * float(direction)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def balanced_score(frame, criteria=BALANCED_CRITERIA) -> np.ndarray:
    """Weighted sum of normalised criteria; lower is better."""

    score = np.zeros(len(frame), dtype=float)
    weight_total = 0.0
    for column, direction, weight in criteria:
        if column not in frame.columns:
            continue
        score += weight * _normalised(frame[column].to_numpy(), direction)
        weight_total += weight
    return score / max(weight_total, 1e-12)


def strict_priority_winner(frame, criteria=BALANCED_CRITERIA):
    """Lexicographic winner under the same criteria, for comparison.

    Included because a strict order is in practice decided by its first
    criterion; showing it next to the weighted result makes that visible.
    """

    keys = []
    for column, direction, _ in criteria:
        if column in frame.columns:
            keys.append(direction * frame[column].to_numpy())
    if not keys:
        return None
    order = np.lexsort(tuple(reversed(keys)))
    return frame.iloc[order[0]]


def select_solutions(aggregated, ari_band: float = ARI_BAND) -> dict[str, Any]:
    """The four reported solutions, plus the retained near-optimal set."""

    import pandas as pd

    frame = aggregated[aggregated["n_seeds"] > 0].copy()
    if frame.empty:
        return {}

    best_ari = float(frame["ari_mean"].max())
    retained = frame[frame["ari_mean"] >= best_ari - ari_band].copy()
    retained["balanced_score"] = balanced_score(retained)

    solutions = {
        "max_ari": frame.loc[frame["ari_mean"].idxmax()],
        "min_vi": frame.loc[frame["vi_total_bits_mean"].idxmin()],
        "min_count": frame.loc[frame["abs_cell_count_error"].idxmin()],
        "balanced": retained.loc[retained["balanced_score"].idxmin()],
    }
    return {
        "solutions": {name: row.to_dict() for name, row in solutions.items()},
        "solution_rows": solutions,
        "retained": retained.sort_values("balanced_score"),
        "best_ari": best_ari,
        "ari_band": float(ari_band),
        "n_retained": int(len(retained)),
        "strict_priority": (
            strict_priority_winner(retained).to_dict()
            if len(retained) else None
        ),
    }


def pareto_front(frame, columns: Sequence[tuple[str, int]]):
    """Non-dominated rows.  ``columns`` is ``(name, +1 if smaller is better)``."""

    values = np.column_stack(
        [float(d) * frame[c].to_numpy(dtype=float) for c, d in columns]
    )
    keep = np.ones(len(frame), dtype=bool)
    order = np.argsort(values[:, 0], kind="stable")
    for position, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[position + 1:]:
            if not keep[j]:
                continue
            if np.all(values[i] <= values[j]) and np.any(values[i] < values[j]):
                keep[j] = False
    return frame.iloc[np.nonzero(keep)[0]]


def describe_solution(row: dict[str, Any]) -> str:
    """One compact line per solution, for the console and the report."""

    return (
        f"local={row['local_threshold_deg']:.5g} "
        f"global={'off' if row['global_threshold_deg'] <= 0 else format(row['global_threshold_deg'], '.4g')} "
        f"tol={row['footprint_tolerance']:.4g}(req {row.get('neighbour_requirement_full_mean', float('nan')):.0f}) "
        f"fp_r={row['footprint_radius_um']:.4g} "
        f"min_size={int(row['min_cell_size'])} "
        f"kam_r={row['kam_radius_um']:.4g} "
        f"| ARI {row['ari_mean']:.4f} VI {row['vi_total_bits_mean']:.3f} "
        f"cells {row['n_cells_pred_mean']:.0f} "
        f"splits {row.get('excess_fragments_total_mean', float('nan')):.0f} "
        f"bF1 {row.get('boundary_f1_at_0p4um_mean', float('nan')):.3f}"
    )

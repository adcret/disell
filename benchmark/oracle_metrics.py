#!/usr/bin/env python3
"""Three-level evaluation of a predicted partition against known ground truth.

The existing :mod:`bench_metrics` scores a partition with ARI, VI, boundary
ASSD and a cell count.  That is not enough to choose parameters: ARI is
dominated by the large cells, so a partition can shed 0.002 of it while
gaining a hundred spurious fragments.  This module adds the two levels that
actually see that failure.

**Voxel partition** -- ARI, and VI split / merge / total.  ``split`` is
``H(pred | truth)`` and is the over-segmentation term.

**Object level** -- Hungarian-matched IoU and Dice, object precision and
recall at IoU 0.5 and 0.75, panoptic quality, and the two counts that name the
failure directly: how many true cells are broken into several *substantial*
predicted fragments, and how many predicted cells swallow several true cells.
"Substantial" is a fraction of the containing object rather than a fixed voxel
count, so it means the same thing for a 200-voxel cell and a 6000-voxel one.

**Boundary** -- anisotropy-aware ASSD, boundary precision / recall / F1 at
physical tolerances, area-weighted true-facet recall, and predicted-interface
precision.  The last one is the counterpart the previous facet analysis was
missing: true-facet recall alone rewards over-segmentation, because a partition
that changes label everywhere recovers every facet.  Predicted-interface
precision asks the opposite question -- what fraction of the interface area the
prediction draws corresponds to a real facet -- and an interface between two
predicted cells that map to the *same* true cell is counted as a false internal
boundary.

Every distance and area is in physical units, using the anisotropic voxel
spacing, so a one-voxel error along the coarse axis costs what it should.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: A predicted fragment counts against a true cell when it claims at least this
#: fraction of it, and vice versa.  Fractional rather than absolute so the test
#: does not silently become "any overlap" for the largest cells.
SUBSTANTIAL_FRACTION = 0.20
#: ...and at least this many voxels, so a 200-voxel cell needs a real fragment.
SUBSTANTIAL_VOXELS = 20
#: Physical tolerances for boundary precision / recall / F1, in micrometres.
BOUNDARY_TOLERANCES_UM = (0.4, 0.8)
#: IoU thresholds for object precision / recall.
IOU_THRESHOLDS = (0.5, 0.75)


# --------------------------------------------------------------- contingency


def contingency(truth: np.ndarray, prediction: np.ndarray):
    """Sparse voxel contingency table of two 1-based label maps.

    Returns ``(rows, cols, counts, truth_sizes, pred_sizes)`` where ``rows`` and
    ``cols`` index the *labels themselves* (0 included), so the caller can index
    the size arrays directly.
    """

    t = np.asarray(truth).ravel().astype(np.int64)
    p = np.asarray(prediction).ravel().astype(np.int64)
    n_pred = int(p.max()) + 1
    key = t * n_pred + p
    unique, counts = np.unique(key, return_counts=True)
    rows = unique // n_pred
    cols = unique % n_pred
    truth_sizes = np.bincount(t, minlength=int(t.max()) + 1)
    pred_sizes = np.bincount(p, minlength=n_pred)
    return rows, cols, counts, truth_sizes, pred_sizes


def variation_of_information(
    rows: np.ndarray,
    cols: np.ndarray,
    counts: np.ndarray,
    truth_sizes: np.ndarray,
    pred_sizes: np.ndarray,
) -> tuple[float, float, float]:
    """``(total, split, merge)`` variation of information, in bits."""

    total = float(counts.sum())
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    joint = counts / total
    p_truth = truth_sizes / total
    p_pred = pred_sizes / total
    mutual = float(
        np.sum(joint * np.log2(joint / (p_truth[rows] * p_pred[cols])))
    )
    h_truth = float(-np.sum(p_truth[p_truth > 0] * np.log2(p_truth[p_truth > 0])))
    h_pred = float(-np.sum(p_pred[p_pred > 0] * np.log2(p_pred[p_pred > 0])))
    split = max(h_pred - mutual, 0.0)
    merge = max(h_truth - mutual, 0.0)
    return split + merge, split, merge


def adjusted_rand_index(
    counts: np.ndarray, truth_sizes: np.ndarray, pred_sizes: np.ndarray
) -> float:
    """ARI from the sparse contingency table.

    Same value as ``sklearn.metrics.adjusted_rand_score`` but without rebuilding
    the table, which is the expensive part when this is called tens of thousands
    of times.
    """

    n = float(truth_sizes.sum())
    if n < 2:
        return float("nan")

    def comb2(x: np.ndarray) -> float:
        x = x.astype(np.float64)
        return float(np.sum(x * (x - 1.0) / 2.0))

    sum_ij = comb2(counts)
    sum_i = comb2(truth_sizes)
    sum_j = comb2(pred_sizes)
    total = n * (n - 1.0) / 2.0
    expected = sum_i * sum_j / total
    maximum = 0.5 * (sum_i + sum_j)
    if maximum == expected:
        return 1.0 if sum_ij == expected else 0.0
    return float((sum_ij - expected) / (maximum - expected))


# -------------------------------------------------------------- object level


def object_metrics(
    rows: np.ndarray,
    cols: np.ndarray,
    counts: np.ndarray,
    truth_sizes: np.ndarray,
    pred_sizes: np.ndarray,
    *,
    substantial_fraction: float = SUBSTANTIAL_FRACTION,
    substantial_voxels: int = SUBSTANTIAL_VOXELS,
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
) -> dict[str, float]:
    """Object-level agreement, including the two fragmentation counts.

    Both label maps are assumed to be complete partitions with 1-based labels
    (label 0 absent), which is what the benchmark's watershed produces.
    """

    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix

    keep = (rows > 0) & (cols > 0)
    rows, cols, counts = rows[keep], cols[keep], counts[keep]
    true_ids = np.nonzero(truth_sizes[1:])[0] + 1
    pred_ids = np.nonzero(pred_sizes[1:])[0] + 1
    n_true, n_pred = int(true_ids.size), int(pred_ids.size)
    if n_true == 0 or n_pred == 0 or counts.size == 0:
        return {"n_cells_true": n_true, "n_cells_pred": n_pred}

    union = truth_sizes[rows] + pred_sizes[cols] - counts
    iou = counts / np.maximum(union, 1)
    dice = 2.0 * counts / np.maximum(truth_sizes[rows] + pred_sizes[cols], 1)

    # --- Hungarian matching on IoU ---------------------------------------
    row_index = np.searchsorted(true_ids, rows)
    col_index = np.searchsorted(pred_ids, cols)
    iou_dense = coo_matrix(
        (iou, (row_index, col_index)), shape=(n_true, n_pred)
    ).toarray()
    matched_rows, matched_cols = linear_sum_assignment(iou_dense, maximize=True)
    matched_iou = iou_dense[matched_rows, matched_cols]
    dice_dense = coo_matrix(
        (dice, (row_index, col_index)), shape=(n_true, n_pred)
    ).toarray()
    matched_dice = dice_dense[matched_rows, matched_cols]

    out: dict[str, float] = {
        "n_cells_true": n_true,
        "n_cells_pred": n_pred,
        "cell_count_error": n_pred - n_true,
        "matched_iou_mean": float(matched_iou.sum() / n_true),
        "matched_dice_mean": float(matched_dice.sum() / n_true),
    }
    for threshold in iou_thresholds:
        true_positive = int(np.sum(matched_iou >= threshold))
        false_positive = n_pred - true_positive
        false_negative = n_true - true_positive
        precision = true_positive / max(n_pred, 1)
        recall = true_positive / max(n_true, 1)
        tag = f"{threshold:g}".replace(".", "p")
        out[f"object_precision_at_{tag}"] = float(precision)
        out[f"object_recall_at_{tag}"] = float(recall)
        out[f"object_f1_at_{tag}"] = float(
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )
        # Panoptic quality = segmentation quality x recognition quality.
        out[f"panoptic_quality_at_{tag}"] = float(
            matched_iou[matched_iou >= threshold].sum()
            / max(true_positive + 0.5 * false_positive + 0.5 * false_negative, 1e-12)
        )

    # --- substantial fragments -------------------------------------------
    big_for_true = (counts >= substantial_fraction * truth_sizes[rows]) & (
        counts >= substantial_voxels
    )
    big_for_pred = (counts >= substantial_fraction * pred_sizes[cols]) & (
        counts >= substantial_voxels
    )
    fragments_per_true = np.bincount(
        rows[big_for_true], minlength=truth_sizes.size
    )[true_ids]
    cells_per_pred = np.bincount(
        cols[big_for_pred], minlength=pred_sizes.size
    )[pred_ids]

    out["true_cells_split"] = int(np.sum(fragments_per_true >= 2))
    out["true_cells_split_fraction"] = float(np.mean(fragments_per_true >= 2))
    out["excess_fragments_total"] = int(
        np.sum(np.maximum(fragments_per_true - 1, 0))
    )
    out["excess_fragments_per_true_cell"] = float(
        np.mean(np.maximum(fragments_per_true - 1, 0))
    )
    out["max_fragments_per_true_cell"] = int(fragments_per_true.max(initial=0))
    out["true_cells_unrepresented"] = int(np.sum(fragments_per_true == 0))
    out["pred_cells_merging"] = int(np.sum(cells_per_pred >= 2))
    out["max_true_cells_per_pred_cell"] = int(cells_per_pred.max(initial=0))
    return out


def dominant_map(
    rows: np.ndarray,
    cols: np.ndarray,
    counts: np.ndarray,
    n_labels: int,
    by: str = "pred",
) -> np.ndarray:
    """Dominant counterpart label of every label, as an index array.

    ``by="pred"`` returns, for each predicted label, the true label holding most
    of its voxels; ``by="truth"`` the reverse.
    """

    key, other = (cols, rows) if by == "pred" else (rows, cols)
    best = np.zeros(n_labels, dtype=np.int64)
    order = np.argsort(counts, kind="stable")
    # Ascending sort means the last write per key is its largest overlap.
    best[key[order]] = other[order]
    return best


# ------------------------------------------------------------ boundary level


def face_boundaries(labels: np.ndarray) -> np.ndarray:
    """Face-connected interface voxels of a label map (both sides marked)."""

    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in range(labels.ndim):
        lo = [slice(None)] * labels.ndim
        hi = [slice(None)] * labels.ndim
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        differs = labels[tuple(lo)] != labels[tuple(hi)]
        boundary[tuple(lo)] |= differs
        boundary[tuple(hi)] |= differs
    return boundary


def face_areas_um2(spacing_um_zyx: Sequence[float]) -> tuple[float, ...]:
    """Physical area of a voxel face normal to each axis."""

    spacing = tuple(float(v) for v in spacing_um_zyx)
    total = float(np.prod(spacing))
    return tuple(total / s for s in spacing)


def boundary_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    spacing_um_zyx: Sequence[float],
    *,
    tolerances_um: Sequence[float] = BOUNDARY_TOLERANCES_UM,
) -> dict[str, float]:
    """ASSD plus boundary precision / recall / F1 at physical tolerances.

    One pair of distance transforms serves both: the ASSD is the mean of the two
    one-sided distance distributions, and the tolerance metrics are the fraction
    of each distribution below the tolerance.
    """

    from scipy.ndimage import distance_transform_edt

    sampling = tuple(float(v) for v in spacing_um_zyx)
    gt = face_boundaries(truth)
    pred = face_boundaries(prediction)
    if not np.any(gt) or not np.any(pred):
        # Schema-v3 policy: an absent interface is legitimately undefined,
        # represented as JSON null and ordered worst by the selector.
        return {"boundary_assd_um": None}

    gt_to_pred = distance_transform_edt(~pred, sampling=sampling)[gt]
    pred_to_gt = distance_transform_edt(~gt, sampling=sampling)[pred]

    out = {
        "boundary_assd_um": float(
            (gt_to_pred.sum() + pred_to_gt.sum())
            / (gt_to_pred.size + pred_to_gt.size)
        ),
        "boundary_hausdorff95_um": float(
            max(np.percentile(gt_to_pred, 95), np.percentile(pred_to_gt, 95))
        ),
    }
    for tolerance in tolerances_um:
        recall = float(np.mean(gt_to_pred <= tolerance))
        precision = float(np.mean(pred_to_gt <= tolerance))
        tag = f"{tolerance:g}".replace(".", "p")
        out[f"boundary_recall_at_{tag}um"] = recall
        out[f"boundary_precision_at_{tag}um"] = precision
        out[f"boundary_f1_at_{tag}um"] = float(
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )
    return out


def facet_table(
    truth: np.ndarray,
    prediction: np.ndarray,
    spacing_um_zyx: Sequence[float],
) -> dict[str, np.ndarray]:
    """Per true facet: physical area, and the area at which the prediction agrees.

    A facet is the interface between one pair of adjacent ground-truth cells.
    ``separated`` records whether the two cells end up with different *dominant*
    predicted labels, which is the version of the question over-segmentation
    cannot win.
    """

    areas = face_areas_um2(spacing_um_zyx)
    n_truth = int(truth.max()) + 1
    total = {}
    found = {}
    for axis in range(truth.ndim):
        lo = [slice(None)] * truth.ndim
        hi = [slice(None)] * truth.ndim
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        one, two = truth[tuple(lo)], truth[tuple(hi)]
        differs = one != two
        if not np.any(differs):
            continue
        low = np.minimum(one[differs], two[differs]).astype(np.int64)
        high = np.maximum(one[differs], two[differs]).astype(np.int64)
        key = low * n_truth + high
        agree = (prediction[tuple(lo)] != prediction[tuple(hi)])[differs]
        counts = np.bincount(key)
        total[axis] = counts
        found[axis] = np.bincount(key, weights=agree.astype(np.float64))

    size = max(max(v.size for v in total.values()), 1)
    total_area = np.zeros(size)
    found_area = np.zeros(size)
    total_faces = np.zeros(size)
    found_faces = np.zeros(size)
    for axis, counts in total.items():
        total_area[: counts.size] += counts * areas[axis]
        found_area[: found[axis].size] += found[axis] * areas[axis]
        total_faces[: counts.size] += counts
        found_faces[: found[axis].size] += found[axis]

    present = np.nonzero(total_faces)[0]
    low = present // n_truth
    high = present % n_truth
    rows, cols, counts, truth_sizes, pred_sizes = contingency(truth, prediction)
    dominant_pred = dominant_map(rows, cols, counts, int(truth_sizes.size), by="truth")
    return {
        "cell_a": low,
        "cell_b": high,
        "area_um2": total_area[present],
        "recovered_area_um2": found_area[present],
        "faces": total_faces[present],
        "recovered_faces": found_faces[present],
        "cells_separated": dominant_pred[low] != dominant_pred[high],
    }


def interface_precision(
    truth: np.ndarray,
    prediction: np.ndarray,
    spacing_um_zyx: Sequence[float],
    dominant_true_of_pred: np.ndarray,
) -> dict[str, float]:
    """Fraction of the predicted interface area that separates real cells.

    Every face at which the prediction changes label is classified by the
    *dominant true labels* of the two predicted cells meeting there.  If they
    are the same true cell the face is a **false internal boundary**: the
    prediction has cut a real cell in half.  Area-weighted, in physical units.

    Faces touching label 0 are skipped, so this works unchanged on a marker map,
    where most voxels are still unlabelled and the interface between a marker
    and the unclaimed pool is not a predicted boundary at all.
    """

    areas = face_areas_um2(spacing_um_zyx)
    total = 0.0
    false_internal = 0.0
    for axis in range(prediction.ndim):
        lo = [slice(None)] * prediction.ndim
        hi = [slice(None)] * prediction.ndim
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        one, two = prediction[tuple(lo)], prediction[tuple(hi)]
        differs = (one != two) & (one > 0) & (two > 0)
        n = int(differs.sum())
        if n == 0:
            continue
        same_true = (
            dominant_true_of_pred[one[differs]]
            == dominant_true_of_pred[two[differs]]
        )
        total += n * areas[axis]
        false_internal += int(same_true.sum()) * areas[axis]
    if total <= 0:
        return {
            "interface_precision": None,
            "false_internal_boundary_area_um2": 0.0,
            "false_internal_boundary_fraction": None,
            "predicted_interface_area_um2": 0.0,
        }
    return {
        "interface_precision": float(1.0 - false_internal / total),
        "false_internal_boundary_area_um2": float(false_internal),
        "false_internal_boundary_fraction": float(false_internal / total),
        "predicted_interface_area_um2": float(total),
    }


# ------------------------------------------------------- marker-level errors


def marker_errors(
    markers: np.ndarray,
    truth: np.ndarray,
    *,
    substantial_fraction: float = SUBSTANTIAL_FRACTION,
    substantial_voxels: int = SUBSTANTIAL_VOXELS,
) -> dict[str, float]:
    """Identification errors, measured on the markers before any refinement.

    Markers are a partial labelling -- most voxels are still 0 -- so the object
    metrics above do not apply.  What can be measured, and what the watershed
    can only propagate, is how many true cells received several markers, how
    many markers straddle several true cells, and how many cells got nothing.

    A marker counts against a true cell when it claims a substantial part of
    that cell, and a true cell counts against a marker when it holds a
    substantial part of that marker -- the same two-sided rule as
    :func:`object_metrics`, so the before/after comparison is like for like.
    """

    selected = markers > 0
    present = np.unique(truth[truth > 0])
    n_markers = int(markers.max())
    if not np.any(selected) or n_markers == 0:
        return {
            "marker_count": 0,
            "markers_covering_multiple_cells": 0,
            "max_cells_per_marker": 0,
            "percolating_marker_volume_fraction": 0.0,
            "cells_with_multiple_markers": 0,
            "cells_unseeded": int(present.size),
            "marker_volume_fraction": 0.0,
        }

    n_truth = int(truth.max())
    key = markers[selected].astype(np.int64) * (n_truth + 1) + truth[
        selected
    ].astype(np.int64)
    unique, counts = np.unique(key, return_counts=True)
    marker_id = unique // (n_truth + 1)
    truth_id = unique % (n_truth + 1)

    marker_sizes = np.bincount(markers.ravel(), minlength=n_markers + 1)
    truth_sizes = np.bincount(truth.ravel(), minlength=n_truth + 1)

    big_for_marker = (
        counts >= substantial_fraction * marker_sizes[marker_id]
    ) & (counts >= substantial_voxels)
    big_for_truth = (counts >= substantial_fraction * truth_sizes[truth_id]) & (
        counts >= substantial_voxels
    )
    cells_per_marker = np.bincount(
        marker_id[big_for_marker], minlength=n_markers + 1
    )[1:]
    markers_per_cell = np.bincount(
        truth_id[big_for_truth], minlength=n_truth + 1
    )[present]

    percolating = cells_per_marker >= 2
    total = float(marker_sizes[1:].sum())
    return {
        "marker_count": n_markers,
        "markers_covering_multiple_cells": int(percolating.sum()),
        "max_cells_per_marker": int(cells_per_marker.max(initial=0)),
        "percolating_marker_volume_fraction": (
            float(marker_sizes[1:][percolating].sum() / total) if total > 0 else 0.0
        ),
        "cells_with_multiple_markers": int((markers_per_cell >= 2).sum()),
        "excess_markers_total": int(np.sum(np.maximum(markers_per_cell - 1, 0))),
        "cells_unseeded": int((markers_per_cell == 0).sum()),
        "marker_volume_fraction": float(total / markers.size),
    }


# ----------------------------------------------------------------- assembled


def evaluate_partition(
    truth: np.ndarray,
    prediction: np.ndarray,
    spacing_um_zyx: Sequence[float],
    *,
    with_boundary: bool = True,
) -> dict[str, float]:
    """Every voxel, object and boundary metric for one predicted partition."""

    if np.any(prediction <= 0):
        raise ValueError("Prediction contains unlabelled voxels.")

    rows, cols, counts, truth_sizes, pred_sizes = contingency(truth, prediction)
    total, split, merge = variation_of_information(
        rows, cols, counts, truth_sizes, pred_sizes
    )
    out: dict[str, float] = {
        "ari": adjusted_rand_index(counts, truth_sizes, pred_sizes),
        "vi_total_bits": total,
        "vi_split_bits": split,
        "vi_merge_bits": merge,
    }
    out.update(
        object_metrics(rows, cols, counts, truth_sizes, pred_sizes)
    )
    if with_boundary:
        out.update(boundary_metrics(truth, prediction, spacing_um_zyx))
        dominant_true = dominant_map(
            rows, cols, counts, int(pred_sizes.size), by="pred"
        )
        out.update(
            interface_precision(
                truth, prediction, spacing_um_zyx, dominant_true
            )
        )
        facets = facet_table(truth, prediction, spacing_um_zyx)
        area = float(facets["area_um2"].sum())
        out["true_facet_recall_area_weighted"] = (
            float(facets["recovered_area_um2"].sum() / area) if area > 0 else None
        )
    return out


def wilson_interval(
    successes: float, trials: float, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used rather than the normal approximation because several facet bands hold
    fewer than a hundred facets and sit near a proportion of 1, where the normal
    interval runs past the end of the scale.
    """

    n = float(trials)
    if n <= 0:
        return float("nan"), float("nan")
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = (
        z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    )
    return float(max(centre - half, 0.0)), float(min(centre + half, 1.0))

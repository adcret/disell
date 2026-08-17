#!/usr/bin/env python3
"""Ground-truth metrics for the synthetic 3D dislocation-cell benchmark.

The selection criterion is the adjusted Rand index.  Variation of information,
the physical boundary distance and the cell-count error are secondary
diagnostics: they say *how* a partition is wrong, which ARI alone does not.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def variation_of_information(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, float]:
    """Return ``(total, split, merge)`` variation of information, in bits.

    ``split = H(prediction | truth)`` penalises over-segmentation;
    ``merge = H(truth | prediction)`` penalises under-segmentation.
    """

    from sklearn.metrics.cluster import contingency_matrix

    gt = np.asarray(truth).ravel()
    pred = np.asarray(prediction).ravel()
    joint = contingency_matrix(gt, pred, sparse=True).astype(np.float64)
    total = float(joint.sum())
    if total <= 0:
        return float("nan"), float("nan"), float("nan")

    joint = joint.multiply(1.0 / total).tocoo()
    p_truth = np.asarray(joint.sum(axis=1)).ravel()
    p_pred = np.asarray(joint.sum(axis=0)).ravel()

    mutual = float(
        np.sum(
            joint.data
            * np.log2(joint.data / (p_truth[joint.row] * p_pred[joint.col]))
        )
    )
    entropy_truth = float(-np.sum(p_truth[p_truth > 0] * np.log2(p_truth[p_truth > 0])))
    entropy_pred = float(-np.sum(p_pred[p_pred > 0] * np.log2(p_pred[p_pred > 0])))
    split = max(entropy_pred - mutual, 0.0)
    merge = max(entropy_truth - mutual, 0.0)
    return split + merge, split, merge


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


def boundary_assd_um(
    truth: np.ndarray,
    prediction: np.ndarray,
    spacing_um_zyx: Sequence[float],
) -> float:
    """Average symmetric surface distance between the two boundary networks.

    In micrometres, using the anisotropic voxel spacing, so a one-voxel error
    along the coarse axis costs more than one along a fine axis.
    """

    from scipy.ndimage import distance_transform_edt

    gt = face_boundaries(truth)
    pred = face_boundaries(prediction)
    if not np.any(gt) or not np.any(pred):
        return float("nan")

    sampling = tuple(float(v) for v in spacing_um_zyx)
    gt_to_pred = distance_transform_edt(~pred, sampling=sampling)[gt]
    pred_to_gt = distance_transform_edt(~gt, sampling=sampling)[pred]
    return float(
        (gt_to_pred.sum() + pred_to_gt.sum()) / (gt_to_pred.size + pred_to_gt.size)
    )


def marker_confusion(
    markers: np.ndarray,
    truth: np.ndarray,
    min_overlap_voxels: int = 25,
) -> dict[str, float]:
    """How identification markers overlap the ground-truth cells.

    Both failure modes of a thresholded edge field appear here, before any
    refinement; a watershed can only propagate them, never undo them.

    * **Percolation.**  One marker leaks across a weak boundary and covers two
      or more ground-truth cells.  Reported as ``percolating_markers``, the
      worst case ``max_cells_per_marker``, and the share of marker volume
      sitting inside percolating markers.
    * **Fragmentation.**  One ground-truth cell is claimed by two or more
      markers, so it is split.  Reported as ``cells_split``.
    * **Unseeded cells**, which get no marker at all and can only be absorbed
      by a neighbour during refinement.

    A marker counts as covering a cell when it claims at least
    ``min_overlap_voxels`` of it, comparable to the minimum accepted marker
    size, so incidental single-voxel contacts are not counted.
    """

    selected = markers > 0
    # Count against the cells that actually exist, not against ``max()``: a
    # config that leaves a label empty must not inflate the unseeded count.
    present = np.unique(truth[truth > 0])
    n_truth = int(truth.max())
    if not np.any(selected):
        return {
            "marker_count": 0,
            "percolating_markers": 0,
            "max_cells_per_marker": 0,
            "percolating_marker_volume_fraction": 0.0,
            "cells_split": 0,
            "cells_unseeded": int(present.size),
        }

    n_markers = int(markers.max())
    key = markers[selected].astype(np.int64) * (n_truth + 1) + truth[
        selected
    ].astype(np.int64)
    unique, counts = np.unique(key, return_counts=True)
    marker_id = unique // (n_truth + 1)
    truth_id = unique % (n_truth + 1)

    keep = counts >= int(min_overlap_voxels)
    cells_per_marker = np.bincount(marker_id[keep], minlength=n_markers + 1)[1:]
    markers_per_cell = np.bincount(truth_id[keep], minlength=n_truth + 1)[present]

    sizes = np.bincount(markers.ravel(), minlength=n_markers + 1)[1:]
    percolating = cells_per_marker >= 2
    total = float(sizes.sum())
    return {
        "marker_count": n_markers,
        "percolating_markers": int(percolating.sum()),
        "max_cells_per_marker": int(cells_per_marker.max(initial=0)),
        "percolating_marker_volume_fraction": (
            float(sizes[percolating].sum() / total) if total > 0 else 0.0
        ),
        "cells_split": int((markers_per_cell >= 2).sum()),
        "cells_unseeded": int((markers_per_cell == 0).sum()),
    }


def facet_recovery(
    truth: np.ndarray,
    prediction: np.ndarray,
    region_means_deg: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per ground-truth facet: its misorientation and how much of it was found.

    Two measures per facet, because one is not enough:

    * ``recovered_fraction`` -- the fraction of the shared interface at which
      the prediction also changes label.  This is pure recall, and an
      over-segmenting prediction scores well on it trivially because it changes
      label almost everywhere.
    * ``cells_separated`` -- whether the two ground-truth cells end up with
      *different dominant predicted labels*.  That is the question that
      matters, and it cannot be won by over-segmenting.

    Pairing either with the facet's own misorientation separates two very
    different failures: a facet with a large angular step that was missed is a
    parameter or algorithm failure, while a facet with near-zero contrast
    carries no information to find it with and no method could recover it.
    """

    n_truth = int(truth.max()) + 1
    # Dominant predicted label of every ground-truth cell, for the
    # oversegmentation-proof separation test below.
    flat_truth = truth.ravel().astype(np.int64)
    flat_pred = prediction.ravel().astype(np.int64)
    n_pred = int(prediction.max()) + 1
    counts = np.bincount(
        flat_truth * n_pred + flat_pred, minlength=n_truth * n_pred
    ).reshape(n_truth, n_pred)
    dominant = counts.argmax(axis=1)

    total_counts = np.zeros(n_truth * n_truth, dtype=np.int64)
    found_counts = np.zeros(n_truth * n_truth, dtype=np.int64)

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
        separated = (
            prediction[tuple(lo)] != prediction[tuple(hi)]
        )[differs].astype(np.int64)
        total_counts += np.bincount(key, minlength=total_counts.size)
        found_counts += np.bincount(
            key, weights=separated, minlength=found_counts.size
        ).astype(np.int64)

    present = np.nonzero(total_counts)[0]
    if present.size == 0:
        empty = np.empty(0)
        return {
            "misorientation_deg": empty,
            "recovered_fraction": empty,
            "facet_voxels": empty,
        }
    low = present // n_truth
    high = present % n_truth
    delta = np.asarray(region_means_deg)[low] - np.asarray(region_means_deg)[high]
    delta = (delta + 180.0) % 360.0 - 180.0
    return {
        "misorientation_deg": np.linalg.norm(delta, axis=-1),
        "recovered_fraction": found_counts[present] / total_counts[present],
        "cells_separated": (dominant[low] != dominant[high]).astype(float),
        "facet_voxels": total_counts[present].astype(float),
    }


def facet_recovery_bands(
    recovery: dict[str, np.ndarray],
    edges_deg: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.40, np.inf),
    recovered_threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Summarise facet recovery in bands of ground-truth misorientation."""

    misorientation = recovery["misorientation_deg"]
    recovered = recovery["recovered_fraction"]
    separated = recovery["cells_separated"]
    out = []
    for low, high in zip(edges_deg[:-1], edges_deg[1:]):
        selected = (misorientation >= low) & (misorientation < high)
        n = int(selected.sum())
        out.append(
            {
                "low_deg": float(low),
                "high_deg": float(high) if np.isfinite(high) else None,
                "n_facets": n,
                "median_recovered_fraction": (
                    float(np.median(recovered[selected])) if n else float("nan")
                ),
                "fraction_of_facets_mostly_recovered": (
                    float(np.mean(recovered[selected] >= recovered_threshold))
                    if n
                    else float("nan")
                ),
                "fraction_of_cell_pairs_separated": (
                    float(np.mean(separated[selected])) if n else float("nan")
                ),
            }
        )
    return out


def evaluate(
    truth: np.ndarray,
    prediction: np.ndarray,
    spacing_um_zyx: Sequence[float],
) -> dict[str, float]:
    """Score one predicted partition against the known ground truth."""

    from sklearn.metrics import adjusted_rand_score

    if np.any(prediction <= 0):
        raise ValueError("Prediction contains unlabelled voxels.")

    total, split, merge = variation_of_information(truth, prediction)
    n_truth = int(np.unique(truth).size)
    n_pred = int(np.unique(prediction).size)
    return {
        "ari": float(adjusted_rand_score(truth.ravel(), prediction.ravel())),
        "vi_total_bits": total,
        "vi_split_bits": split,
        "vi_merge_bits": merge,
        "boundary_assd_um": boundary_assd_um(truth, prediction, spacing_um_zyx),
        "n_cells_truth": n_truth,
        "n_cells_pred": n_pred,
        "cell_count_error": n_pred - n_truth,
    }

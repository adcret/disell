#!/usr/bin/env python3
"""Cell-identity and cell-mean-orientation metrics for labelled partitions.

Identity is deliberately not defined by IoU >= 0.5.  A predicted/true pair is
eligible when it passes explicit purity and completeness thresholds; among
eligible pairs, a Hungarian assignment maximises reciprocal overlap (Dice),
with wrapped cell-mean angular error as a small, deterministic tie-breaker.
This tolerates displaced interfaces while preventing one region from claiming
more than one cell.  Orientation never rescues a pair that fails the identity
criteria.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_IDENTITY_TOLERANCES = ((0.5, 0.5), (0.6, 0.6), (0.7, 0.7))
DEFAULT_ORIENTATION_TOLERANCES_DEG = (0.005, 0.01, 0.02, 0.05)


def wrapped_delta_deg(a, b, periods_deg: Sequence[float] = (360.0, 360.0)):
    """Signed shortest per-channel angular difference ``a - b`` in degrees."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    period = np.asarray(periods_deg, float)
    return (a - b + period / 2.0) % period - period / 2.0


def circular_cell_means(field, labels, periods_deg=(360.0, 360.0)):
    """Equal-voxel circular mean of each channel for every positive label."""
    field, labels = np.asarray(field), np.asarray(labels)
    ids = np.unique(labels[labels > 0]).astype(int)
    out = np.full((int(labels.max(initial=0)) + 1, field.shape[-1]), np.nan)
    periods = np.asarray(periods_deg, float)
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=out.shape[0])
    for channel, period in enumerate(periods):
        phase = field[..., channel].ravel() * (2 * np.pi / period)
        real = np.bincount(flat, weights=np.cos(phase), minlength=out.shape[0])
        imag = np.bincount(flat, weights=np.sin(phase), minlength=out.shape[0])
        valid = counts > 0
        # Keep the signed principal branch.  Mapping negative small angles to
        # values near 360 degrees would make ordinary 1-D Wasserstein distances
        # meaningless despite the wrapped pairwise errors being correct.
        out[valid, channel] = np.arctan2(imag[valid], real[valid]) * period / (2 * np.pi)
    return out


def contingency(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    valid = (truth > 0) & (prediction > 0)
    pairs, counts = np.unique(
        np.stack((truth[valid], prediction[valid]), axis=1), axis=0,
        return_counts=True,
    )
    nt, npred = int(truth.max(initial=0)), int(prediction.max(initial=0))
    ts = np.bincount(truth[truth > 0].ravel(), minlength=nt + 1)
    ps = np.bincount(prediction[prediction > 0].ravel(), minlength=npred + 1)
    return pairs[:, 0], pairs[:, 1], counts, ts, ps


def _centroids(labels, spacing):
    labels = np.asarray(labels)
    out = np.full((int(labels.max(initial=0)) + 1, labels.ndim), np.nan)
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=out.shape[0])
    for axis in range(labels.ndim):
        coordinate = np.indices(labels.shape, sparse=True)[axis]
        values = np.broadcast_to(coordinate, labels.shape).ravel()
        totals = np.bincount(flat, weights=values, minlength=out.shape[0])
        valid = counts > 0
        out[valid, axis] = totals[valid] / counts[valid] * spacing[axis]
    return out


def adjacent_misorientations(labels, means, periods_deg=(360.0, 360.0)):
    """Wrapped two-channel RMS misorientation for face-adjacent cell pairs."""
    edges = set()
    for axis in range(labels.ndim):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis)
        b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        mask = (a > 0) & (b > 0) & (a != b)
        for x, y in zip(a[mask], b[mask]):
            edges.add(tuple(sorted((int(x), int(y)))))
    if not edges:
        return np.empty(0)
    return np.asarray([
        np.sqrt(np.sum(wrapped_delta_deg(means[a], means[b], periods_deg) ** 2))
        for a, b in sorted(edges)
    ])


def _wasserstein(a, b, weights_a=None, weights_b=None):
    from scipy.stats import wasserstein_distance
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0:
        return None
    if a.ndim == 1:
        return float(wasserstein_distance(a, b, weights_a, weights_b))
    # Report a defensible channel-averaged 1-D marginal distance.  The report
    # also exposes each channel separately; this is not claimed as multivariate OT.
    return float(np.mean([
        wasserstein_distance(a[:, c], b[:, c], weights_a, weights_b)
        for c in range(a.shape[1])
    ]))


def match_cells(
    truth, prediction, field, spacing_um_zyx,
    *, purity_threshold=0.5, completeness_threshold=0.5,
    periods_deg=(360.0, 360.0), substantial_fraction=0.10,
    substantial_voxels=5,
):
    """Return pair records and a summary for one explicit identity criterion."""
    from scipy.optimize import linear_sum_assignment

    truth, prediction, field = map(np.asarray, (truth, prediction, field))
    rows, cols, overlap, ts, ps = contingency(truth, prediction)
    true_ids = np.flatnonzero(ts[1:]) + 1
    pred_ids = np.flatnonzero(ps[1:]) + 1
    tmean = circular_cell_means(field, truth, periods_deg)
    pmean = circular_cell_means(field, prediction, periods_deg)
    tc, pc = _centroids(truth, np.asarray(spacing_um_zyx)), _centroids(prediction, np.asarray(spacing_um_zyx))
    shape = (len(true_ids), len(pred_ids))
    eligible = np.zeros(shape, bool)
    score = np.zeros(shape, float)
    lookup = {}
    for t, p, n in zip(rows, cols, overlap):
        purity, completeness = n / ps[p], n / ts[t]
        dice = 2 * n / (ps[p] + ts[t])
        delta = wrapped_delta_deg(pmean[p], tmean[t], periods_deg)
        error = float(np.sqrt(np.sum(delta ** 2)))
        i, j = np.searchsorted(true_ids, t), np.searchsorted(pred_ids, p)
        ok = purity >= purity_threshold and completeness >= completeness_threshold
        eligible[i, j] = ok
        # Orientation is only a bounded tie-breaker after reciprocal overlap.
        score[i, j] = dice + 1e-6 / (1.0 + error) if ok else -1e6
        lookup[(int(t), int(p))] = (int(n), purity, completeness, dice, delta, error)
    if shape[0] and shape[1]:
        ii, jj = linear_sum_assignment(score, maximize=True)
    else:
        ii, jj = np.empty(0, int), np.empty(0, int)
    pairs = []
    voxel_volume = float(np.prod(spacing_um_zyx))
    for i, j in zip(ii, jj):
        if not eligible[i, j]:
            continue
        t, p = int(true_ids[i]), int(pred_ids[j])
        n, purity, completeness, dice, delta, error = lookup[(t, p)]
        pairs.append({
            "true_cell": t, "predicted_cell": p, "overlap_voxels": n,
            "prediction_purity": float(purity), "true_cell_completeness": float(completeness),
            "iou": float(n / (ts[t] + ps[p] - n)), "dice": float(dice),
            "centroid_displacement_um": float(np.linalg.norm(pc[p] - tc[t])),
            "true_volume_um3": float(ts[t] * voxel_volume),
            "predicted_volume_um3": float(ps[p] * voxel_volume),
            "physical_volume_ratio": float(ps[p] / ts[t]),
            "true_mean_chi_deg": float(tmean[t, 0]), "true_mean_phi_deg": float(tmean[t, 1]),
            "predicted_mean_chi_deg": float(pmean[p, 0]), "predicted_mean_phi_deg": float(pmean[p, 1]),
            "mean_orientation_error_deg": error,
            "chi_mean_error_deg": float(abs(delta[0])), "phi_mean_error_deg": float(abs(delta[1])),
        })

    true_fragment_count = np.zeros(ts.size, int)
    pred_source_count = np.zeros(ps.size, int)
    substantial_for_true = (overlap >= substantial_voxels) & (overlap >= substantial_fraction * ts[rows])
    substantial_for_pred = (overlap >= substantial_voxels) & (overlap >= substantial_fraction * ps[cols])
    np.add.at(true_fragment_count, rows[substantial_for_true], 1)
    np.add.at(pred_source_count, cols[substantial_for_pred], 1)
    matched_pred = {x["predicted_cell"] for x in pairs}
    errors = np.asarray([x["mean_orientation_error_deg"] for x in pairs])
    volumes = np.asarray([x["true_volume_um3"] for x in pairs])
    pred_values = pmean[pred_ids]
    true_values = tmean[true_ids]
    summary = {
        "purity_threshold": float(purity_threshold),
        "completeness_threshold": float(completeness_threshold),
        "n_true_cells": int(len(true_ids)), "n_predicted_cells": int(len(pred_ids)),
        "cell_count_error": int(len(pred_ids) - len(true_ids)),
        "one_to_one_recovered_cells": int(len(pairs)),
        "one_to_one_recovered_fraction": float(len(pairs) / max(len(true_ids), 1)),
        "unrepresented_true_cells": int(np.sum(true_fragment_count[true_ids] == 0)),
        "unmatched_true_cells": int(len(true_ids) - len(pairs)),
        "unmatched_predictions": int(len(pred_ids) - len(pairs)),
        "split_true_cells": int(np.sum(true_fragment_count[true_ids] >= 2)),
        "merged_predicted_cells": int(np.sum(pred_source_count[pred_ids] >= 2)),
        "duplicate_predictions": int(sum(p not in matched_pred and pred_source_count[p] == 1 for p in pred_ids)),
        "absolute_cell_count_error": int(abs(len(pred_ids) - len(true_ids))),
        "relative_cell_count_error": float((len(pred_ids) - len(true_ids)) / max(len(true_ids), 1)),
        "mean_matched_mean_orientation_error_deg": float(np.mean(errors)) if errors.size else None,
        "median_matched_mean_orientation_error_deg": float(np.median(errors)) if errors.size else None,
        "p90_matched_mean_orientation_error_deg": float(np.percentile(errors, 90)) if errors.size else None,
        "p95_matched_mean_orientation_error_deg": float(np.percentile(errors, 95)) if errors.size else None,
        "equal_cell_weighted_orientation_rmse_deg": float(np.sqrt(np.mean(errors ** 2))) if errors.size else None,
        "volume_weighted_orientation_rmse_deg": float(np.sqrt(np.average(errors ** 2, weights=volumes))) if errors.size else None,
        "cell_mean_wasserstein_deg": _wasserstein(pred_values, true_values),
        "cell_mean_wasserstein_chi_deg": _wasserstein(pred_values[:, 0], true_values[:, 0]),
        "cell_mean_wasserstein_phi_deg": _wasserstein(pred_values[:, 1], true_values[:, 1]),
        "adjacent_misorientation_wasserstein_deg": _wasserstein(
            adjacent_misorientations(prediction, pmean, periods_deg),
            adjacent_misorientations(truth, tmean, periods_deg),
        ),
    }
    if pairs:
        signed = np.asarray([
            wrapped_delta_deg(
                [x["predicted_mean_chi_deg"], x["predicted_mean_phi_deg"]],
                [x["true_mean_chi_deg"], x["true_mean_phi_deg"]], periods_deg,
            ) for x in pairs
        ])
        summary["chi_mean_orientation_bias_deg"] = float(np.mean(signed[:, 0]))
        summary["phi_mean_orientation_bias_deg"] = float(np.mean(signed[:, 1]))
    else:
        summary["chi_mean_orientation_bias_deg"] = None
        summary["phi_mean_orientation_bias_deg"] = None
    for tolerance in DEFAULT_ORIENTATION_TOLERANCES_DEG:
        tag = str(tolerance).replace(".", "p")
        n = int(np.sum(errors <= tolerance))
        summary[f"orientation_correct_cells_at_{tag}deg"] = n
        summary[f"orientation_correct_recovered_fraction_at_{tag}deg"] = float(n / max(len(true_ids), 1))
    return pairs, summary

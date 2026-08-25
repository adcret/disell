#!/usr/bin/env python3
"""Strict cell recovery: how many true cells come out almost exactly right.

The objective this serves
-------------------------
"Maximise the cells I recover that have over 90 % correct voxels, and let the
rest be over-segmented rather than fused."

A true cell counts as **recovered at tau** when some predicted cell satisfies

    purity        = |t and p| / |p|  >= tau      (the prediction is not padded
                                                  with another cell's voxels)
    completeness  = |t and p| / |t|  >= tau      (it holds nearly all of t)

For ``tau > 0.5`` this pairing is automatically one-to-one: a predicted cell
holding more than half of ``t`` leaves too little for any other, and vice
versa.  So unlike the 0.6/0.6 criterion used by
``object_orientation_metrics.match_cells`` -- which needs a Hungarian
assignment because two candidates can both qualify -- this needs no assignment
at all and is cheap enough to score every configuration in a search.

The rest of the cells
---------------------
Among true cells that are *not* recovered, the two failure modes are not
equivalent:

``split``   the cell is divided between several predicted cells.  A downstream
            merge can fold them back together, so this is recoverable.
``fused``   the predicted cell covering it also substantially covers another
            true cell.  Nothing downstream can separate them again.

The reported ``fused_true_cells`` is therefore the quantity to minimise, and
``split_true_cells`` the one to tolerate.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TAU = 0.9

#: A predicted cell "substantially" covers a true cell when it takes at least
#: this fraction of it, and at least this many voxels.  Matches the convention
#: in ``object_orientation_metrics``.
SUBSTANTIAL_FRACTION = 0.10
SUBSTANTIAL_VOXELS = 5


def contingency(truth: np.ndarray, prediction: np.ndarray):
    """Overlapping (true, predicted) pairs and the two label size vectors."""

    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    valid = (truth > 0) & (prediction > 0)
    if not np.any(valid):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, np.zeros(1, np.int64), np.zeros(1, np.int64)

    t = truth[valid].astype(np.int64).ravel()
    p = prediction[valid].astype(np.int64).ravel()
    n_pred = int(prediction.max()) + 1
    flat = t * n_pred + p
    keys, counts = np.unique(flat, return_counts=True)
    rows, cols = np.divmod(keys, n_pred)

    truth_sizes = np.bincount(truth[truth > 0].ravel(),
                              minlength=int(truth.max()) + 1).astype(np.int64)
    pred_sizes = np.bincount(prediction[prediction > 0].ravel(),
                             minlength=n_pred).astype(np.int64)
    return rows, cols, counts, truth_sizes, pred_sizes


def strict_recovery(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    tau: float = DEFAULT_TAU,
    substantial_fraction: float = SUBSTANTIAL_FRACTION,
    substantial_voxels: int = SUBSTANTIAL_VOXELS,
) -> dict:
    """Count true cells recovered at ``tau``, and classify the failures.

    ``tau`` must exceed 0.5 for the one-to-one guarantee to hold.
    """

    if not 0.5 < float(tau) <= 1.0:
        raise ValueError(f"tau must lie in (0.5, 1.0]; got {tau}")

    rows, cols, counts, truth_sizes, pred_sizes = contingency(truth, prediction)
    present_true = np.flatnonzero(truth_sizes)
    present_true = present_true[present_true > 0]
    n_true = int(present_true.size)
    n_pred = int(np.count_nonzero(pred_sizes[1:]))

    if counts.size == 0 or n_true == 0:
        return {
            "tau": float(tau), "n_cells_true": n_true, "n_cells_pred": n_pred,
            "recovered_at_tau": 0, "recovery_rate_at_tau": 0.0,
            "recovered_purity_only": 0,
            "fused_true_cells": 0, "split_true_cells": 0,
            "unrecovered_true_cells": n_true,
            "fused_true_cell_fraction": 0.0,
            "contamination": 0.0, "contaminated_voxels": 0,
            "fused_contamination": 0.0, "fused_predicted_cells": 0,
            "fused_voxel_fraction": 0.0,
        }

    purity = counts / pred_sizes[cols]
    completeness = counts / truth_sizes[rows]

    strict = (purity >= tau) & (completeness >= tau)
    recovered_true = np.unique(rows[strict])

    # Purity alone, reported separately because "90 % correct voxels" can be
    # read either way: this counts a prediction that is clean but clipped.
    purity_only = np.unique(rows[(purity >= tau) & (completeness >= 0.5)])

    substantial = (counts >= substantial_voxels) & (
        counts >= substantial_fraction * truth_sizes[rows]
    )
    # A predicted cell that substantially covers two or more true cells has
    # fused them; every true cell it touches that way is counted as fused.
    per_pred = np.bincount(cols[substantial], minlength=pred_sizes.size)
    fused_mask = substantial & (per_pred[cols] >= 2)
    fused_true = np.unique(rows[fused_mask])
    fused_true = np.setdiff1d(fused_true, recovered_true)

    # A true cell split across two or more predicted cells, none of which
    # recovered it.
    per_true = np.bincount(rows[substantial], minlength=truth_sizes.size)
    split_true = np.flatnonzero(per_true >= 2)
    split_true = np.setdiff1d(split_true, recovered_true)
    split_true = np.setdiff1d(split_true, fused_true)

    # How far the fused cells are from being one true cell each.  Counting
    # fused cells treats a predicted cell that is 50/50 across two true cells
    # the same as one that is 95/5, but the second is nearly right and only
    # needs trimming.  Contamination measures the voxels that would have to
    # move: for every predicted cell, the share not belonging to its dominant
    # true cell.
    dominant_share = np.zeros(pred_sizes.size)
    np.maximum.at(dominant_share, cols, counts)
    labelled = pred_sizes.copy()
    labelled[0] = 0
    total_labelled = int(labelled.sum())
    contamination_voxels = int(labelled.sum() - dominant_share[1:].sum())
    contamination = contamination_voxels / max(total_labelled, 1)

    # The same quantity restricted to the predicted cells that actually fuse
    # two or more true cells, which is the population the policy cares about.
    fusing_preds = np.unique(cols[substantial & (per_pred[cols] >= 2)])
    if fusing_preds.size:
        fused_voxels = int(pred_sizes[fusing_preds].sum())
        fused_contamination = (
            fused_voxels - float(dominant_share[fusing_preds].sum())
        ) / max(fused_voxels, 1)
    else:
        fused_voxels, fused_contamination = 0, 0.0

    recovered = int(recovered_true.size)
    return {
        "contamination": float(contamination),
        "contaminated_voxels": contamination_voxels,
        "fused_contamination": float(fused_contamination),
        "fused_predicted_cells": int(fusing_preds.size),
        "fused_voxel_fraction": fused_voxels / max(total_labelled, 1),
        "tau": float(tau),
        "n_cells_true": n_true,
        "n_cells_pred": n_pred,
        "recovered_at_tau": recovered,
        "recovery_rate_at_tau": recovered / max(n_true, 1),
        "recovered_purity_only": int(purity_only.size),
        "fused_true_cells": int(fused_true.size),
        "split_true_cells": int(split_true.size),
        "unrecovered_true_cells": n_true - recovered,
        "fused_true_cell_fraction": int(fused_true.size) / max(n_true, 1),
    }

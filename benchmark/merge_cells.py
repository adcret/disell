#!/usr/bin/env python3
"""Orientation-gated merging of small cells, as a post-watershed step.

Why this exists.  ``min_cell_size`` controls the recovered cell count by
*deleting* markers below a size, which is a blunt instrument: on the phantom,
raising it keeps improving identity F1 long after the cell count has collapsed,
because it is buying count control by destroying genuinely small cells.  This
module offers the alternative -- segment with a small ``min_cell_size``, accept
the resulting over-segmentation, then merge each fragment into the neighbour it
is orientationally closest to.

The rule is deliberately conservative and is the one specified for this study:

    for each cell smaller than ``merge_size_voxels``, smallest first:
        n = the adjacent cell whose mean orientation is closest
        if misorientation(cell, n) < merge_threshold_deg:
            merge cell into n and recompute n's mean

Two large cells are therefore never merged into each other, so the step can
only ever remove fragments -- it cannot dissolve a genuine interface between
two well-resolved cells.

Cell mean orientations are carried as per-channel complex resultant sums, so a
merge is an exact addition rather than a recomputation, and the merged mean
equals the circular mean over the union of the voxels.  This matches
``object_orientation_metrics.circular_cell_means``, which is what the metrics
score against.
"""

from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np

DEFAULT_PERIODS_DEG = (360.0, 360.0)


def adjacency_pairs(labels: np.ndarray) -> np.ndarray:
    """Unique unordered pairs of face-adjacent positive labels, shape (n, 2)."""

    labels = np.asarray(labels)
    collected = []
    for axis in range(labels.ndim):
        rolled = np.moveaxis(labels, axis, 0)
        lower, upper = rolled[:-1].ravel(), rolled[1:].ravel()
        keep = (lower != upper) & (lower > 0) & (upper > 0)
        if not np.any(keep):
            continue
        collected.append(
            np.stack(
                (np.minimum(lower[keep], upper[keep]),
                 np.maximum(lower[keep], upper[keep])),
                axis=1,
            )
        )
    if not collected:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(collected).astype(np.int64), axis=0)


def _resultant_sums(field: np.ndarray, labels: np.ndarray, periods_deg):
    """Per-label per-channel (sum cos, sum sin) of the phase-mapped field."""

    flat = labels.ravel()
    size = int(labels.max(initial=0)) + 1
    channels = field.shape[-1]
    real = np.zeros((size, channels))
    imag = np.zeros((size, channels))
    for channel, period in enumerate(periods_deg[:channels]):
        phase = field[..., channel].ravel() * (2 * np.pi / period)
        real[:, channel] = np.bincount(flat, weights=np.cos(phase), minlength=size)
        imag[:, channel] = np.bincount(flat, weights=np.sin(phase), minlength=size)
    return real, imag


def _means_from_sums(real, imag, periods_deg) -> np.ndarray:
    periods = np.asarray(periods_deg[: real.shape[-1]], float)
    return np.arctan2(imag, real) * periods / (2 * np.pi)


def _misorientation_deg(mean_a, mean_b, periods) -> float:
    delta = (mean_a - mean_b + periods / 2.0) % periods - periods / 2.0
    return float(np.sqrt(np.sum(delta * delta)))


def _spread_deg(real, imag, count, periods) -> float:
    """Circular spread of one region, in degrees, from its resultant sums.

    ``R = |resultant| / n`` is the mean resultant length; the circular standard
    deviation is ``sqrt(-2 ln R)`` in radians.  Because the sums are additive,
    the spread of a merged region is available before committing to the merge --
    which is what lets it be used as a gate rather than a diagnostic.
    """

    if count <= 1:
        return 0.0
    out = 0.0
    for channel, period in enumerate(np.atleast_1d(periods)):
        resultant = np.hypot(real[channel], imag[channel]) / count
        resultant = min(max(float(resultant), 1e-12), 1.0)
        radians = np.sqrt(max(-2.0 * np.log(resultant), 0.0))
        out += (radians * float(period) / (2 * np.pi)) ** 2
    return float(np.sqrt(out))


def reference_spread_deg(real, imag, sizes, periods, *, large_voxels: int,
                         percentile: float = 80.0) -> float | None:
    """Internal spread of the confidently resolved regions, at a percentile.

    The gate compares a candidate merge against this.  The percentile sets how
    permissive it is: 50 asks that the merged region be no more varied than a
    typical cell, 80 allows it to be as varied as a fairly varied one.
    """

    big = np.flatnonzero(sizes >= int(large_voxels))
    big = big[big > 0]
    if big.size < 8:
        return None
    values = [_spread_deg(real[i], imag[i], sizes[i], periods) for i in big]
    return float(np.percentile(values, percentile))


def reference_misorientation_deg(
    labels: np.ndarray,
    real,
    imag,
    sizes,
    periods,
    *,
    large_voxels: int,
    percentile: float = 50.0,
) -> float | None:
    """Typical misorientation between neighbouring *large* regions.

    The large regions are the ones the identification step resolved
    confidently, so the spread between them is a measurement of how far apart
    genuine neighbouring cells sit in this particular volume.  Using it as the
    reference makes the merge rule self-calibrating: nothing has to be supplied
    in degrees, and the same setting transfers between volumes whose
    misorientation scale differs -- which is exactly what happens across strain.

    Returns ``None`` when too few large neighbours exist to estimate it.
    """

    pairs = adjacency_pairs(labels)
    if pairs.size == 0:
        return None
    big = sizes >= int(large_voxels)
    keep = big[pairs[:, 0]] & big[pairs[:, 1]]
    if np.count_nonzero(keep) < 8:
        return None
    values = [
        _misorientation_deg(_means_from_sums(real[a], imag[a], periods),
                            _means_from_sums(real[b], imag[b], periods), periods)
        for a, b in pairs[keep]
    ]
    return float(np.percentile(values, percentile))


def merge_small_cells(
    labels: np.ndarray,
    field: np.ndarray,
    *,
    merge_size_voxels: int,
    merge_threshold_deg: float = 0.0,
    merge_mode: str = "absolute",
    merge_factor: float = 0.25,
    spread_factor: float | None = None,
    spread_percentile: float = 80.0,
    local_threshold_deg: float | None = None,
    periods_deg=DEFAULT_PERIODS_DEG,
    return_diagnostics: bool = False,
):
    """Merge cells below ``merge_size_voxels`` into their closest neighbour.

    ``merge_mode`` selects what a fragment's misorientation is compared against:

    ``absolute``    ``merge_threshold_deg``, a fixed angle.  Simple, but tied to
                    the misorientation scale of the volume it was tuned on.
    ``relative``    ``merge_factor`` times the typical misorientation between
                    neighbouring *large* regions in this volume, measured by
                    :func:`reference_misorientation_deg`.  Self-calibrating: a
                    fragment merges when it is much closer to its neighbour than
                    genuine neighbouring cells are to each other.
    ``local``       ``merge_factor`` times ``local_threshold_deg``, tying the
                    merge to the tolerance the identification step already used.

    ``spread_factor`` adds a second, independent gate.  Two genuinely distinct
    cells can have similar mean orientations, so the misorientation test above
    does not catch every bad merge; what does catch them is that combining two
    real cells inflates the internal spread of the result.  When set, a merge is
    refused unless the spread of the combined region stays below
    ``spread_factor`` times the spread of the confidently resolved regions at
    ``spread_percentile``.  With the defaults, a merge is refused unless the
    combined region is no more internally varied than a fairly varied real cell.
    ``None`` disables the check.

    Returns relabelled, gap-free labels (1..n).  With ``return_diagnostics``
    also returns a dict recording how many merges happened and why the rest
    did not, which is what tells you whether the threshold is doing any work.
    """

    labels = np.ascontiguousarray(labels, dtype=np.int32)
    merge_size_voxels = int(merge_size_voxels)
    periods = np.asarray(periods_deg[: field.shape[-1]], float)

    sizes = np.bincount(labels.ravel()).astype(np.int64)
    n_labels = sizes.size
    real, imag = _resultant_sums(field, labels, periods_deg)

    spread_limit = None
    if spread_factor is not None:
        reference_spread = reference_spread_deg(
            real, imag, sizes, periods, large_voxels=merge_size_voxels,
            percentile=float(spread_percentile))
        if reference_spread is not None:
            spread_limit = float(spread_factor) * reference_spread

    reference = None
    if merge_mode == "absolute":
        threshold = float(merge_threshold_deg)
    elif merge_mode == "relative":
        reference = reference_misorientation_deg(
            labels, real, imag, sizes, periods, large_voxels=merge_size_voxels)
        if reference is None:
            # Nothing to calibrate against; decline to merge rather than guess.
            threshold = 0.0
        else:
            threshold = float(merge_factor) * reference
    elif merge_mode == "local":
        if local_threshold_deg is None:
            raise ValueError("merge_mode='local' needs local_threshold_deg")
        threshold = float(merge_factor) * float(local_threshold_deg)
    else:
        raise ValueError(f"unknown merge_mode {merge_mode!r}")
    merge_threshold_deg = threshold

    neighbours: dict[int, set[int]] = defaultdict(set)
    for a, b in adjacency_pairs(labels):
        neighbours[int(a)].add(int(b))
        neighbours[int(b)].add(int(a))

    parent = np.arange(n_labels, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return root

    queue = [
        (int(sizes[label]), int(label))
        for label in range(1, n_labels)
        if 0 < sizes[label] < merge_size_voxels
    ]
    heapq.heapify(queue)

    merged = 0
    blocked_by_threshold = 0
    blocked_by_spread = 0
    isolated = 0
    while queue:
        size, label = heapq.heappop(queue)
        root = find(label)
        # Lazy deletion: the entry is stale if the cell has since been merged
        # or grown, and irrelevant if it is no longer under the cutoff.
        if root != label or sizes[root] != size:
            continue
        if sizes[root] >= merge_size_voxels:
            continue

        candidates = {find(other) for other in neighbours[root]} - {root}
        if not candidates:
            isolated += 1
            continue

        own = _means_from_sums(real[root], imag[root], periods)
        best, best_error = None, np.inf
        for other in candidates:
            error = _misorientation_deg(
                own, _means_from_sums(real[other], imag[other], periods), periods
            )
            if error < best_error:
                best, best_error = other, error
        if best_error >= merge_threshold_deg:
            blocked_by_threshold += 1
            continue

        if spread_limit is not None:
            combined = _spread_deg(real[root] + real[best], imag[root] + imag[best],
                                   sizes[root] + sizes[best], periods)
            if combined > spread_limit:
                blocked_by_spread += 1
                continue

        # Merge root into best.  Summing the resultants is exact.
        parent[root] = best
        sizes[best] += sizes[root]
        sizes[root] = 0
        real[best] += real[root]
        imag[best] += imag[root]
        neighbours[best] |= neighbours[root]
        neighbours[best].discard(root)
        neighbours[best].discard(best)
        merged += 1
        if sizes[best] < merge_size_voxels:
            heapq.heappush(queue, (int(sizes[best]), int(best)))

    roots = np.array([find(i) for i in range(n_labels)], dtype=np.int64)
    roots[0] = 0
    survivors = np.unique(roots[1:])
    survivors = survivors[survivors > 0]
    remap = np.zeros(n_labels, dtype=np.int32)
    remap[survivors] = np.arange(1, survivors.size + 1, dtype=np.int32)
    out = remap[roots][labels]

    if not return_diagnostics:
        return out
    diagnostics = {
        "cells_before": int(np.count_nonzero(np.bincount(labels.ravel())[1:])),
        "cells_after": int(survivors.size),
        "merges": merged,
        "blocked_by_threshold": blocked_by_threshold,
        "blocked_by_spread": blocked_by_spread,
        "spread_limit_deg": spread_limit,
        "spread_percentile": float(spread_percentile),
        "isolated_small_cells": isolated,
        "merge_size_voxels": merge_size_voxels,
        "merge_threshold_deg": merge_threshold_deg,
        "merge_mode": merge_mode,
        "merge_factor": float(merge_factor),
        "reference_misorientation_deg": reference,
        "remaining_below_cutoff": int(
            np.count_nonzero((sizes[survivors] > 0) & (sizes[survivors] < merge_size_voxels))
        ),
    }
    return out, diagnostics

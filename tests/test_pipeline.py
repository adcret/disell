"""Tiny end-to-end test for KAM -> multi-seed flood fill -> watershed.

The test builds a small 3D ``(Z, Y, X, 2)`` orientation field with four
clean ``(chi, phi)`` quadrants in the ``(Y, X)`` plane and tiny noise.
On this volume the pipeline must:

1. produce a KAM scalar field that is close to zero inside each
   quadrant and clearly elevated on the inter-quadrant boundaries;
2. recover at least four distinct multi-seed flood-fill markers;
3. after watershed refinement, partition each ground-truth quadrant
   into exactly one dominant label.

This is intentionally a *behavioural* test, not a numerical
reproducibility test: the C++ flood fill is run with explicit
seed_points so the assertions are stable.
"""

from __future__ import annotations

import numpy as np
import pytest

import disell


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


GRID_Z, GRID_Y, GRID_X = 6, 24, 24
QUADRANT_VALUES = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)


def _build_synthetic_volume(rng: np.random.Generator):
    """Return ``(field, mask, ground_truth_labels)`` for the test volume.

    Layout in the (Y, X) plane (each quadrant is half the grid):

        +--------+--------+
        | gt = 1 | gt = 2 |
        +--------+--------+
        | gt = 3 | gt = 4 |
        +--------+--------+

    Repeated identically along ``Z``. The mask zeros the outermost
    voxel of every face; this is the same mitigation the
    paper-figure scripts apply to dodge the C++ neighbour
    out-of-bounds defect (AUDIT.md, B5).
    """
    Z, Y, X = GRID_Z, GRID_Y, GRID_X
    field = np.zeros((Z, Y, X, 2), dtype=np.float32)

    midY, midX = Y // 2, X // 2
    field[:, :midY, :midX] = QUADRANT_VALUES[0]
    field[:, :midY, midX:] = QUADRANT_VALUES[1]
    field[:, midY:, :midX] = QUADRANT_VALUES[2]
    field[:, midY:, midX:] = QUADRANT_VALUES[3]

    # tiny in-cell noise so KAM is not exactly zero everywhere
    field = field + rng.normal(0.0, 1e-3, field.shape).astype(np.float32)

    gt = np.zeros((Z, Y, X), dtype=np.int32)
    gt[:, :midY, :midX] = 1
    gt[:, :midY, midX:] = 2
    gt[:, midY:, :midX] = 3
    gt[:, midY:, midX:] = 4

    mask = np.ones((Z, Y, X), dtype=bool)
    mask[0] = False
    mask[-1] = False
    mask[:, 0] = False
    mask[:, -1] = False
    mask[..., 0] = False
    mask[..., -1] = False

    return field, mask, gt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_seeds(mask: np.ndarray, n_seeds: int, rng: np.random.Generator):
    """Pick ``n_seeds`` seed coordinates uniformly at random inside ``mask``.

    Returns an ``(n, 3)`` int64 array of ``(z, y, x)`` indices, suitable
    for the C++ ``seed_points`` argument. We pre-sample in numpy so the
    test does not depend on the C++ ``std::random_device`` (the
    extension is not seedable from Python).
    """
    Z, Y, X = mask.shape
    flat = np.flatnonzero(mask.ravel())
    chosen = rng.choice(flat, size=min(n_seeds, flat.size), replace=False)
    z = chosen // (Y * X)
    y = (chosen // X) % Y
    x = chosen % X
    return np.stack([z, y, x], axis=-1).astype(np.int64)


def _run_multiseed_flood_fill(field, mask, *, footprint, local_threshold,
                              footprint_tolerance, min_grain_size,
                              max_seed_attempts, stagnation_tolerance,
                              random_seed):
    """Deterministic Python wrapper around ``flood_fill_random_seeds_3d``.

    Mitigations applied (mirrors the paper-figure scripts):

    * pad property map and mask by the footprint half-extent so the
      C++ neighbour reads stay in-bounds;
    * pre-sample seeds with ``numpy.random.default_rng(random_seed)``
      and pass them as ``seed_points`` so the C++ random draw is
      bypassed;
    * pass a *copy* of the mask, since the C++ mutates it in place.
    """
    half = tuple(s // 2 for s in footprint.shape)
    pad = ((half[0],) * 2, (half[1],) * 2, (half[2],) * 2)

    field_p = np.pad(np.ascontiguousarray(field, dtype=np.float32),
                     pad + ((0, 0),), mode="constant")
    mask_p = np.pad(np.ascontiguousarray(mask.astype(np.uint8)),
                    pad, mode="constant")

    rng = np.random.default_rng(random_seed)
    seeds_p = _sample_seeds(mask_p.astype(bool), max_seed_attempts, rng)

    result = disell.flood_fill_random_seeds_3d(
        field_p,
        footprint.astype(bool),
        float(local_threshold),
        float(footprint_tolerance),
        mask_p.copy(),
        int(max_seed_attempts),
        int(min_grain_size),
        False,                       # recycle_small_grains
        int(stagnation_tolerance),
        seeds_p,
    )
    seg_p = np.asarray(result["segmentation"], dtype=np.int32)
    sz, sy, sx = half
    return seg_p[sz: seg_p.shape[0] - sz,
                 sy: seg_p.shape[1] - sy,
                 sx: seg_p.shape[2] - sx].copy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_kam_low_inside_quadrants_high_at_boundaries():
    """KAM is near-zero inside each clean quadrant and large at boundaries."""
    rng = np.random.default_rng(0)
    field, mask, gt = _build_synthetic_volume(rng)

    kam = disell.kam(field, ndim=3, size=3)
    assert kam.shape == field.shape[:3], f"KAM shape mismatch: {kam.shape}"

    # Pick voxels strictly inside one quadrant and strictly inside mask.
    inside_q1 = (gt == 1) & mask
    inside_q1[:, GRID_Y // 2 - 2 :, :] = False  # avoid the y-boundary
    inside_q1[..., GRID_X // 2 - 2 :] = False  # avoid the x-boundary
    inside_q1[:2] = False
    inside_q1[-2:] = False

    assert inside_q1.sum() > 0, "no clean interior voxels selected; check geometry"

    # Boundary band: voxels close to the y midline.
    midY = GRID_Y // 2
    boundary = mask & np.zeros_like(mask)
    boundary[:, midY - 1 : midY + 1, :] = True
    boundary &= mask

    kam_inside = kam[inside_q1]
    kam_boundary = kam[boundary]

    assert kam_inside.size > 0
    assert kam_boundary.size > 0
    assert np.nanmax(kam_inside) < 0.05, (
        "KAM should be small inside a clean quadrant; got "
        f"max={float(np.nanmax(kam_inside)):.4g}"
    )
    assert np.nanmedian(kam_boundary) > 10 * np.nanmedian(kam_inside), (
        "KAM at quadrant boundaries should clearly exceed the interior; "
        f"got median(boundary)={float(np.nanmedian(kam_boundary)):.4g} vs "
        f"median(interior)={float(np.nanmedian(kam_inside)):.4g}"
    )


def test_pipeline_recovers_four_quadrants():
    """KAM -> multi-seed flood fill -> watershed recovers four labels."""
    rng = np.random.default_rng(42)
    field, mask, gt = _build_synthetic_volume(rng)

    kam = disell.kam(field, ndim=3, size=3).astype(np.float32)
    # KAM is 0 in the kernel-radius border (AUDIT.md B13). Replace by
    # a large value inside the test mask so the watershed prefers cell
    # interiors. Outside the mask the value is irrelevant.
    finite_kam = kam[np.isfinite(kam) & mask]
    assert finite_kam.size > 0
    kam_for_watershed = np.where(
        np.isfinite(kam), kam, float(finite_kam.max() + 1.0)
    ).astype(np.float32)

    # Multi-seed flood fill (deterministic).
    footprint = np.ones((3, 3, 3), dtype=bool)
    markers = _run_multiseed_flood_fill(
        field, mask,
        footprint=footprint,
        local_threshold=0.1,    # ||·||² < 0.1**2 * 2 = 0.02; well above noise
        footprint_tolerance=0.85,
        min_grain_size=50,
        max_seed_attempts=300,
        stagnation_tolerance=200,
        random_seed=42,
    )

    n_markers = int(markers.max())
    assert n_markers >= 4, (
        f"expected at least 4 multi-seed markers (one per quadrant), "
        f"got {n_markers}"
    )

    # Watershed refinement.
    labels = disell.region_grow_watershed(
        markers, mask, kam_for_watershed, connectivity=1
    ).astype(np.int32)

    # Every in-mask voxel must receive a label.
    unassigned = (labels == 0) & mask
    assert unassigned.sum() == 0, (
        f"watershed left {int(unassigned.sum())} in-mask voxels unlabelled"
    )

    # Each ground-truth quadrant must be dominated by exactly one label.
    for gt_id in (1, 2, 3, 4):
        quadrant = (gt == gt_id) & mask
        if quadrant.sum() == 0:
            continue
        unique, counts = np.unique(labels[quadrant], return_counts=True)
        non_zero = unique != 0
        unique = unique[non_zero]
        counts = counts[non_zero]
        assert unique.size > 0, f"quadrant {gt_id} got no labels at all"
        dominant_fraction = counts.max() / counts.sum()
        assert dominant_fraction > 0.9, (
            f"quadrant {gt_id} is not dominated by a single label "
            f"(dominant fraction {dominant_fraction:.3f}); "
            f"label counts: {dict(zip(unique.tolist(), counts.tolist()))}"
        )

    # The four quadrants must map to four *different* dominant labels.
    dominant = []
    for gt_id in (1, 2, 3, 4):
        quadrant = (gt == gt_id) & mask
        unique, counts = np.unique(labels[quadrant], return_counts=True)
        non_zero = unique != 0
        unique = unique[non_zero]
        counts = counts[non_zero]
        dominant.append(int(unique[counts.argmax()]))
    assert len(set(dominant)) == 4, (
        f"expected the four quadrants to map to four distinct labels, "
        f"got {dominant}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

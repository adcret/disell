"""Reproducibility tests for the optional ``random_seed`` parameter.

When ``random_seed`` is a non-negative integer the internal ``std::mt19937``
RNG is seeded deterministically, so repeated calls with identical inputs must
produce identical segmentations. When ``random_seed`` is None the behaviour
falls back to ``std::random_device`` (non-reproducible) but must still return a
valid segmentation array.
"""

import numpy as np
import pytest

from disell.cell_identification import flood_fill_dfxm, flood_fill_dfxm_two_stage


def _make_inputs():
    # Small structured-but-noisy 2D property map so the segmentation depends on
    # the random seed order (and is therefore a meaningful reproducibility test).
    rng = np.random.default_rng(0)
    H, W = 16, 16
    prop = rng.random((H, W, 1)).astype(np.float32)
    # The C++ flood fill assumes a masked-off border (it does not bounds-check
    # neighbour indices), which is the realistic usage pattern. Keep a 1-voxel
    # false frame so edge voxels are never grown.
    mask = np.zeros((H, W), dtype=bool)
    mask[1:-1, 1:-1] = True
    footprint = np.ones((3, 3), dtype=bool)
    return prop, footprint, mask


def _is_valid_segmentation(seg, shape):
    return (
        isinstance(seg, np.ndarray)
        and seg.shape == shape
        and np.issubdtype(seg.dtype, np.integer)
    )


def test_flood_fill_dfxm_reproducible():
    prop, footprint, mask = _make_inputs()
    kwargs = dict(
        footprint=footprint,
        local_threshold=0.2,
        footprint_tolerance=0.5,
        mask=mask,
        max_iterations=200,
        min_grain_size=2,
        stagnation_tolerance=0,
    )

    result1 = flood_fill_dfxm(prop, random_seed=42, **kwargs)
    result2 = flood_fill_dfxm(prop, random_seed=42, **kwargs)

    assert np.array_equal(result1["segmentation"], result2["segmentation"])

    # random_seed=None must still run and return a valid segmentation array.
    result_none = flood_fill_dfxm(prop, random_seed=None, **kwargs)
    assert _is_valid_segmentation(result_none["segmentation"], prop.shape[:2])


def test_flood_fill_dfxm_two_stage_reproducible():
    prop, footprint, mask = _make_inputs()
    kwargs = dict(
        footprint=footprint,
        local_misorientation_threshold=0.2,
        footprint_tolerance=0.5,
        mask=mask,
        max_iterations=200,
        min_grain_size=2,
        stagnation_tolerance=0,
    )

    result1, sizes_initial1 = flood_fill_dfxm_two_stage(prop, random_seed=42, **kwargs)
    result2, sizes_initial2 = flood_fill_dfxm_two_stage(prop, random_seed=42, **kwargs)

    assert np.array_equal(sizes_initial1, sizes_initial2)
    assert np.array_equal(result1["segmentation"], result2["segmentation"])

    # random_seed=None must still run and return a valid segmentation array.
    result_none, _ = flood_fill_dfxm_two_stage(prop, random_seed=None, **kwargs)
    assert _is_valid_segmentation(result_none["segmentation"], prop.shape[:2])

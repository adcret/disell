"""Tests for input-mask preservation of the flood-fill wrappers and the
cell-size statistics helper."""

import numpy as np

from disell import flood_fill_dfxm, flood_fill_dfxm_two_stage, get_cell_size_list


def _phantom():
    rng = np.random.default_rng(0)
    field = np.zeros((4, 20, 20, 2), dtype=np.float32)
    field[:, :, 10:, 0] = 1.0
    field += rng.normal(scale=0.01, size=field.shape).astype(np.float32)
    mask = np.ones(field.shape[:3], dtype=np.uint8)
    footprint = np.ones((3, 3, 3), dtype=bool)
    return field, mask, footprint


def test_flood_fill_dfxm_preserves_mask():
    field, mask, footprint = _phantom()
    mask_before = mask.copy()
    flood_fill_dfxm(
        field, footprint=footprint, local_threshold=0.1, mask=mask,
        max_iterations=500, min_grain_size=10, random_seed=1,
    )
    np.testing.assert_array_equal(mask, mask_before)


def test_flood_fill_two_stage_preserves_mask():
    field, mask, footprint = _phantom()
    mask_before = mask.copy()
    flood_fill_dfxm_two_stage(
        field, footprint=footprint, local_misorientation_threshold=0.1,
        mask=mask, max_iterations=500, min_grain_size=10, random_seed=1,
    )
    np.testing.assert_array_equal(mask, mask_before)


def test_get_cell_size_list_min_size_filter():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[:2, :2] = 1      # 4 px
    labels[5:, 5:] = 2      # 25 px
    labels[0, 5:8] = 3      # 3 px
    ids, sizes = get_cell_size_list(labels, min_cell_size=4)
    assert list(ids) == [2]
    assert list(sizes) == [25.0]


def test_get_cell_size_list_physical_units():
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[:2, :3] = 1
    ids, sizes = get_cell_size_list(labels, pixel_size=[2.0, 0.5])
    assert list(ids) == [1]
    assert sizes[0] == 6 * 1.0


def test_get_cell_size_list_background_sequence():
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0] = 1
    labels[1] = 2
    ids, sizes = get_cell_size_list(labels, background=[0, 1])
    assert list(ids) == [2]

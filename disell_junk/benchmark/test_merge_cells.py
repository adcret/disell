#!/usr/bin/env python3
"""Tests for the orientation-gated small-cell merge."""

from __future__ import annotations

import numpy as np
import pytest

import merge_cells as mc


def block_labels(sizes):
    """A 1-D chain of cells along x, each ``sizes[i]`` voxels wide."""

    labels = np.zeros((1, 1, sum(sizes)), dtype=np.int32)
    start = 0
    for index, width in enumerate(sizes, start=1):
        labels[0, 0, start:start + width] = index
        start += width
    return labels


def field_for(labels, per_label_deg):
    """A two-channel field that is constant inside each label."""

    field = np.zeros(labels.shape + (2,), dtype=np.float32)
    for label, value in per_label_deg.items():
        field[labels == label] = value
    return field


def test_adjacent_small_cells_with_equal_orientation_merge():
    labels = block_labels([2, 2, 20])
    field = field_for(labels, {1: (0.0, 0.0), 2: (0.0, 0.0), 3: (5.0, 5.0)})
    out, diagnostics = mc.merge_small_cells(
        labels, field, merge_size_voxels=5, merge_threshold_deg=0.1,
        return_diagnostics=True,
    )
    # Cells 1 and 2 are identical and adjacent; cell 3 is far away in
    # orientation and above the cutoff, so it must survive untouched.
    assert diagnostics["cells_after"] == 2
    assert diagnostics["merges"] == 1
    assert np.count_nonzero(out == out[0, 0, 0]) == 4


def test_threshold_blocks_a_dissimilar_neighbour():
    labels = block_labels([2, 20])
    field = field_for(labels, {1: (1.0, 0.0), 2: (0.0, 0.0)})
    out, diagnostics = mc.merge_small_cells(
        labels, field, merge_size_voxels=5, merge_threshold_deg=0.5,
        return_diagnostics=True,
    )
    assert diagnostics["merges"] == 0
    assert diagnostics["blocked_by_threshold"] == 1
    assert int(out.max()) == 2


def test_cells_above_the_cutoff_are_never_merged():
    labels = block_labels([30, 30])
    field = field_for(labels, {1: (0.0, 0.0), 2: (0.0, 0.0)})
    # Identical orientation and adjacent, but both are above the cutoff.
    out, diagnostics = mc.merge_small_cells(
        labels, field, merge_size_voxels=10, merge_threshold_deg=10.0,
        return_diagnostics=True,
    )
    assert diagnostics["merges"] == 0
    assert int(out.max()) == 2


def test_merge_picks_the_closest_neighbour_not_the_first():
    labels = block_labels([20, 2, 20])
    # The fragment sits between two large cells; it must join the right one.
    field = field_for(labels, {1: (0.9, 0.0), 2: (0.0, 0.0), 3: (0.05, 0.0)})
    out = mc.merge_small_cells(
        labels, field, merge_size_voxels=5, merge_threshold_deg=0.5,
    )
    fragment_value = out[0, 0, 21]
    assert fragment_value == out[0, 0, 40], "fragment joined the wrong neighbour"
    assert fragment_value != out[0, 0, 0]


def test_merging_cascades_until_above_the_cutoff():
    labels = block_labels([2, 2, 2, 2])
    field = field_for(labels, {i: (0.0, 0.0) for i in range(1, 5)})
    out, diagnostics = mc.merge_small_cells(
        labels, field, merge_size_voxels=8, merge_threshold_deg=0.1,
        return_diagnostics=True,
    )
    # All four are identical; they collapse into one cell of 8 voxels, which is
    # no longer under the cutoff.
    assert diagnostics["cells_after"] == 1
    assert int(out.max()) == 1


def test_isolated_small_cell_survives():
    labels = np.zeros((1, 1, 20), dtype=np.int32)
    labels[0, 0, :3] = 1          # background gap at 3:7 leaves cell 1 isolated
    labels[0, 0, 7:] = 2          # 13 voxels, comfortably above the cutoff
    field = field_for(labels, {1: (0.0, 0.0), 2: (0.0, 0.0)})
    out, diagnostics = mc.merge_small_cells(
        labels, field, merge_size_voxels=5, merge_threshold_deg=1.0,
        return_diagnostics=True,
    )
    assert diagnostics["isolated_small_cells"] == 1
    assert diagnostics["merges"] == 0
    assert int(out.max()) == 2


def test_output_labels_are_consecutive_and_gap_free():
    rng = np.random.default_rng(0)
    labels = rng.integers(1, 40, size=(4, 12, 12)).astype(np.int32)
    field = rng.normal(0, 0.3, size=labels.shape + (2,)).astype(np.float32)
    out = mc.merge_small_cells(
        labels, field, merge_size_voxels=20, merge_threshold_deg=0.2,
    )
    present = np.unique(out[out > 0])
    assert present.min() == 1
    assert np.array_equal(present, np.arange(1, present.size + 1))
    assert out.shape == labels.shape


def test_background_is_preserved():
    labels = block_labels([2, 20])
    labels[0, 0, :1] = 0
    field = field_for(labels, {1: (0.0, 0.0), 2: (0.0, 0.0)})
    out = mc.merge_small_cells(
        labels, field, merge_size_voxels=5, merge_threshold_deg=1.0,
    )
    assert out[0, 0, 0] == 0, "background voxel was given a label"


def test_merged_mean_matches_the_circular_mean_over_the_union():
    """The resultant sums must make a merge exact, not approximate."""

    import object_orientation_metrics as oom

    labels = block_labels([3, 5, 20])
    field = field_for(labels, {1: (0.10, -0.20), 2: (0.14, -0.16), 3: (9.0, 9.0)})
    out = mc.merge_small_cells(
        labels, field, merge_size_voxels=9, merge_threshold_deg=1.0,
    )
    merged_id = out[0, 0, 0]
    assert out[0, 0, 4] == merged_id, "the two fragments should have merged"

    direct = oom.circular_cell_means(field, out)[merged_id]
    union = (labels == 1) | (labels == 2)
    expected = oom.circular_cell_means(
        field, np.where(union, 1, 0).astype(np.int32)
    )[1]
    assert np.allclose(direct, expected, atol=1e-9)


def test_zero_threshold_merges_nothing_but_exact_matches():
    labels = block_labels([2, 2, 20])
    field = field_for(labels, {1: (0.0, 0.0), 2: (1e-6, 0.0), 3: (5.0, 5.0)})
    _, diagnostics = mc.merge_small_cells(
        labels, field, merge_size_voxels=5, merge_threshold_deg=0.0,
        return_diagnostics=True,
    )
    assert diagnostics["merges"] == 0


def test_adjacency_pairs_are_unique_and_ordered():
    labels = block_labels([2, 2, 2])
    pairs = mc.adjacency_pairs(labels)
    assert pairs.shape == (2, 2)
    assert np.array_equal(pairs, np.array([[1, 2], [2, 3]]))
    assert np.all(pairs[:, 0] < pairs[:, 1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

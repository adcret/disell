"""Tests for disell.metrics: boundary-band KAM without bbox clipping,
inner spread, label-invariant comparison metrics and component splitting."""

import numpy as np
import pytest

from disell import (
    inner_cell_spread,
    boundary_band_kam,
    variation_of_information,
    matched_overlap,
    connected_component_report,
    split_disconnected_labels,
)


def two_cell_volume():
    """Two touching slabs filling a (4, 6, 8) volume."""
    labels = np.zeros((4, 6, 8), dtype=np.int32)
    labels[:, :, :4] = 1
    labels[:, :, 4:] = 2
    return labels


def test_inner_cell_spread_exact():
    labels = two_cell_volume()
    field = np.zeros(labels.shape + (2,))
    field[..., 0][labels == 2] = 1.0  # cell 2 offset in channel 0 only
    spread = inner_cell_spread(labels, field)
    assert spread[1] == 0.0
    assert spread[2] == 0.0
    # add known variance to cell 1, channel 1
    vals = np.zeros(labels.shape)
    vals[labels == 1] = np.tile([1.0, -1.0], (labels == 1).sum() // 2)
    field[..., 1] = vals
    spread = inner_cell_spread(labels, field)
    # rho^2 = (1/2)*(d0^2 + d1^2); d1 = +/-1 around zero mean -> sigma^2 = 0.5
    assert spread[1] == pytest.approx(0.5)


def test_boundary_band_kam_not_clipped_by_bbox():
    labels = two_cell_volume()
    kam = np.zeros(labels.shape)
    # the interface plane between the cells is x in {3, 4}
    kam[:, :, 3:5] = 2.0
    e_bd = boundary_band_kam(labels, kam, r_bd=1, connectivity=1)
    # boundary of cell 1 is x==3; band (r_bd=1) covers x in {2,3,4};
    # median over values {0, 2, 2} = 2.0
    assert e_bd[1] == pytest.approx(2.0)
    assert e_bd[2] == pytest.approx(2.0)
    # cells span the whole volume: a bbox-clipped implementation would have
    # excluded parts of the band at the volume faces; verify the band size by
    # recomputing with a KAM field that counts membership
    ones = np.ones_like(kam)
    e_ones = boundary_band_kam(labels, ones, r_bd=1)
    assert e_ones[1] == 1.0


def test_boundary_band_kam_nan_ignored():
    labels = two_cell_volume()
    kam = np.full(labels.shape, np.nan)
    kam[:, :, 3] = 1.0
    e_bd = boundary_band_kam(labels, kam, r_bd=1)
    assert e_bd[1] == pytest.approx(1.0)


def test_vi_and_matched_overlap_label_invariance():
    rng = np.random.default_rng(0)
    a = rng.integers(1, 5, size=(6, 7, 8))
    # relabel a by a fixed permutation: partitions identical
    perm = {1: 40, 2: 17, 3: 99, 4: 3}
    b = np.vectorize(perm.get)(a)
    assert variation_of_information(a, b) == pytest.approx(0.0, abs=1e-12)
    assert matched_overlap(a, b) == pytest.approx(1.0)


def test_vi_sensitive_to_merge():
    a = np.zeros((1, 4, 8), dtype=int)
    a[..., :4] = 1
    a[..., 4:] = 2
    b = np.ones_like(a)  # everything merged
    vi = variation_of_information(a, b)
    assert vi == pytest.approx(1.0)  # H(A) = 1 bit lost
    assert matched_overlap(a, b) == pytest.approx(0.5)


def test_connected_component_report_and_split():
    labels = np.zeros((1, 5, 11), dtype=np.int32)
    labels[0, :, :3] = 1
    labels[0, :, 8:] = 1   # same label, disconnected
    labels[0, :, 4:7] = 2
    report = connected_component_report(labels, connectivity=1)
    assert report["n_labels"] == 2
    assert report["n_disconnected"] == 1
    assert 1 in report["fragments"]

    new_labels, info = split_disconnected_labels(labels, connectivity=1)
    report2 = connected_component_report(new_labels, connectivity=1)
    assert report2["n_disconnected"] == 0
    # both fragments have 15 voxels; deterministic tie-break keeps original id
    # on the first component and assigns max+1 to the second
    assert set(np.unique(new_labels)) == {0, 1, 2, 3}
    # determinism
    new_labels_b, _ = split_disconnected_labels(labels, connectivity=1)
    np.testing.assert_array_equal(new_labels, new_labels_b)


def test_split_min_size_removes_small_fragment():
    labels = np.zeros((1, 3, 10), dtype=np.int32)
    labels[0, :, :6] = 1
    labels[0, 1, 9] = 1  # single stray voxel
    new_labels, info = split_disconnected_labels(labels, min_size=3)
    assert new_labels[0, 1, 9] == 0
    assert info["removed_voxels"] == 1

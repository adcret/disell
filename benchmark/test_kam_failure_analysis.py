import numpy as np

import oracle_metrics as om
import pipelines
from kam_failure_analysis import kam_field


def test_kam_neighbourhood_is_spherical_in_physical_coordinates():
    footprint = pipelines.isotropic_footprint((1.0, 0.4, 0.4), 1.01)
    assert footprint.shape == (3, 5, 5)
    assert footprint[0, 2, 2]
    assert not footprint[0, 0, 2]
    assert footprint[1, 0, 2]


def test_known_planar_boundary_gives_a_local_kam_ridge():
    field = np.zeros((5, 15, 15, 2), np.float32)
    field[:, :, 8:, 0] = 1.0
    kam = kam_field(field, np.ones(field.shape[:-1], bool), (1, 1, 1), 1.01, 0)
    assert np.nanmedian(kam[:, :, 7:9]) > 20 * np.nanmedian(kam[:, :, :5])


def test_weak_interface_has_lower_kam_than_strong_interface():
    weak = np.zeros((5, 15, 15, 2), np.float32)
    weak[:, :, 8:, 0] = 0.05
    strong = weak * 10
    mask = np.ones(weak.shape[:-1], bool)
    a = kam_field(weak, mask, (1, 1, 1), 1.01, 0)
    b = kam_field(strong, mask, (1, 1, 1), 1.01, 0)
    np.testing.assert_allclose(
        np.nanmedian(b[:, :, 7:9]), 10 * np.nanmedian(a[:, :, 7:9]), rtol=1e-5
    )


def test_smooth_internal_gradient_produces_nonzero_kam():
    x = np.arange(15, dtype=np.float32)[None, None, :, None]
    field = np.broadcast_to(x, (5, 15, 15, 2)).copy()
    kam = kam_field(field, np.ones(field.shape[:-1], bool), (1, 1, 1), 1.01, 0)
    assert np.nanmedian(kam[:, :, 2:-2]) > 0


def test_partition_metrics_are_exact_for_perfect_labels():
    truth = np.ones((3, 5, 7), np.int32)
    truth[:, :, 4:] = 2
    result = om.evaluate_partition(truth, truth.copy(), (2.0, 1.0, 0.5))
    assert result["ari"] == 1.0
    assert result["vi_total_bits"] < 1e-12
    assert result["cell_count_error"] == 0
    assert result["boundary_f1_at_0p4um"] == 1.0

import math

import numpy as np

from cell_scale_audit import labelled_cell_geometry
from overnight_continue import measurable_proxies


def test_labelled_cell_geometry_known_anisotropic_voxel_volumes():
    labels = np.zeros((2, 3, 4), dtype=np.int32)
    labels[:, :, :2] = 1
    labels[:, :, 2:] = 2
    spacing = (2.0, 1.0, 0.5)
    result, cells, _ = labelled_cell_geometry(labels, spacing)
    assert cells.voxel_count.tolist() == [12, 12]
    assert cells.physical_volume_um3.tolist() == [12.0, 12.0]
    expected = (6.0 * 12.0 / math.pi) ** (1.0 / 3.0)
    assert np.allclose(cells.equivalent_sphere_diameter_um, expected)
    assert result["cell_count_all"] == 2
    assert result["cell_count_interior"] == 0


def test_background_is_excluded_and_interior_cell_is_retained():
    labels = np.zeros((5, 5, 5), dtype=np.int32)
    labels[1:4, 1:4, 1:4] = 7
    result, cells, _ = labelled_cell_geometry(labels, (1.0, 1.0, 1.0))
    assert cells.label.tolist() == [7]
    assert cells.voxel_count.tolist() == [27]
    assert result["cell_count_interior"] == 1
    assert result["equivalent_sphere_diameter_interior_mean_um"] == result["equivalent_sphere_diameter_all_mean_um"]


def test_measurable_cell_scale_is_not_a_fixed_quantile_fraction():
    rng = np.random.default_rng(4)
    coarse = np.repeat(rng.normal(size=(4, 4, 4, 2)), 4, axis=1)
    fine = rng.normal(size=(4, 16, 4, 2))
    a = measurable_proxies(coarse, (1.0, 0.4, 0.4))["cell_scale_proxy_um"]
    b = measurable_proxies(fine, (1.0, 0.4, 0.4))["cell_scale_proxy_um"]
    assert np.isfinite([a, b]).all()
    assert not np.isclose(a, b)

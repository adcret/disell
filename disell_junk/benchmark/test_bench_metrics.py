"""Analytic checks for physical boundary metrics."""

import numpy as np

import bench_metrics


def test_assd_uses_anisotropic_physical_spacing():
    # The interface moves by one voxel along z.  With dz=2 um, both directed
    # mean distances are (0 + 2) / 2 = 1 um, so the pooled ASSD is 1 um.
    truth = np.array([[[1]], [[1]], [[2]], [[2]]], dtype=np.int32)
    prediction = np.array([[[1]], [[1]], [[1]], [[2]]], dtype=np.int32)

    assert bench_metrics.boundary_assd_um(
        truth, prediction, spacing_um_zyx=(2.0, 0.5, 0.5)
    ) == 1.0

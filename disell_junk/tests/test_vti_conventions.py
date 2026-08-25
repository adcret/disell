"""Axis-order and physical-spacing conventions of the VTI export used for
ParaView renderings."""

import sys
from pathlib import Path

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "paper_figures"))
from pf_io import VoxelSpacing, save_vti  # noqa: E402


def test_vti_axis_order_and_spacing(tmp_path):
    Z, Y, X = 2, 3, 4
    arr = np.arange(Z * Y * X, dtype=np.float32).reshape(Z, Y, X)
    spacing = VoxelSpacing(dz_nm=500.0, dy_nm=1240.0, dx_nm=400.0)
    path = tmp_path / "vol.vti"
    save_vti(path, {"data": arr}, spacing=spacing)

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    img = reader.GetOutput()

    assert img.GetDimensions() == (X, Y, Z)
    assert img.GetSpacing() == (400.0, 1240.0, 500.0)

    from vtk.util import numpy_support as vns
    flat = vns.vtk_to_numpy(img.GetPointData().GetArray("data"))
    # VTK flattens x-fastest, which matches C-order (Z, Y, X) ravel
    np.testing.assert_array_equal(flat.reshape(Z, Y, X), arr)

"""Smoke test: ``import disell`` works on a fresh install.

This is a deliberately small test. It exists to catch the three
defects fixed in the previous patch round:

* the Pybind11 module name (``PYBIND11_MODULE(_flood_fill, m)``) must
  match the compiled extension filename, otherwise
  ``from disell._flood_fill import …`` fails at import time;
* ``cell_identification.py`` must use ``from .cell_statistics``, not
  ``from cell_statistics``;
* ``cell_statistics.py`` must use ``from .properties``, not ``from
  properties``.

If any of these regress, this test fails before any other test even
runs. Treat any failure here as a blocker for release.
"""

from __future__ import annotations

import importlib

import pytest


def test_import_disell_top_level():
    """``import disell`` succeeds and exposes the documented public API."""
    disell = importlib.import_module("disell")

    expected_symbols = [
        # cell_identification
        "flood_fill_dfxm",
        "flood_fill_dfxm_two_stage",
        "overtreshold_kam_array",
        "top_down_cell_identification_based_on_misorientation_treshold",
        # cell_statistics
        "neighbour_misorientation",
        "get_cell_size_list",
        "cell_stats_orientation_based",
        # properties
        "kam",
        "batch_erode_labels",
        "batch_dilate_labels",
        # region_growing
        "region_grow_minimum_cell_orientation_differences",
        "region_grow_watershed",
        # registration
        "register_slice_2_volume",
        "register",
        "apply_transforms",
        # visualization
        "export_grain_meshes",
        # _flood_fill C extension
        "flood_fill_random_seeds_3d",
        "flood_fill_collect_seeds",
    ]

    missing = [s for s in expected_symbols if not hasattr(disell, s)]
    assert not missing, (
        f"`import disell` succeeded but the following expected public "
        f"symbols are missing: {missing}. Check src/disell/__init__.py "
        f"and the relative-import patches in cell_identification.py / "
        f"cell_statistics.py."
    )


def test_import_disell_flood_fill_extension():
    """The C++ extension imports as ``disell._flood_fill``.

    This is the single import that would fail with the old
    ``PYBIND11_MODULE(flood_fill, m)``.
    """
    ext = importlib.import_module("disell._flood_fill")
    assert hasattr(ext, "flood_fill_random_seeds_3d"), (
        "disell._flood_fill imported but does not expose "
        "flood_fill_random_seeds_3d. Either the build is broken or the "
        "PYBIND11_MODULE name in src/cpp/bindings.cpp regressed."
    )
    assert hasattr(ext, "flood_fill_collect_seeds")


def test_import_disell_submodules_individually():
    """Each Python submodule imports on its own.

    This guards against the relative-import bug returning: the old
    ``from cell_statistics import …`` worked only by accident if a
    top-level ``cell_statistics`` was on ``sys.path``.
    """
    for name in (
        "disell.cell_identification",
        "disell.cell_statistics",
        "disell.properties",
        "disell.region_growing",
        "disell.registration",
        "disell.visualization",
    ):
        importlib.import_module(name)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

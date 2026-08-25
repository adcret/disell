"""Flood-fill behaviour at volume boundaries and with anisotropic footprints."""

import numpy as np

from disell import flood_fill_dfxm


def test_flood_fill_region_touching_all_boundaries():
    """A uniform field reaching every volume face must segment without
    out-of-bounds access and label the whole (masked) volume as one region."""
    field = np.zeros((3, 8, 9, 2), dtype=np.float32)
    mask = np.ones(field.shape[:3], dtype=np.uint8)
    footprint = np.ones((3, 3, 3), dtype=bool)
    res = flood_fill_dfxm(
        field,
        footprint=footprint,
        local_threshold=0.1,
        mask=mask,
        max_iterations=100,
        min_grain_size=1,
        footprint_tolerance=0.9,
        random_seed=0,
    )
    seg = res["segmentation"]
    assert seg.shape == field.shape[:3]
    assert (seg == 1).all()


def test_anisotropic_footprint_controls_z_connectivity():
    """Two identical slabs stacked in z merge only when the footprint
    provides z connectivity."""
    field = np.zeros((2, 6, 6, 2), dtype=np.float32)
    mask = np.ones(field.shape[:3], dtype=np.uint8)

    fp_inplane = np.zeros((1, 3, 3), dtype=bool)
    fp_inplane[0] = True
    res = flood_fill_dfxm(
        field,
        footprint=fp_inplane,
        local_threshold=0.1,
        mask=mask,
        max_iterations=100,
        min_grain_size=1,
        footprint_tolerance=0.9,
        random_seed=0,
    )
    seg = res["segmentation"]
    # no z connectivity: each slab is its own region
    assert len(np.unique(seg[0])) == 1
    assert len(np.unique(seg[1])) == 1
    assert seg[0, 0, 0] != seg[1, 0, 0]

    fp_z = np.zeros((3, 3, 3), dtype=bool)
    fp_z[1] = True
    fp_z[0, 1, 1] = fp_z[2, 1, 1] = True
    res = flood_fill_dfxm(
        field,
        footprint=fp_z,
        local_threshold=0.1,
        mask=mask,
        max_iterations=100,
        min_grain_size=1,
        footprint_tolerance=0.9,
        random_seed=0,
    )
    assert (res["segmentation"] == 1).all()


def test_min_grain_size_parked_regions_absorbable():
    """Voxels of a rejected small region stay unlabelled (0) but in the
    growth domain, so watershed or later growth can still claim them."""
    field = np.zeros((1, 4, 12, 2), dtype=np.float32)
    field[..., 0] = 0.0
    field[0, :, 8:, 0] = 5.0  # small distinct corner region (16 px)
    mask = np.ones(field.shape[:3], dtype=np.uint8)
    fp = np.ones((1, 3, 3), dtype=bool)
    res = flood_fill_dfxm(
        field,
        footprint=fp,
        local_threshold=0.1,
        mask=mask,
        max_iterations=200,
        min_grain_size=20,
        footprint_tolerance=0.9,
        random_seed=3,
    )
    seg = res["segmentation"]
    # the large region is labelled, the small one parked at 0
    assert (seg[0, :, :6] > 0).all()
    assert (seg[0, :, 9:] == 0).all()

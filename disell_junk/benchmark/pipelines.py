#!/usr/bin/env python3
"""The two segmentation arms compared by the benchmark.

Both arms are built exactly as in the experimental 6.2 % analysis
(``scripts/paper_figures/common.py``): the manuscript KAM (per-channel RMS,
physically isotropic footprint, NaN where undefined), and the same
marker-based watershed on that KAM field for refinement.  The arms differ
only in how the markers are identified:

``flood_fill``
    ``disell.flood_fill_dfxm`` on the angular feature field.

``kam_threshold``
    connected components of the sub-threshold KAM field.

Using one refinement step in both arms keeps the comparison about marker
construction and gives the KAM baseline a complete partition, rather than
scoring it with its wall voxels left unassigned.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def isotropic_footprint(
    spacing_um_zyx: Sequence[float], radius_um: float, ndim: int = 3
) -> np.ndarray:
    """Centred boolean ellipsoid that is a sphere in physical units."""

    spacing = np.asarray(spacing_um_zyx, dtype=np.float64)[:ndim]
    radius = float(radius_um)
    # The tolerance matters: 1.2 / 0.4 is 2.9999999999999996 in binary floating
    # point, so a plain floor silently drops the outermost ring whenever the
    # radius is a whole multiple of the spacing.
    half = np.maximum(np.floor(radius / spacing + 1e-9).astype(int), 1)
    axes = [
        np.arange(-h, h + 1, dtype=np.float64) * s for h, s in zip(half, spacing)
    ]
    grids = np.meshgrid(*axes, indexing="ij")
    footprint = sum(g * g for g in grids) <= radius * radius + 1e-12
    footprint[tuple(half)] = True
    return np.ascontiguousarray(footprint, dtype=bool)


def masked_kam(
    field: np.ndarray, mask: np.ndarray, footprint: np.ndarray
) -> np.ndarray:
    """Manuscript KAM (per-channel RMS) of the masked angular feature field.

    The volume is padded with NaN by the kernel half-width so border voxels
    still get a KAM from their available valid neighbours.  Returns NaN where
    the centre is invalid or has no valid neighbour.
    """

    from disell import kam

    values = field.astype(np.float64).copy()
    values[~mask] = np.nan
    half = [s // 2 for s in footprint.shape]
    padded = np.pad(values, [(h, h) for h in half] + [(0, 0)], constant_values=np.nan)
    result = kam(
        padded,
        ndim=mask.ndim,
        footprint=footprint,
        fill_invalid=np.nan,
        per_channel_rms=True,
    )
    crop = tuple(
        slice(h, result.shape[i] - h if h else None) for i, h in enumerate(half)
    )
    return np.ascontiguousarray(result[crop], dtype=np.float32)


def watershed_elevation(kam_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Finite watershed elevation: undefined KAM is flooded last."""

    elevation = kam_map.astype(np.float32).copy()
    finite = elevation[mask & np.isfinite(elevation)]
    ceiling = float(np.max(finite)) if finite.size else 1.0
    elevation[~np.isfinite(elevation)] = ceiling
    elevation[~mask] = ceiling
    return elevation


def _drop_small_labels(labels: np.ndarray, min_size: int) -> np.ndarray:
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(sizes.size, dtype=bool)
    if sizes.size > 1:
        keep[1:] = sizes[1:] >= int(min_size)
    mapping = np.zeros(sizes.size, dtype=np.int32)
    mapping[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return mapping[labels]


def run_flood_fill(
    field: np.ndarray,
    mask: np.ndarray,
    kam_map: np.ndarray,
    footprint: np.ndarray,
    *,
    local_threshold_deg: float,
    global_threshold_deg: float,
    footprint_tolerance: float,
    min_cell_size: int,
    max_seed_attempts: int,
    stagnation_tolerance: int,
    random_seed: int,
    watershed_connectivity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-seed flood-fill identification followed by KAM-guided watershed."""

    import disell

    result = disell.flood_fill_dfxm(
        np.ascontiguousarray(field, dtype=np.float32),
        footprint=np.ascontiguousarray(footprint, dtype=bool),
        local_threshold=float(local_threshold_deg),
        global_threshold=float(global_threshold_deg),
        footprint_tolerance=float(footprint_tolerance),
        # The C++ kernel consumes the mask in place; hand it a private copy.
        mask=np.ascontiguousarray(mask.astype(np.uint8)),
        max_iterations=int(max_seed_attempts),
        min_grain_size=int(min_cell_size),
        recycle_small_grains=False,
        stagnation_tolerance=int(stagnation_tolerance),
        random_seed=int(random_seed),
    )
    markers = np.asarray(result["segmentation"], dtype=np.int32)
    if int(markers.max()) == 0:
        raise RuntimeError("Flood fill produced no accepted markers.")
    labels = disell.region_grow_watershed(
        markers,
        mask,
        watershed_elevation(kam_map, mask),
        connectivity=int(watershed_connectivity),
    )
    return np.asarray(labels, dtype=np.int32), markers


def run_flood_fill_two_stage(
    field, mask, kam_map, footprint, *, local_threshold_deg,
    global_threshold_deg, footprint_tolerance, min_cell_size,
    max_seed_attempts, stagnation_tolerance, random_seed,
    watershed_connectivity, recycle_small_grains=False,
):
    """Size-prioritised two-pass markers followed by one KAM watershed."""
    import disell

    result, sizes_initial = disell.flood_fill_dfxm_two_stage(
        np.ascontiguousarray(field, dtype=np.float32),
        footprint=np.ascontiguousarray(footprint, dtype=bool),
        local_misorientation_threshold=float(local_threshold_deg),
        global_threshold=float(global_threshold_deg),
        footprint_tolerance=float(footprint_tolerance),
        mask=np.ascontiguousarray(mask.astype(np.uint8)),
        max_iterations=int(max_seed_attempts),
        min_grain_size=int(min_cell_size),
        recycle_small_grains=bool(recycle_small_grains),
        stagnation_tolerance=int(stagnation_tolerance),
        random_seed=int(random_seed),
    )
    markers = np.asarray(result["segmentation"], dtype=np.int32)
    if int(markers.max()) == 0:
        # Preserve candidate/final-pass diagnostics for an expected invalid
        # configuration.  The caller must classify this and must not treat the
        # all-zero placeholder as a completed watershed partition.
        return (np.zeros_like(markers), markers, np.asarray(sizes_initial),
                result.get("diagnostics", {}),
                np.asarray(result.get("sizes", [])))
    labels = disell.region_grow_watershed(
        markers, mask, watershed_elevation(kam_map, mask),
        connectivity=int(watershed_connectivity),
    )
    return (np.asarray(labels, dtype=np.int32), markers,
            np.asarray(sizes_initial), result.get("diagnostics", {}),
            np.asarray(result.get("sizes", [])))


def run_kam_threshold(
    kam_map: np.ndarray,
    mask: np.ndarray,
    *,
    percentile: float,
    min_cell_size: int,
    connectivity: int,
    watershed_connectivity: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Low-KAM connected components followed by the same watershed."""

    import disell
    from scipy.ndimage import generate_binary_structure, label

    valid = mask & np.isfinite(kam_map)
    values = kam_map[valid]
    if values.size == 0:
        raise RuntimeError("No valid KAM values inside the analysis mask.")
    threshold = float(np.percentile(values, float(percentile)))

    structure = generate_binary_structure(kam_map.ndim, int(connectivity))
    markers, _ = label(valid & (kam_map < threshold), structure=structure)
    markers = _drop_small_labels(markers.astype(np.int32), min_cell_size)
    if int(markers.max()) == 0:
        raise RuntimeError("KAM thresholding produced no accepted markers.")
    labels = disell.region_grow_watershed(
        markers,
        mask,
        watershed_elevation(kam_map, mask),
        connectivity=int(watershed_connectivity),
    )
    return np.asarray(labels, dtype=np.int32), markers, threshold

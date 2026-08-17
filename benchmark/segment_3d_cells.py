#!/usr/bin/env python3
"""Run the full ``disell`` 3D segmentation pipeline on a DFXM volume.

Pipeline:

1. load a 3D ``(Z, Y, X, 2)`` orientation volume + ``(Z, Y, X)`` mask
   (via ``darling.DataSet`` + ``darling.properties.mean`` by default;
   a fallback raw-HDF5 path exists but should be avoided),
2. (optionally) smooth the orientation field with an anisotropic
   per-channel median filter,
3. compute the 3D KAM scalar field,
4. run the multi-seed flood-fill identification step (deterministic
   when ``--random-seed`` is passed),
5. run the marker-based watershed refinement on the KAM field,
6. save labels, KAM, and markers to HDF5 and VTI, write diagnostic
   slice overlays, and dump every parameter into
   ``parameters_used.json``.

Important notes (see ``AUDIT.md`` for context):

* The C++ flood-fill in ``disell._flood_fill`` does not perform
  in-bounds checks on neighbour indices (B5). To avoid out-of-bounds
  reads and accidental wrap-around merging, this script pads the
  property map and the mask by the maximum footprint half-extent
  before calling the C++ kernel and crops the labels back to the
  original shape afterwards.
* The current public ``flood_fill_dfxm_two_stage`` wrapper passes an integer
  ``random_seed`` to both candidate collection and final growth; only
  ``random_seed=None`` falls back to ``std::random_device``. This standalone
  lower-level script predates that wrapper and remains deterministic by
  sampling candidate points with ``numpy.random.default_rng(seed)`` and
  supplying them explicitly as ``seed_points``.
* The KAM kernel can be made anisotropy-aware via ``--kam-radius-nm``
  if voxel spacing is anisotropic. By default this script honours the
  voxel spacing.

Run ``--help`` for the full CLI.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _io import (  # noqa: E402  (path manipulation above)
    OrientationVolume,
    VoxelSpacing,
    isotropic_physical_footprint,
    load_orientation_volume_from_darling,
    load_orientation_volume_from_h5,
    median_filter_orientation,
    save_h5_volume,
    save_vti,
    write_parameters_json,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("data source (one of --darling-* OR --h5-*)")
    src.add_argument("--input-h5", type=str, required=True,
                     help="Path to the ID03 HDF5 file (darling) or the "
                          "preprocessed orientation HDF5 file (raw).")
    src.add_argument("--scan-ids", nargs="+",
                     help="One or more darling scan ids (e.g. 1.1 2.1 3.1) "
                          "stacked along Z.")
    src.add_argument("--orientation-method", choices=["mean", "peak"],
                     default="mean",
                     help="darling.properties function to use for the "
                          "orientation map (default: mean).")
    src.add_argument("--roi", nargs=4, type=int, metavar=("R0", "R1", "C0", "C1"),
                     default=None,
                     help="Detector ROI passed to darling.")
    src.add_argument("--intensity-threshold", type=float, default=None,
                     help="If set, voxels with summed intensity below this "
                          "value are masked out by darling-loader.")
    src.add_argument("--no-darling", action="store_true",
                     help="Disable darling and read a preprocessed orientation "
                          "field directly from HDF5.")
    src.add_argument("--field-dataset", type=str, default=None,
                     help="HDF5 path to the (Z, Y, X, 2) orientation field. "
                          "Required when --no-darling is set.")
    src.add_argument("--mask-dataset", type=str, default=None,
                     help="HDF5 path to the (Z, Y, X) mask. Optional even "
                          "with --no-darling (mask is then derived from "
                          "isfinite of the field).")

    geom = p.add_argument_group("geometry")
    geom.add_argument("--voxel-spacing-nm", nargs=3, type=float, required=True,
                      metavar=("DZ", "DY", "DX"),
                      help="Voxel spacing in nanometres in (z, y, x) order.")
    geom.add_argument("--axis-order", choices=["ZYX"], default="ZYX",
                      help="Spatial axis order. Only ZYX is supported here.")
    geom.add_argument("--angle-unit", choices=["deg", "rad", "mrad"],
                      required=True,
                      help="Unit of the orientation field. Recorded in every "
                           "output file.")

    smooth = p.add_argument_group("preprocessing")
    smooth.add_argument("--smoothing-kernel-zyx", nargs=3, type=int,
                        default=(1, 3, 3), metavar=("KZ", "KY", "KX"),
                        help="Per-channel median filter kernel (default: 1 3 3, "
                             "leaving Z untouched -- typical for thin DFXM "
                             "layers).")
    smooth.add_argument("--no-smoothing", action="store_true",
                        help="Skip the median filter step.")

    kam = p.add_argument_group("KAM")
    kam.add_argument("--kam-radius-nm", type=float, default=None,
                     help="If set, build a physically-isotropic KAM kernel "
                          "from --voxel-spacing-nm with this radius. "
                          "Overrides --kam-kernel-zyx.")
    kam.add_argument("--kam-kernel-zyx", nargs=3, type=int,
                     default=(3, 5, 5), metavar=("KZ", "KY", "KX"),
                     help="KAM kernel size in voxels (must be odd). Default: "
                          "(3, 5, 5) -- a reasonable choice when dz_nm > "
                          "dy_nm = dx_nm.")

    ff = p.add_argument_group("multi-seed flood fill")
    ff.add_argument("--ff-radius-nm", type=float, default=None,
                    help="Build a physical-isotropic flood-fill footprint with "
                         "this radius. Overrides --ff-kernel-zyx.")
    ff.add_argument("--ff-kernel-zyx", nargs=3, type=int,
                    default=(3, 3, 3), metavar=("KZ", "KY", "KX"),
                    help="Flood-fill footprint size in voxels (default 3 3 3).")
    ff.add_argument("--local-threshold", type=float, required=True,
                    help="Local misorientation threshold tau_loc (per channel, "
                         "in --angle-unit). NB: the C++ kernel implements a "
                         "sum-of-squares criterion thr^2 * C; the same value "
                         "supplied here will mean what Algorithm 2 of the "
                         "manuscript intends only if 'local-threshold' is "
                         "interpreted per-channel.")
    ff.add_argument("--footprint-tolerance", type=float, default=0.85,
                    help="Footprint tolerance tau_fp in [0, 1] (default 0.85).")
    ff.add_argument("--min-cell-size", type=int, default=200,
                    help="Minimum region size in voxels (default 200).")
    ff.add_argument("--max-seed-attempts", type=int, default=10000,
                    help="Maximum number of seeds to draw (default 10000).")
    ff.add_argument("--stagnation-tolerance", type=int, default=2000,
                    help="Number of consecutive failed seed draws before "
                         "early termination (default 2000).")
    ff.add_argument("--random-seed", type=int, required=True,
                    help="Random seed for reproducible seed sampling. The "
                         "value is recorded in parameters_used.json.")

    ws = p.add_argument_group("watershed")
    ws.add_argument("--watershed-connectivity", type=int, default=1,
                    choices=[1, 2, 3],
                    help="skimage watershed connectivity (1=face, 2=edge, "
                         "3=corner). Default 1.")

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", type=str, required=True,
                     help="Output directory. Created if missing.")
    out.add_argument("--diagnostic-slices", nargs="+", type=int, default=None,
                     help="Z indices for which to render diagnostic 2D "
                          "overlays. Default: 5 evenly spaced slices.")

    return p


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def load_volume(args: argparse.Namespace) -> OrientationVolume:
    spacing = VoxelSpacing(*args.voxel_spacing_nm)
    if args.no_darling:
        if args.field_dataset is None:
            raise SystemExit(
                "--no-darling requires --field-dataset HDFPATH"
            )
        return load_orientation_volume_from_h5(
            args.input_h5,
            args.field_dataset,
            mask_dataset=args.mask_dataset,
            spacing=spacing,
            angle_unit=args.angle_unit,
        )

    if not args.scan_ids:
        raise SystemExit(
            "--scan-ids is required for the darling loading path. "
            "Pass one scan id per Z layer."
        )
    return load_orientation_volume_from_darling(
        args.input_h5,
        args.scan_ids,
        spacing=spacing,
        angle_unit=args.angle_unit,
        method=args.orientation_method,
        roi=tuple(args.roi) if args.roi else None,
        intensity_threshold=args.intensity_threshold,
    )


def compute_kam(
    volume: OrientationVolume, args: argparse.Namespace
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Compute KAM with an explicit, recorded kernel size."""
    from disell.properties import kam as kam_fn

    if args.kam_radius_nm is not None:
        footprint = isotropic_physical_footprint(
            volume.spacing, radius_nm=args.kam_radius_nm, ndim=3
        )
        kam_size = footprint.shape
    else:
        kam_size = tuple(int(s) for s in args.kam_kernel_zyx)
        for s in kam_size:
            if s < 3 or s % 2 == 0:
                raise SystemExit(
                    f"KAM kernel sizes must be odd integers >=3; got {kam_size}"
                )

    kam_field = kam_fn(volume.field, ndim=3, size=kam_size).astype(np.float32)
    # KAM is undefined inside the kernel border (returned as zeros). Mark
    # those voxels as NaN so downstream code does not mistake them for
    # cell interiors.
    nan_border = np.zeros(kam_field.shape, dtype=bool)
    rz, ry, rx = (s // 2 for s in kam_size)
    nan_border[:rz] = True
    nan_border[-rz:] = True
    nan_border[:, :ry] = True
    nan_border[:, -ry:] = True
    nan_border[..., :rx] = True
    nan_border[..., -rx:] = True
    kam_field = np.where(nan_border, np.nan, kam_field)
    return kam_field, kam_size


def _build_ff_footprint(
    volume: OrientationVolume, args: argparse.Namespace
) -> np.ndarray:
    if args.ff_radius_nm is not None:
        return isotropic_physical_footprint(
            volume.spacing, radius_nm=args.ff_radius_nm, ndim=3
        )
    sizes = tuple(int(s) for s in args.ff_kernel_zyx)
    for s in sizes:
        if s < 3 or s % 2 == 0:
            raise SystemExit(
                f"Flood-fill footprint sizes must be odd >=3; got {sizes}"
            )
    return np.ones(sizes, dtype=bool)


def _sample_deterministic_seeds(
    mask: np.ndarray, n_seeds: int, rng: np.random.Generator
) -> np.ndarray:
    """Pre-sample ``(N, 3)`` seed coordinates uniformly at random from ``mask``."""
    in_mask = np.flatnonzero(mask.ravel())
    if in_mask.size == 0:
        raise RuntimeError("Cannot sample seeds: mask is empty.")
    n = min(n_seeds, in_mask.size)
    chosen = rng.choice(in_mask, size=n, replace=False)
    Z, Y, X = mask.shape
    z = chosen // (Y * X)
    y = (chosen // X) % Y
    x = chosen % X
    return np.stack([z, y, x], axis=-1).astype(np.int64)


def run_multiseed_flood_fill(
    volume: OrientationVolume,
    footprint: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Run the deterministic, padded multi-seed flood fill.

    Mitigations applied here:

    * pad property map and mask by ``max(footprint half-extent)`` to
      avoid the C++ out-of-bounds read on neighbours;
    * pre-sample seeds in numpy with ``args.random_seed`` so the
      C++ random draw is bypassed.
    """
    from disell import _flood_fill as ff_cpp

    half = tuple(s // 2 for s in footprint.shape)
    pad = ((half[0],) * 2, (half[1],) * 2, (half[2],) * 2)

    field = np.ascontiguousarray(volume.field, dtype=np.float32)
    mask = np.ascontiguousarray(volume.mask.astype(np.uint8))

    field_padded = np.pad(field, pad + ((0, 0),), mode="constant",
                          constant_values=0.0)
    mask_padded = np.pad(mask, pad, mode="constant", constant_values=0)

    rng = np.random.default_rng(args.random_seed)
    seeds_padded = _sample_deterministic_seeds(
        mask_padded.astype(bool),
        n_seeds=args.max_seed_attempts,
        rng=rng,
    )

    # Important: the C++ wrapper *mutates* the mask in place. We pass a
    # copy so the caller's mask is preserved.
    mask_for_cpp = mask_padded.copy()

    result = ff_cpp.flood_fill_random_seeds_3d(
        field_padded,
        footprint.astype(bool),
        float(args.local_threshold),
        -1.0,
        float(args.footprint_tolerance),
        mask_for_cpp,
        int(args.max_seed_attempts),
        int(args.min_cell_size),
        False,                       # recycle_small_grains
        int(args.stagnation_tolerance),
        seeds_padded,
    )

    seg_padded = np.asarray(result["segmentation"], dtype=np.int32)
    sz, sy, sx = half
    if sz == 0 and sy == 0 and sx == 0:
        seg = seg_padded
    else:
        seg = seg_padded[
            sz: seg_padded.shape[0] - sz,
            sy: seg_padded.shape[1] - sy,
            sx: seg_padded.shape[2] - sx,
        ].copy()
    return seg


def run_watershed(
    markers: np.ndarray,
    mask: np.ndarray,
    kam_field: np.ndarray,
    *,
    connectivity: int,
) -> np.ndarray:
    from disell.region_growing import region_grow_watershed

    feature = np.where(np.isnan(kam_field), np.nanmax(kam_field), kam_field)
    feature = feature.astype(np.float32, copy=False)

    labels = region_grow_watershed(
        markers.astype(np.int32),
        mask.astype(bool),
        feature,
        connectivity=connectivity,
    ).astype(np.int32)
    return labels


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _set_paper_matplotlib_defaults() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def render_diagnostic_slices(
    out_dir: Path,
    volume: OrientationVolume,
    labels: np.ndarray,
    kam_field: np.ndarray,
    z_indices,
) -> None:
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    _set_paper_matplotlib_defaults()
    chi = volume.field[..., 0]
    phi = volume.field[..., 1]

    for z in z_indices:
        z = int(z)
        if not 0 <= z < volume.shape_zyx[0]:
            print(f"[warn] slice index {z} out of range; skipping")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

        for ax, arr, title in zip(
            axes,
            [chi[z], phi[z], kam_field[z]],
            [f"chi [{volume.angle_unit}]",
             f"phi [{volume.angle_unit}]",
             f"KAM [{volume.angle_unit}/voxel]"],
        ):
            arr = np.where(volume.mask[z], arr, np.nan)
            im = ax.imshow(
                arr, origin="lower", cmap="viridis", interpolation="nearest"
            )
            boundaries = find_boundaries(labels[z], mode="thick")
            ax.contour(
                boundaries.astype(int),
                levels=[0.5], colors="black", linewidths=0.4,
            )
            ax.set_title(f"{title}  z={z}")
            ax.set_xlabel("x [voxels]")
            ax.set_ylabel("y [voxels]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(
                out_dir / f"segmentation_overlay_slice_{z:04d}.{ext}",
                bbox_inches="tight",
            )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume(args)

    if not args.no_smoothing:
        smoothed = median_filter_orientation(
            volume.field, kernel_zyx=tuple(args.smoothing_kernel_zyx)
        )
        volume = OrientationVolume(
            field=smoothed,
            mask=volume.mask,
            spacing=volume.spacing,
            angle_unit=volume.angle_unit,
            source={**volume.source, "smoothing_kernel_zyx":
                    tuple(args.smoothing_kernel_zyx)},
        )

    kam_field, kam_size = compute_kam(volume, args)

    footprint = _build_ff_footprint(volume, args)
    markers = run_multiseed_flood_fill(volume, footprint, args)
    n_markers = int(markers.max())

    labels = run_watershed(
        markers,
        volume.mask,
        kam_field,
        connectivity=args.watershed_connectivity,
    )
    n_labels = int(labels.max())

    save_h5_volume(
        out_dir / "labels_3d.h5", labels, dataset="labels",
        spacing=volume.spacing, angle_unit=volume.angle_unit,
        extra_attrs={
            "n_labels": n_labels,
            "n_markers": n_markers,
            "method": "multiseed_floodfill+watershed",
        },
    )
    save_h5_volume(
        out_dir / "markers_3d.h5", markers.astype(np.int32),
        dataset="markers", spacing=volume.spacing,
        angle_unit=volume.angle_unit,
        extra_attrs={"n_markers": n_markers,
                     "method": "multiseed_floodfill"},
    )
    save_h5_volume(
        out_dir / "kam_3d.h5", kam_field.astype(np.float32),
        dataset="kam", spacing=volume.spacing,
        angle_unit=volume.angle_unit,
        extra_attrs={
            "kam_kernel_zyx": np.array(kam_size, int),
        },
    )
    save_vti(
        out_dir / "segmentation_3d.vti",
        {
            "labels": labels.astype(np.int32),
            "markers": markers.astype(np.int32),
            "kam": np.where(np.isnan(kam_field), 0.0, kam_field).astype(np.float32),
            "mask": volume.mask.astype(np.uint8),
        },
        spacing=volume.spacing,
    )

    if args.diagnostic_slices is None:
        Z = volume.shape_zyx[0]
        z_indices = np.linspace(0, Z - 1, num=min(5, Z), dtype=int)
    else:
        z_indices = list(args.diagnostic_slices)
    render_diagnostic_slices(out_dir, volume, labels, kam_field, z_indices)

    write_parameters_json(
        out_dir,
        {
            "script": "segment_3d_cells.py",
            "args": vars(args),
            "kam_kernel_zyx_actual": list(kam_size),
            "ff_footprint_shape": list(footprint.shape),
            "n_markers": n_markers,
            "n_labels": n_labels,
            "voxel_spacing_nm_zyx": list(volume.spacing.as_tuple_nm()),
            "angle_unit": volume.angle_unit,
            "data_source": volume.source,
        },
    )

    print(
        f"[done] wrote {n_markers} markers and {n_labels} labels "
        f"to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

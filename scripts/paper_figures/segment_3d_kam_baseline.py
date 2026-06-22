#!/usr/bin/env python3
"""3D KAM-thresholding baseline (the manuscript's "fails in 3D" path).

This script is intentionally minimalist. It reproduces the baseline that
the paper compares the new method against, with one critical difference:
the threshold is exposed as a CLI parameter and the script computes,
for each threshold, both the resulting labelling *and* a small CSV with
connected-component statistics (count, max-component fraction,
size distribution). The output is what the manuscript needs to make the
"KAM fails in 3D" claim quantitative rather than visual.

Pipeline:

1. load the same DFXM orientation volume as ``segment_3d_cells.py``,
2. (optionally) smooth with the same anisotropic median filter,
3. compute KAM,
4. threshold the KAM field at a given percentile of in-mask values,
5. apply 3D binary erosion + dilation (the EBSD-style cleanup step
   used in the 2D paper figure),
6. compute connected components (via ``scipy.ndimage.label``,
   anisotropy-aware connectivity),
7. drop components below ``--min-component-size`` voxels,
8. save labels + KAM + diagnostics + CSV statistics + ParaView VTI.

This script does NOT itself sweep the threshold (call it from a shell
loop or use ``compare_3d_methods.py``).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _io import (  # noqa: E402
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("data source")
    src.add_argument("--input-h5", type=str, required=True)
    src.add_argument("--scan-ids", nargs="+")
    src.add_argument("--orientation-method", choices=["mean", "peak"],
                     default="mean")
    src.add_argument("--roi", nargs=4, type=int, default=None,
                     metavar=("R0", "R1", "C0", "C1"))
    src.add_argument("--intensity-threshold", type=float, default=None)
    src.add_argument("--no-darling", action="store_true")
    src.add_argument("--field-dataset", type=str, default=None)
    src.add_argument("--mask-dataset", type=str, default=None)

    geom = p.add_argument_group("geometry")
    geom.add_argument("--voxel-spacing-nm", nargs=3, type=float, required=True,
                      metavar=("DZ", "DY", "DX"))
    geom.add_argument("--axis-order", choices=["ZYX"], default="ZYX")
    geom.add_argument("--angle-unit", choices=["deg", "rad", "mrad"],
                      required=True)

    smooth = p.add_argument_group("preprocessing")
    smooth.add_argument("--smoothing-kernel-zyx", nargs=3, type=int,
                        default=(1, 3, 3), metavar=("KZ", "KY", "KX"))
    smooth.add_argument("--no-smoothing", action="store_true")

    kam = p.add_argument_group("KAM")
    kam.add_argument("--kam-radius-nm", type=float, default=None)
    kam.add_argument("--kam-kernel-zyx", nargs=3, type=int,
                     default=(3, 5, 5), metavar=("KZ", "KY", "KX"))

    base = p.add_argument_group("KAM-thresholding baseline")
    base.add_argument(
        "--threshold-percentile", type=float, required=True,
        help="Percentile (in [0,100]) of in-mask KAM values used as the "
             "threshold. The paper baseline uses the *complement*, e.g. "
             "0.71 in the source means 'keep top 29%'; here we use the "
             "more transparent 'percentile = 71 keeps voxels with KAM >= "
             "the 71st percentile = top 29%'.",
    )
    base.add_argument(
        "--erosion-iterations", type=int, default=1,
        help="Iterations of 3D binary erosion applied to the thresholded "
             "mask before dilation. Mirrors the 2D paper baseline.",
    )
    base.add_argument(
        "--dilation-iterations", type=int, default=1,
        help="Iterations of 3D binary dilation applied after erosion.",
    )
    base.add_argument(
        "--connectivity", type=int, choices=[1, 2, 3], default=1,
        help="3D connectivity for connected components (1=face, 2=edge, "
             "3=corner). Default 1.",
    )
    base.add_argument(
        "--min-component-size", type=int, default=200,
        help="Minimum component size in voxels (smaller components are "
             "dropped).",
    )

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", type=str, required=True)
    out.add_argument("--diagnostic-slices", nargs="+", type=int, default=None)

    return p


def load_volume(args) -> OrientationVolume:
    spacing = VoxelSpacing(*args.voxel_spacing_nm)
    if args.no_darling:
        if args.field_dataset is None:
            raise SystemExit("--no-darling requires --field-dataset HDFPATH")
        return load_orientation_volume_from_h5(
            args.input_h5, args.field_dataset,
            mask_dataset=args.mask_dataset,
            spacing=spacing, angle_unit=args.angle_unit,
        )
    if not args.scan_ids:
        raise SystemExit("--scan-ids is required without --no-darling")
    return load_orientation_volume_from_darling(
        args.input_h5, args.scan_ids,
        spacing=spacing, angle_unit=args.angle_unit,
        method=args.orientation_method,
        roi=tuple(args.roi) if args.roi else None,
        intensity_threshold=args.intensity_threshold,
    )


def compute_kam(
    volume: OrientationVolume, args
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    from disell.properties import kam as kam_fn
    if args.kam_radius_nm is not None:
        footprint = isotropic_physical_footprint(
            volume.spacing, radius_nm=args.kam_radius_nm, ndim=3
        )
        size = footprint.shape
    else:
        size = tuple(int(s) for s in args.kam_kernel_zyx)
    kam_field = kam_fn(volume.field, ndim=3, size=size).astype(np.float32)
    rz, ry, rx = (s // 2 for s in size)
    nan_border = np.zeros(kam_field.shape, dtype=bool)
    nan_border[:rz] = True
    nan_border[-rz:] = True
    nan_border[:, :ry] = True
    nan_border[:, -ry:] = True
    nan_border[..., :rx] = True
    nan_border[..., -rx:] = True
    kam_field = np.where(nan_border, np.nan, kam_field)
    return kam_field, size


def threshold_and_label(
    kam_field: np.ndarray,
    mask: np.ndarray,
    *,
    percentile: float,
    erode_iter: int,
    dilate_iter: int,
    connectivity: int,
    min_size: int,
):
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
    from scipy.ndimage import label as cc_label

    masked_kam = kam_field[mask & np.isfinite(kam_field)]
    if masked_kam.size == 0:
        raise RuntimeError("No valid KAM values inside mask.")
    threshold_value = float(np.percentile(masked_kam, percentile))

    raw = (kam_field >= threshold_value) & mask & np.isfinite(kam_field)

    structure = generate_binary_structure(3, connectivity)
    eroded = binary_erosion(raw, structure=structure, iterations=erode_iter) if erode_iter > 0 else raw
    cleaned = binary_dilation(eroded, structure=structure, iterations=dilate_iter) if dilate_iter > 0 else eroded

    # The paper uses the *complement* as cell-interior candidates: cells
    # are the regions that are NOT cell-wall.
    interiors = mask & ~cleaned
    labels, n_initial = cc_label(interiors, structure=structure)

    sizes = np.bincount(labels.ravel())
    if sizes.size == 0:
        return labels.astype(np.int32), threshold_value, raw, cleaned, sizes

    # Drop too-small components
    keep = np.zeros(sizes.size, dtype=bool)
    keep[1:] = sizes[1:] >= min_size
    relabel = np.zeros(sizes.size, dtype=np.int32)
    relabel[keep] = np.arange(1, keep.sum() + 1, dtype=np.int32)
    new_labels = relabel[labels]

    return new_labels.astype(np.int32), threshold_value, raw, cleaned, sizes


def write_component_stats(
    out_csv: Path,
    labels: np.ndarray,
    spacing: VoxelSpacing,
    *,
    mask: np.ndarray,
    threshold_value: float,
    threshold_percentile: float,
) -> None:
    voxel_volume_nm3 = spacing.voxel_volume_nm3()
    sizes_voxels = np.bincount(labels.ravel())
    background = int(sizes_voxels[0]) if sizes_voxels.size > 0 else 0
    cell_sizes_voxels = sizes_voxels[1:] if sizes_voxels.size > 1 else np.array([], dtype=int)

    n = int(cell_sizes_voxels.size)
    in_mask_voxels = int(mask.sum())
    largest_voxels = int(cell_sizes_voxels.max()) if n > 0 else 0

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label", "size_voxels", "size_nm3", "size_um3",
            ]
        )
        for lbl, sv in enumerate(cell_sizes_voxels, start=1):
            w.writerow([
                int(lbl),
                int(sv),
                float(sv) * voxel_volume_nm3,
                float(sv) * voxel_volume_nm3 * 1e-9,
            ])

    summary = out_csv.with_suffix(".summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "threshold_percentile",
            "threshold_value",
            "n_components",
            "in_mask_voxels",
            "largest_component_voxels",
            "largest_component_fraction",
            "background_voxels",
            "voxel_volume_nm3",
        ])
        w.writerow([
            float(threshold_percentile),
            float(threshold_value),
            int(n),
            int(in_mask_voxels),
            int(largest_voxels),
            float(largest_voxels) / max(in_mask_voxels, 1),
            int(background),
            float(voxel_volume_nm3),
        ])


def render_diagnostic_slices(
    out_dir: Path,
    volume: OrientationVolume,
    labels: np.ndarray,
    kam_field: np.ndarray,
    z_indices,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from skimage.segmentation import find_boundaries

    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.grid": False,
        "pdf.fonttype": 42,
    })

    chi = volume.field[..., 0]
    phi = volume.field[..., 1]
    for z in z_indices:
        z = int(z)
        if not 0 <= z < volume.shape_zyx[0]:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
        for ax, arr, title in zip(
            axes,
            [chi[z], phi[z], kam_field[z]],
            [f"chi [{volume.angle_unit}]",
             f"phi [{volume.angle_unit}]",
             "KAM"],
        ):
            arr = np.where(volume.mask[z], arr, np.nan)
            im = ax.imshow(arr, origin="lower", cmap="viridis",
                           interpolation="nearest")
            boundaries = find_boundaries(labels[z], mode="thick")
            ax.contour(boundaries.astype(int), levels=[0.5],
                       colors="black", linewidths=0.4)
            ax.set_title(f"{title}  z={z}")
            ax.set_xlabel("x [voxels]")
            ax.set_ylabel("y [voxels]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(
                out_dir / f"kam_threshold_overlay_slice_{z:04d}.{ext}",
                bbox_inches="tight",
            )
        plt.close(fig)


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
            field=smoothed, mask=volume.mask, spacing=volume.spacing,
            angle_unit=volume.angle_unit, source={**volume.source,
                "smoothing_kernel_zyx": tuple(args.smoothing_kernel_zyx)},
        )

    kam_field, kam_size = compute_kam(volume, args)

    labels, thr, raw_mask, cleaned_mask, sizes_full = threshold_and_label(
        kam_field=kam_field,
        mask=volume.mask,
        percentile=float(args.threshold_percentile),
        erode_iter=int(args.erosion_iterations),
        dilate_iter=int(args.dilation_iterations),
        connectivity=int(args.connectivity),
        min_size=int(args.min_component_size),
    )

    save_h5_volume(
        out_dir / "kam_threshold_labels_3d.h5",
        labels.astype(np.int32),
        dataset="labels",
        spacing=volume.spacing,
        angle_unit=volume.angle_unit,
        extra_attrs={
            "method": "kam_threshold",
            "threshold_percentile": float(args.threshold_percentile),
            "threshold_value": float(thr),
            "kam_kernel_zyx": np.array(kam_size, int),
            "erosion_iterations": int(args.erosion_iterations),
            "dilation_iterations": int(args.dilation_iterations),
            "connectivity": int(args.connectivity),
            "min_component_size": int(args.min_component_size),
        },
    )
    save_h5_volume(
        out_dir / "kam_3d.h5",
        kam_field.astype(np.float32),
        dataset="kam",
        spacing=volume.spacing,
        angle_unit=volume.angle_unit,
        extra_attrs={"kam_kernel_zyx": np.array(kam_size, int)},
    )
    save_vti(
        out_dir / "kam_threshold_3d.vti",
        {
            "labels": labels.astype(np.int32),
            "wall_mask": cleaned_mask.astype(np.uint8),
            "raw_threshold": raw_mask.astype(np.uint8),
            "kam": np.where(np.isnan(kam_field), 0.0, kam_field).astype(np.float32),
            "mask": volume.mask.astype(np.uint8),
        },
        spacing=volume.spacing,
    )

    write_component_stats(
        out_dir / "kam_connected_component_stats.csv",
        labels=labels,
        spacing=volume.spacing,
        mask=volume.mask,
        threshold_value=float(thr),
        threshold_percentile=float(args.threshold_percentile),
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
            "script": "segment_3d_kam_baseline.py",
            "args": vars(args),
            "kam_kernel_zyx_actual": list(kam_size),
            "threshold_value": float(thr),
            "n_components": int(labels.max()),
            "voxel_spacing_nm_zyx": list(volume.spacing.as_tuple_nm()),
            "angle_unit": volume.angle_unit,
            "data_source": volume.source,
        },
    )

    print(
        f"[done] threshold_percentile={args.threshold_percentile} "
        f"threshold_value={thr:.4g} n_components={int(labels.max())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

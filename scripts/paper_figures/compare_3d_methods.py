#!/usr/bin/env python3
"""Quantitatively compare ``disell`` 3D segmentation against the KAM
baseline on the same volume.

This script consumes the labels produced by ``segment_3d_cells.py`` and
``segment_3d_kam_baseline.py`` (passed as HDF5 paths). It deliberately
does *not* re-run the segmentation: the user is in charge of choosing
the operating point for each method (threshold percentile for KAM,
``--local-threshold`` etc. for the flood-fill method) and this script
makes the comparison reproducible at *those* operating points.

Per-method metrics computed:

* number of cells (components above ``--min-cell-size``)
* size distribution: voxel counts, physical volumes (nm^3 and um^3),
  equivalent diameters d = (6V/pi)^(1/3),
* fraction of in-mask voxels in the largest component (the
  "percolation" indicator),
* per-cell intra-cell orientation variance ``E_inner``
  (chi/phi component variance + total),
* per-cell boundary-KAM mean and median ``E_boundary``,
* a log-normal fit of cell volumes (where applicable),
* representative slice overlays at user-chosen Z indices, with
  identical extents and colour limits across methods.

The output is a CSV table, four PDF/PNG plots, and (optionally) one
overlay PDF per chosen slice index.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _io import (  # noqa: E402
    OrientationVolume,
    VoxelSpacing,
    load_orientation_volume_from_darling,
    load_orientation_volume_from_h5,
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
    src = p.add_argument_group("data source (must match both runs)")
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
    src.add_argument("--voxel-spacing-nm", nargs=3, type=float, required=True,
                     metavar=("DZ", "DY", "DX"))
    src.add_argument("--angle-unit", choices=["deg", "rad", "mrad"],
                     required=True)

    inp = p.add_argument_group("inputs from the two segmentation scripts")
    inp.add_argument("--floodfill-labels-h5", type=str, required=True,
                     help="Path to labels_3d.h5 from segment_3d_cells.py")
    inp.add_argument("--floodfill-kam-h5", type=str, required=True,
                     help="Path to kam_3d.h5 from segment_3d_cells.py")
    inp.add_argument("--kam-baseline-labels-h5", type=str, required=True,
                     help="Path to kam_threshold_labels_3d.h5 from "
                          "segment_3d_kam_baseline.py")

    cmp = p.add_argument_group("comparison parameters")
    cmp.add_argument("--min-cell-size", type=int, default=200,
                     help="Components below this voxel count are excluded "
                          "from per-method statistics.")
    cmp.add_argument("--exclude-edge-touching", action="store_true",
                     help="Drop cells that touch the volume border. Useful "
                          "to avoid size-bias.")
    cmp.add_argument("--lognormal-fit", action="store_true",
                     help="Fit a log-normal distribution to cell volumes.")
    cmp.add_argument("--slice-indices", nargs="+", type=int, default=None,
                     help="Slice z indices for overlay figures. Default: 3 "
                          "evenly spaced.")
    cmp.add_argument("--out-dir", type=str, required=True)
    return p


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


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


def load_h5_labels(path: Path, dataset: str = "labels") -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        if dataset not in f:
            raise KeyError(f"{dataset} not found in {path}; keys: {list(f.keys())}")
        return np.asarray(f[dataset][...]).astype(np.int32)


def load_h5_scalar(path: Path, dataset: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        if dataset not in f:
            raise KeyError(f"{dataset} not found in {path}")
        return np.asarray(f[dataset][...]).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-method statistics
# ---------------------------------------------------------------------------


def _drop_edge_touching(labels: np.ndarray) -> np.ndarray:
    """Return ``labels`` with all components touching the volume boundary set to 0."""
    edge = set()
    edge.update(np.unique(labels[0]))
    edge.update(np.unique(labels[-1]))
    edge.update(np.unique(labels[:, 0, :]))
    edge.update(np.unique(labels[:, -1, :]))
    edge.update(np.unique(labels[:, :, 0]))
    edge.update(np.unique(labels[:, :, -1]))
    edge.discard(0)
    if not edge:
        return labels
    drop_mask = np.isin(labels, np.fromiter(edge, dtype=labels.dtype))
    out = labels.copy()
    out[drop_mask] = 0
    return out


def _filter_by_size(labels: np.ndarray, min_size: int) -> np.ndarray:
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(sizes.size, dtype=bool)
    if sizes.size > 1:
        keep[1:] = sizes[1:] >= min_size
    relabel = np.zeros(sizes.size, dtype=np.int32)
    relabel[keep] = np.arange(1, keep.sum() + 1, dtype=np.int32)
    return relabel[labels]


def _per_cell_metrics(
    labels: np.ndarray,
    volume: OrientationVolume,
    kam_field: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-cell size/inner-variance/boundary-KAM."""
    from scipy.ndimage import find_objects, binary_dilation, generate_binary_structure

    structure = generate_binary_structure(3, 1)

    n = int(labels.max())
    sizes = np.zeros(n, dtype=np.int64)
    inner_var = np.full(n, np.nan, dtype=np.float64)
    boundary_kam_mean = np.full(n, np.nan, dtype=np.float64)
    boundary_kam_median = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return sizes, inner_var, boundary_kam_mean, boundary_kam_median

    slices = find_objects(labels)
    chi = volume.field[..., 0]
    phi = volume.field[..., 1]

    for k, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        crop_label = labels[sl]
        cell_mask = crop_label == k
        sizes[k - 1] = int(cell_mask.sum())
        if sizes[k - 1] == 0:
            continue
        chi_vals = chi[sl][cell_mask]
        phi_vals = phi[sl][cell_mask]
        chi_vals = chi_vals[np.isfinite(chi_vals)]
        phi_vals = phi_vals[np.isfinite(phi_vals)]
        if chi_vals.size > 1 and phi_vals.size > 1:
            inner_var[k - 1] = float(chi_vals.var(ddof=1) + phi_vals.var(ddof=1))
        # boundary = (dilated mask) AND NOT mask, intersected with cell bounding-box
        dil = binary_dilation(cell_mask, structure=structure, iterations=1)
        boundary_mask = dil & ~cell_mask
        if boundary_mask.any():
            kam_b = kam_field[sl][boundary_mask]
            kam_b = kam_b[np.isfinite(kam_b)]
            if kam_b.size > 0:
                boundary_kam_mean[k - 1] = float(kam_b.mean())
                boundary_kam_median[k - 1] = float(np.median(kam_b))

    return sizes, inner_var, boundary_kam_mean, boundary_kam_median


def summarise_method(
    name: str,
    labels: np.ndarray,
    volume: OrientationVolume,
    kam_field: np.ndarray,
    *,
    min_size: int,
    drop_edge: bool,
    fit_lognormal: bool,
) -> Dict:
    if drop_edge:
        labels = _drop_edge_touching(labels)
    labels = _filter_by_size(labels, min_size).astype(np.int32)

    sizes, inner_var, b_mean, b_median = _per_cell_metrics(
        labels, volume, kam_field
    )

    in_mask_voxels = int(volume.mask.sum())
    voxel_volume_nm3 = volume.spacing.voxel_volume_nm3()
    sizes_nm3 = sizes.astype(np.float64) * voxel_volume_nm3
    sizes_um3 = sizes_nm3 * 1e-9
    eq_diam_um = (6.0 * sizes_um3 / np.pi) ** (1.0 / 3.0)

    largest = int(sizes.max()) if sizes.size > 0 else 0
    largest_fraction = float(largest) / max(in_mask_voxels, 1)

    summary: Dict = {
        "method": name,
        "n_cells": int(sizes.size),
        "in_mask_voxels": in_mask_voxels,
        "largest_component_voxels": largest,
        "largest_component_fraction": largest_fraction,
        "median_size_voxels": float(np.median(sizes)) if sizes.size else np.nan,
        "iqr_size_voxels": float(np.subtract(*np.percentile(sizes, [75, 25]))) if sizes.size else np.nan,
        "median_volume_nm3": float(np.median(sizes_nm3)) if sizes.size else np.nan,
        "median_volume_um3": float(np.median(sizes_um3)) if sizes.size else np.nan,
        "median_inner_variance": float(np.nanmedian(inner_var)) if sizes.size else np.nan,
        "median_boundary_kam_mean": float(np.nanmedian(b_mean)) if sizes.size else np.nan,
        "median_boundary_kam_median": float(np.nanmedian(b_median)) if sizes.size else np.nan,
        "voxel_volume_nm3": voxel_volume_nm3,
    }

    if fit_lognormal and sizes.size >= 5:
        try:
            from scipy.stats import lognorm
            shape, loc, scale = lognorm.fit(sizes_um3[sizes_um3 > 0], floc=0.0)
            summary["lognormal_sigma"] = float(shape)
            summary["lognormal_mu"] = float(np.log(scale))
            summary["lognormal_loc"] = float(loc)
            summary["lognormal_n_used"] = int((sizes_um3 > 0).sum())
        except Exception as exc:  # pragma: no cover
            summary["lognormal_error"] = str(exc)

    summary["_per_cell_sizes_voxels"] = sizes
    summary["_per_cell_volumes_um3"] = sizes_um3
    summary["_per_cell_eq_diam_um"] = eq_diam_um
    summary["_per_cell_inner_var"] = inner_var
    summary["_per_cell_boundary_kam_mean"] = b_mean
    summary["_labels"] = labels
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _set_paper_matplotlib_defaults() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def plot_size_distribution(out_dir: Path, methods: List[Dict]) -> None:
    import matplotlib.pyplot as plt
    _set_paper_matplotlib_defaults()
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for s in methods:
        vol = s["_per_cell_volumes_um3"]
        vol = vol[vol > 0]
        if vol.size == 0:
            continue
        ax.hist(
            vol, bins="auto", histtype="step", log=True,
            label=f"{s['method']} (n={s['n_cells']})",
        )
    ax.set_xlabel("cell volume [um^3]")
    ax.set_ylabel("count")
    ax.set_xscale("log")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "cell_size_distribution.pdf")
    fig.savefig(out_dir / "cell_size_distribution.png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_largest_component_fraction(out_dir: Path, methods: List[Dict]) -> None:
    import matplotlib.pyplot as plt
    _set_paper_matplotlib_defaults()
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    names = [s["method"] for s in methods]
    fracs = [s["largest_component_fraction"] for s in methods]
    ax.bar(names, fracs, color=["C0", "C1", "C2"][: len(methods)])
    ax.set_ylabel("largest connected component / in-mask voxels")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "largest_component_fraction.pdf")
    fig.savefig(out_dir / "largest_component_fraction.png")
    plt.close(fig)


def plot_inner_vs_boundary(out_dir: Path, methods: List[Dict]) -> None:
    import matplotlib.pyplot as plt
    _set_paper_matplotlib_defaults()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for s in methods:
        x = s["_per_cell_inner_var"]
        y = s["_per_cell_boundary_kam_mean"]
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(
            x[m], y[m], s=6, alpha=0.6,
            label=f"{s['method']} (n={int(m.sum())})",
        )
    ax.set_xlabel("intra-cell orientation variance E_inner")
    ax.set_ylabel("mean boundary-KAM E_boundary")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "boundary_vs_inner_metric.pdf")
    fig.savefig(out_dir / "boundary_vs_inner_metric.png")
    plt.close(fig)


def render_method_overlays(
    out_dir: Path,
    methods: List[Dict],
    volume: OrientationVolume,
    kam_field: np.ndarray,
    z_indices: List[int],
) -> None:
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    _set_paper_matplotlib_defaults()

    chi = volume.field[..., 0]
    finite_chi = chi[volume.mask & np.isfinite(chi)]
    if finite_chi.size == 0:
        return
    vmin, vmax = np.percentile(finite_chi, [1, 99])

    for z in z_indices:
        z = int(z)
        if not 0 <= z < volume.shape_zyx[0]:
            continue
        fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.4))
        if len(methods) == 1:
            axes = [axes]
        for ax, s in zip(axes, methods):
            arr = np.where(volume.mask[z], chi[z], np.nan)
            im = ax.imshow(
                arr, origin="lower", cmap="viridis",
                vmin=vmin, vmax=vmax, interpolation="nearest",
            )
            boundaries = find_boundaries(s["_labels"][z], mode="thick")
            ax.contour(boundaries.astype(int), levels=[0.5],
                       colors="black", linewidths=0.4)
            ax.set_title(f"{s['method']}  z={z}  n={s['n_cells']}")
            ax.set_xlabel("x [voxels]"); ax.set_ylabel("y [voxels]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=f"chi [{volume.angle_unit}]")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(
                out_dir / f"comparison_overlay_slice_{z:04d}.{ext}",
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
    floodfill_labels = load_h5_labels(Path(args.floodfill_labels_h5))
    kam_field = load_h5_scalar(Path(args.floodfill_kam_h5), dataset="kam")
    kam_baseline_labels = load_h5_labels(Path(args.kam_baseline_labels_h5))

    if floodfill_labels.shape != volume.shape_zyx:
        raise SystemExit(
            "flood-fill labels shape "
            f"{floodfill_labels.shape} differs from volume shape {volume.shape_zyx}; "
            "are the two scripts pointing at the same data?"
        )
    if kam_baseline_labels.shape != volume.shape_zyx:
        raise SystemExit(
            "KAM-baseline labels shape "
            f"{kam_baseline_labels.shape} differs from volume shape "
            f"{volume.shape_zyx}"
        )
    if kam_field.shape != volume.shape_zyx:
        raise SystemExit(
            f"KAM field shape {kam_field.shape} differs from volume shape "
            f"{volume.shape_zyx}"
        )

    summaries: List[Dict] = []
    for name, labels in [
        ("floodfill+watershed", floodfill_labels),
        ("kam_threshold", kam_baseline_labels),
    ]:
        s = summarise_method(
            name,
            labels=labels,
            volume=volume,
            kam_field=kam_field,
            min_size=int(args.min_cell_size),
            drop_edge=bool(args.exclude_edge_touching),
            fit_lognormal=bool(args.lognormal_fit),
        )
        summaries.append(s)

    # CSV summary table
    csv_path = out_dir / "method_comparison_stats.csv"
    keys = [k for k in summaries[0].keys() if not k.startswith("_")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, "") for k in keys})

    # plots
    plot_size_distribution(out_dir, summaries)
    plot_largest_component_fraction(out_dir, summaries)
    plot_inner_vs_boundary(out_dir, summaries)

    if args.slice_indices is None:
        Z = volume.shape_zyx[0]
        slice_idx = np.linspace(0, Z - 1, num=min(3, Z), dtype=int).tolist()
    else:
        slice_idx = list(args.slice_indices)
    render_method_overlays(out_dir, summaries, volume, kam_field, slice_idx)

    write_parameters_json(
        out_dir,
        {
            "script": "compare_3d_methods.py",
            "args": vars(args),
            "summaries_csv": str(csv_path.name),
            "voxel_spacing_nm_zyx": list(volume.spacing.as_tuple_nm()),
            "angle_unit": volume.angle_unit,
            "data_source": volume.source,
            "summaries": [
                {k: v for k, v in s.items() if not k.startswith("_")}
                for s in summaries
            ],
        },
    )

    print("[done] wrote method comparison to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

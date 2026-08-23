#!/usr/bin/env python
"""Stage 5: quantitative comparison of flood-fill+watershed vs KAM thresholding.

All methods are evaluated on the identical registered volume and KAM field.
Reported per method:

- number of cells; empirical physical cell-volume distribution (um^3) and
  equivalent sphere diameter; largest-cell fraction; assigned fraction;
- fraction of edge-truncated cells (touching a volume face);
- intra-cell angular spread sigma_k (deg, manuscript Eq. inner_cell_spread);
- finite-width boundary-band KAM E_bd (deg, manuscript Eq. boundary_kam);
- label-invariant agreement between the methods (VI, matched overlap).

These are internal-consistency measures; no ground truth exists. Note that
intra-cell coherence alone rewards over-segmentation and must be read
together with cell counts and boundary-band contrast.

Usage:
    python run_comparison.py --config config_6_2pct.json
"""

from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    load_config,
    load_registered_volume,
    out_dir_for,
    provenance,
    spacing_from_config,
    write_parameters_json,
)

import disell


def per_method_metrics(name, labels, field, mask, kam_map, voxel_um3, r_bd):
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    volumes_um3 = counts * voxel_um3
    eq_diam_um = (6.0 * volumes_um3 / np.pi) ** (1.0 / 3.0)

    # edge-truncated: touching any volume face
    face_ids = set()
    for sl in (np.s_[0, :, :], np.s_[-1, :, :], np.s_[:, 0, :],
               np.s_[:, -1, :], np.s_[:, :, 0], np.s_[:, :, -1]):
        face_ids |= set(np.unique(labels[sl]))
    face_ids.discard(0)
    truncated = np.isin(ids, sorted(face_ids))

    spread_sq = disell.inner_cell_spread(labels, field)
    sigma_deg = {k: float(np.sqrt(v)) for k, v in spread_sq.items()}
    e_bd = disell.boundary_band_kam(labels, kam_map, r_bd=r_bd, connectivity=1)

    n_valid = int(mask.sum())
    per_cell = [{
        "label": int(i),
        "voxels": int(c),
        "volume_um3": float(v),
        "equivalent_diameter_um": float(d),
        "edge_truncated": bool(t),
        "sigma_deg": sigma_deg.get(int(i), np.nan),
        "boundary_band_kam_deg": e_bd.get(int(i), np.nan),
    } for i, c, v, d, t in zip(ids, counts, volumes_um3, eq_diam_um, truncated)]

    summary = {
        "method": name,
        "n_cells": int(ids.size),
        "assigned_fraction": float(counts.sum() / n_valid),
        "largest_cell_fraction": float(counts.max() / n_valid) if ids.size else np.nan,
        "median_volume_um3": float(np.median(volumes_um3)),
        "mean_volume_um3": float(np.mean(volumes_um3)),
        "median_equivalent_diameter_um": float(np.median(eq_diam_um)),
        "edge_truncated_fraction": float(truncated.mean()) if ids.size else np.nan,
        "median_sigma_deg": float(np.nanmedian(list(sigma_deg.values()))),
        "median_boundary_band_kam_deg": float(np.nanmedian(list(e_bd.values()))),
    }
    return per_cell, summary, volumes_um3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--r-bd", type=int, default=1,
                    help="boundary-band dilation radius (elements)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = out_dir_for(cfg, "comparison", args.out_root)
    vol_dir = out_dir_for(cfg, "volume", args.out_root)
    seg3d_dir = out_dir_for(cfg, "segmentation_3d", args.out_root)
    kam_dir = out_dir_for(cfg, "kam_baseline", args.out_root)

    field, mask, _, _ = load_registered_volume(vol_dir / "volume_registered.h5")

    import h5py
    with h5py.File(seg3d_dir / "labels_3d.h5", "r") as f:
        labels_ff = f["labels"][...]
    with h5py.File(seg3d_dir / "kam_3d.h5", "r") as f:
        kam_map = f["kam"][...]
    with h5py.File(kam_dir / "kam_threshold_labels_raw.h5", "r") as f:
        labels_kam_raw = f["labels"][...]
        thr_sel = float(f["labels"].attrs["threshold_deg"])
    with h5py.File(kam_dir / "kam_threshold_labels_expanded.h5", "r") as f:
        labels_kam_exp = f["labels"][...]

    spacing = spacing_from_config(cfg)
    voxel_um3 = spacing.voxel_volume_nm3() / 1e9

    methods = {
        "floodfill_watershed": labels_ff,
        "kam_threshold_raw": labels_kam_raw,
        "kam_threshold_expanded": labels_kam_exp,
    }

    summaries = []
    volume_dists = {}
    for name, labels in methods.items():
        per_cell, summary, volumes = per_method_metrics(
            name, labels, field, mask, kam_map, voxel_um3, args.r_bd
        )
        summaries.append(summary)
        volume_dists[name] = volumes
        with open(out / f"per_cell_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_cell[0].keys())
            w.writeheader()
            w.writerows(per_cell)
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in summary.items()})

    agreement = {
        "vi_ff_vs_kam_expanded_bits": disell.variation_of_information(
            labels_ff, labels_kam_exp, mask=mask),
        "overlap_ff_vs_kam_expanded": disell.matched_overlap(
            labels_ff, labels_kam_exp, mask=mask),
        "vi_ff_vs_kam_raw_bits": disell.variation_of_information(
            labels_ff, labels_kam_raw, mask=mask),
        "overlap_ff_vs_kam_raw": disell.matched_overlap(
            labels_ff, labels_kam_raw, mask=mask),
    }
    print("agreement:", {k: round(v, 4) for k, v in agreement.items()})

    with open(out / "method_summaries.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader()
        w.writerows(summaries)
    np.savez(out / "volume_distributions_um3.npz", **volume_dists)

    # --- empirical volume distributions ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.logspace(
        np.log10(max(min(v.min() for v in volume_dists.values()), 1e-2)),
        np.log10(max(v.max() for v in volume_dists.values())), 40,
    )
    for name, v in volume_dists.items():
        axes[0].hist(v, bins=bins, histtype="step", label=name, density=True)
        vs = np.sort(v)
        axes[1].plot(vs, 1.0 - np.arange(vs.size) / vs.size, label=name)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"cell volume ($\mu m^3$)")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"cell volume ($\mu m^3$)")
    axes[1].set_ylabel("survival fraction")
    fig.suptitle("Empirical cell-volume distributions, 6.2% volume")
    fig.tight_layout()
    fig.savefig(out / "fig_volume_distributions.png", dpi=250)
    fig.savefig(out / "fig_volume_distributions.pdf")
    plt.close(fig)

    # --- per-cell metric distributions --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in methods:
        rows = list(csv.DictReader(open(out / f"per_cell_{name}.csv")))
        sig = np.array([float(r["sigma_deg"]) for r in rows])
        ebd = np.array([float(r["boundary_band_kam_deg"]) for r in rows])
        axes[0].hist(sig[np.isfinite(sig)], bins=40, histtype="step",
                     label=name, density=True)
        axes[1].hist(ebd[np.isfinite(ebd)], bins=40, histtype="step",
                     label=name, density=True)
    axes[0].set_xlabel(r"intra-cell spread $\sigma_k$ (deg)")
    axes[1].set_xlabel(r"boundary-band KAM $E_{bd}$ (deg)")
    for ax in axes:
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_cell_metric_distributions.png", dpi=250)
    fig.savefig(out / "fig_cell_metric_distributions.pdf")
    plt.close(fig)

    # --- slice overlay -------------------------------------------------------
    from skimage.segmentation import find_boundaries

    z = labels_ff.shape[0] // 2
    dy_um, dx_um = spacing.dy_nm / 1e3, spacing.dx_nm / 1e3
    ny, nx = labels_ff.shape[1:]
    extent = [0, nx * dx_um, ny * dy_um, 0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, (name, labels) in zip(
        axes, [("flood fill + watershed", labels_ff),
               (f"KAM threshold ({thr_sel:.3f} deg), expanded", labels_kam_exp)]
    ):
        ax.imshow(field[z, ..., 0], cmap="viridis", extent=extent)
        b = find_boundaries(labels[z], mode="outer")
        yy, xx = np.where(b)
        ax.scatter(xx * dx_um, yy * dy_um, s=0.05, c="k")
        ax.set_title(f"{name}, layer {z}")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
    fig.tight_layout()
    fig.savefig(out / "fig_method_overlay.png", dpi=250)
    fig.savefig(out / "fig_method_overlay.pdf")
    plt.close(fig)

    write_parameters_json(out, {
        "stage": "comparison",
        "config": cfg,
        "r_bd": args.r_bd,
        "kam_selected_threshold_deg": thr_sel,
        "summaries": summaries,
        "agreement": agreement,
        "provenance": provenance(),
    })
    with open(out / "agreement.json", "w") as f:
        json.dump(agreement, f, indent=2)
    print(f"outputs written to {out}")


if __name__ == "__main__":
    main()

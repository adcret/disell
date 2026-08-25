#!/usr/bin/env python
"""Stage 3: slice-wise 2D segmentation and 3D-vs-2D coherence comparison.

Runs identification + watershed independently on every layer with physically
comparable parameters (in-plane part of the 3D footprints, same thresholds in
degrees), then quantifies slice-to-slice coherence of (a) the slice-wise
labels and (b) the direct-3D labels restricted to slices, using
label-invariant metrics defined *before* computation:

- variation of information VI(A, B) in bits (0 = identical partitions),
  computed on voxels labelled in both slices;
- matched overlap: size-weighted symmetric best-IoU (1 = identical).

Also produces the manuscript-style figure of three neighbouring layers under
both approaches.

Usage:
    python run_slicewise.py --config config_6_2pct.json
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    flood_footprint,
    kam_footprint,
    load_config,
    load_registered_volume,
    masked_kam,
    out_dir_for,
    provenance,
    save_h5_volume,
    spacing_from_config,
    watershed_feature_from_kam,
    write_parameters_json,
)

import disell


def segment_slice(field_2d, mask_2d, cfg):
    """2D identification + watershed with the in-plane parameters."""
    p = cfg["segmentation"]
    fp3 = flood_footprint(cfg)
    fp2 = fp3[fp3.shape[0] // 2]          # in-plane part of the 3D footprint

    finite_field = np.nan_to_num(field_2d, nan=0.0).astype(np.float32)
    result, _ = disell.flood_fill_dfxm_two_stage(
        finite_field,
        footprint=fp2,
        local_misorientation_threshold=p["local_threshold_deg"],
        global_threshold=(None if p["global_threshold_deg"] is None
                          or p["global_threshold_deg"] <= 0
                          else p["global_threshold_deg"]),
        footprint_tolerance=p["footprint_tolerance"],
        mask=mask_2d.astype(np.uint8),
        max_iterations=p["max_iterations"],
        min_grain_size=p["min_grain_size"],
        stagnation_tolerance=p["stagnation_tolerance"],
        random_seed=p["random_seed"],
    )
    markers = np.asarray(result["segmentation"], dtype=np.int32)

    kfp3 = kam_footprint(cfg, ndim=3)
    kfp2 = kfp3[kfp3.shape[0] // 2]
    kam_map = masked_kam(field_2d, mask_2d, kfp2)
    feature = watershed_feature_from_kam(kam_map, mask_2d)
    labels = disell.region_grow_watershed(
        markers, mask_2d, feature, connectivity=p["watershed_connectivity"]
    )
    return markers, np.asarray(labels, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = out_dir_for(cfg, "slicewise", args.out_root)
    vol_dir = out_dir_for(cfg, "volume", args.out_root)
    seg3d_dir = out_dir_for(cfg, "segmentation_3d", args.out_root)

    field, mask, _, _ = load_registered_volume(vol_dir / "volume_registered.h5")
    import h5py
    with h5py.File(seg3d_dir / "labels_3d.h5", "r") as f:
        labels3d = f["labels"][...]

    Z = field.shape[0]
    labels2d = np.zeros_like(labels3d)
    markers2d = np.zeros_like(labels3d)
    for z in range(Z):
        m, l = segment_slice(field[z], mask[z], cfg)
        markers2d[z], labels2d[z] = m, l
        print(f"layer {z}: {l.max()} 2D cells")

    spacing = spacing_from_config(cfg)
    save_h5_volume(out / "labels_slicewise.h5", labels2d, dataset="labels",
                   spacing=spacing)
    save_h5_volume(out / "markers_slicewise.h5", markers2d, dataset="markers",
                   spacing=spacing)

    # --- coherence metrics --------------------------------------------------
    rows = []
    for z in range(Z - 1):
        pair_mask = mask[z] & mask[z + 1]
        rows.append({
            "z_pair": f"{z}-{z+1}",
            "vi_slicewise": disell.variation_of_information(
                labels2d[z], labels2d[z + 1], mask=pair_mask),
            "vi_3d": disell.variation_of_information(
                labels3d[z], labels3d[z + 1], mask=pair_mask),
            "overlap_slicewise": disell.matched_overlap(
                labels2d[z], labels2d[z + 1], mask=pair_mask),
            "overlap_3d": disell.matched_overlap(
                labels3d[z], labels3d[z + 1], mask=pair_mask),
        })
    same_slice = []
    for z in range(Z):
        same_slice.append({
            "z": z,
            "vi_2d_vs_3d": disell.variation_of_information(
                labels2d[z], labels3d[z], mask=mask[z]),
            "overlap_2d_vs_3d": disell.matched_overlap(
                labels2d[z], labels3d[z], mask=mask[z]),
            "n_cells_2d": int(len(np.unique(labels2d[z][labels2d[z] > 0]))),
            "n_cells_3d_in_slice": int(len(np.unique(labels3d[z][labels3d[z] > 0]))),
        })

    with open(out / "coherence_adjacent_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(out / "coherence_same_slice.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=same_slice[0].keys())
        w.writeheader()
        w.writerows(same_slice)

    def med(key, table):
        return float(np.nanmedian([r[key] for r in table]))

    summary = {
        "median_vi_adjacent_slicewise": med("vi_slicewise", rows),
        "median_vi_adjacent_3d": med("vi_3d", rows),
        "median_overlap_adjacent_slicewise": med("overlap_slicewise", rows),
        "median_overlap_adjacent_3d": med("overlap_3d", rows),
        "median_vi_2d_vs_3d_same_slice": med("vi_2d_vs_3d", same_slice),
    }
    print("coherence summary:", summary)

    # --- figure: three neighbouring layers, slice-wise vs 3D ---------------
    zc = Z // 2
    zs = [zc - 1, zc, zc + 1]
    rng = np.random.default_rng(0)

    def label_rgb(lab, lut=None):
        if lut is None:
            lut = rng.uniform(0.15, 1.0, size=(int(lab.max()) + 1, 3))
        img = lut[lab]
        img[lab == 0] = 0
        return img, lut

    # one fixed LUT for the whole 3D label volume so cell colours persist
    # across the displayed layers; slice-wise LUTs are per-slice by design
    lut3d = rng.uniform(0.15, 1.0, size=(int(labels3d.max()) + 1, 3))
    dy_um, dx_um = spacing.dy_nm / 1e3, spacing.dx_nm / 1e3
    ny, nx = labels3d.shape[1:]
    extent = [0, nx * dx_um, ny * dy_um, 0]
    # zoom window (pixels) making individual cells legible in print
    zoom_y, zoom_x = (40, 140), (150, 300)
    for tag, view in (("", None), ("_zoom", (zoom_x, zoom_y))):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
        for j, z in enumerate(zs):
            img2d, _ = label_rgb(labels2d[z])
            img3d, lut3d = label_rgb(labels3d[z], lut3d)
            axes[0, j].imshow(img2d, extent=extent, interpolation="nearest")
            axes[0, j].set_title(f"slice-wise 2D, layer {z}")
            axes[1, j].imshow(img3d, extent=extent, interpolation="nearest")
            axes[1, j].set_title(f"direct 3D, layer {z}")
        for ax in axes.flat:
            if view is not None:
                (x0, x1), (y0, y1) = view
                ax.set_xlim(x0 * dx_um, x1 * dx_um)
                ax.set_ylim(y1 * dy_um, y0 * dy_um)
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
        fig.tight_layout()
        fig.savefig(out / f"fig_slicewise_vs_3d{tag}.png", dpi=250)
        fig.savefig(out / f"fig_slicewise_vs_3d{tag}.pdf")
        plt.close(fig)

    # coherence line plot
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(Z - 1) + 0.5
    ax.plot(x, [r["vi_slicewise"] for r in rows], "o-", label="slice-wise 2D")
    ax.plot(x, [r["vi_3d"] for r in rows], "s-", label="direct 3D")
    ax.set_xlabel("layer boundary")
    ax.set_ylabel("variation of information (bits)")
    ax.set_title("slice-to-slice coherence (lower = more coherent)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_coherence_vi.png", dpi=250)
    fig.savefig(out / "fig_coherence_vi.pdf")
    plt.close(fig)

    write_parameters_json(out, {
        "stage": "slicewise",
        "config": cfg,
        "metric_definitions": {
            "vi": "variation of information, bits, voxels labelled in both",
            "matched_overlap": "size-weighted symmetric best-IoU",
        },
        "summary": summary,
        "figure_layers": zs,
        "provenance": provenance(),
    })
    print(f"outputs written to {out}")


if __name__ == "__main__":
    main()

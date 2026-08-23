#!/usr/bin/env python
"""Stage 2: direct 3D multi-seed flood-fill identification + watershed refinement.

Runs the complete manuscript method on the registered 6.2% volume:
local threshold, running-mean global threshold, footprint tolerance, minimum
size with parking/absorption, stagnation termination, deterministic seeds;
then KAM-guided marker-based watershed, followed by connected-component
validation. Markers and final labels are stored separately.

Usage:
    python run_segment_3d.py --config config_6_2pct.json
"""

from __future__ import annotations

import argparse
import json

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
    save_vti,
    spacing_from_config,
    watershed_feature_from_kam,
    write_parameters_json,
)

import disell


def segment_volume(field, mask, cfg, *, random_seed=None, params=None):
    """Identification + refinement with the config (or overridden) parameters.

    Returns dict with markers, labels, kam, and bookkeeping. ``field`` must be
    the registered/preprocessed (Z, Y, X, 2) array in degrees with NaN outside
    ``mask``.
    """
    p = dict(cfg["segmentation"])
    if params:
        p.update(params)
    seed = p["random_seed"] if random_seed is None else random_seed

    fp = flood_footprint({"segmentation": p, **{k: cfg[k] for k in ("spacing_nm_zyx",)}}) \
        if isinstance(p.get("flood_footprint"), str) else p["flood_footprint"]

    finite_field = np.nan_to_num(field, nan=0.0).astype(np.float32)

    result, sizes_initial = disell.flood_fill_dfxm_two_stage(
        finite_field,
        footprint=fp,
        local_misorientation_threshold=p["local_threshold_deg"],
        global_threshold=(None if p["global_threshold_deg"] is None
                          or p["global_threshold_deg"] <= 0
                          else p["global_threshold_deg"]),
        footprint_tolerance=p["footprint_tolerance"],
        mask=mask.astype(np.uint8),
        max_iterations=p["max_iterations"],
        min_grain_size=p["min_grain_size"],
        stagnation_tolerance=p["stagnation_tolerance"],
        random_seed=seed,
    )
    markers = np.asarray(result["segmentation"], dtype=np.int32)

    kfp = kam_footprint({**cfg, "segmentation": p})
    kam_map = masked_kam(field, mask, kfp)
    feature = watershed_feature_from_kam(kam_map, mask)

    labels = disell.region_grow_watershed(
        markers, mask, feature, connectivity=p["watershed_connectivity"]
    )
    labels = np.asarray(labels, dtype=np.int32)

    report = disell.connected_component_report(
        labels, connectivity=p["watershed_connectivity"]
    )
    split_info = None
    if report["n_disconnected"] > 0:
        labels_split, split_info = disell.split_disconnected_labels(
            labels, connectivity=p["watershed_connectivity"],
            min_size=p["split_min_component_size"],
        )
        labels = labels_split

    return {
        "markers": markers,
        "labels": labels,
        "kam": kam_map,
        "sizes_initial": np.asarray(sizes_initial),
        "marker_sizes": np.asarray(result["sizes"]) if result["sizes"] is not None else np.array([]),
        "cc_report": report,
        "split_info": split_info,
        "params": p,
        "random_seed": seed,
        "flood_footprint_array": np.asarray(fp),
        "kam_footprint_shape": list(kfp.shape),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--volume-dir", default=None,
                    help="directory holding volume_registered.h5 "
                         "(default: <out-root>/volume)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = out_dir_for(cfg, "segmentation_3d", args.out_root)
    vol_dir = args.volume_dir or out_dir_for(cfg, "volume", args.out_root)

    field, mask, transforms, attrs = load_registered_volume(
        vol_dir / "volume_registered.h5"
    )
    spacing = spacing_from_config(cfg)

    res = segment_volume(field, mask, cfg)
    markers, labels, kam_map = res["markers"], res["labels"], res["kam"]

    n_markers = int(markers.max())
    n_cells = len(np.unique(labels)) - (1 if (labels == 0).any() else 0)
    unassigned_before = float(((markers == 0) & mask).sum() / mask.sum())
    print(f"markers: {n_markers} regions, unassigned before watershed "
          f"{unassigned_before:.3f}; final cells: {n_cells}")
    print(f"connectivity check: {res['cc_report']['n_disconnected']} disconnected labels, "
          f"fragment voxel fraction {res['cc_report']['fragment_voxel_fraction']:.5f}")

    save_h5_volume(out / "markers_3d.h5", markers, dataset="markers",
                   spacing=spacing)
    save_h5_volume(out / "labels_3d.h5", labels, dataset="labels",
                   spacing=spacing)
    save_h5_volume(out / "kam_3d.h5", kam_map.astype(np.float32),
                   dataset="kam", spacing=spacing,
                   angle_unit=cfg["angle_unit"],
                   extra_attrs={"definition": "per-channel RMS neighbour distance"})
    save_vti(out / "segmentation_3d.vti", {
        "labels": labels.astype(np.int32),
        "markers": markers.astype(np.int32),
        "kam": np.nan_to_num(kam_map, nan=-1.0).astype(np.float32),
        "mask": mask.astype(np.uint8),
    }, spacing=spacing)

    # --- quicklook overlays ------------------------------------------------
    from skimage.segmentation import find_boundaries

    dy_um, dx_um = spacing.dy_nm / 1e3, spacing.dx_nm / 1e3
    ny, nx = labels.shape[1:]
    extent = [0, nx * dx_um, ny * dy_um, 0]
    for z in (2, labels.shape[0] // 2, labels.shape[0] - 3):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        axes[0].imshow(field[z, ..., 0], cmap="viridis", extent=extent)
        axes[0].set_title(f"chi (deg), z={z}")
        axes[1].imshow(np.nan_to_num(kam_map[z], nan=0), cmap="magma", extent=extent)
        axes[1].set_title("KAM (deg)")
        b = find_boundaries(labels[z], mode="outer")
        img = field[z, ..., 0].copy()
        axes[2].imshow(img, cmap="viridis", extent=extent)
        yy, xx = np.where(b)
        axes[2].scatter(xx * dx_um, yy * dy_um, s=0.05, c="k")
        axes[2].set_title(f"final labels ({n_cells} cells)")
        for ax in axes:
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
        fig.tight_layout()
        fig.savefig(out / f"overlay_z{z}.png", dpi=200)
        plt.close(fig)

    write_parameters_json(out, {
        "stage": "segment_3d",
        "config": cfg,
        "params_used": res["params"],
        "random_seed": res["random_seed"],
        "flood_footprint": res["flood_footprint_array"].astype(int).tolist(),
        "kam_footprint_shape": res["kam_footprint_shape"],
        "n_markers": n_markers,
        "n_cells": n_cells,
        "unassigned_fraction_before_watershed": unassigned_before,
        "cc_report": {k: v for k, v in res["cc_report"].items() if k != "fragments"},
        "n_fragmented_labels": res["cc_report"]["n_disconnected"],
        "split_applied": res["split_info"] is not None,
        "provenance": provenance(),
    })

    with open(out / "cc_report.json", "w") as f:
        json.dump({"cc_report": res["cc_report"],
                   "split_info": res["split_info"]}, f, indent=2, default=str)
    print(f"outputs written to {out}")


if __name__ == "__main__":
    main()

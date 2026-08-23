#!/usr/bin/env python
"""Stage 6: parameter and random-seed sensitivity on a representative ROI.

Varies one parameter at a time around the configured operating point on a
fixed spatial ROI (all layers, an interior in-plane window), tracking cell
count, median cell volume, largest-cell fraction, unassigned fraction before
watershed, intra-cell spread, boundary-band KAM and (for repeated seeds)
pairwise label-invariant agreement.

The internal metrics are reported for transparency, not used to optimise the
operating point against itself.

Usage:
    python run_sensitivity.py --config config_6_2pct.json
"""

from __future__ import annotations

import argparse
import csv
import itertools

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
from run_segment_3d import segment_volume

import disell

ROI = (slice(None), slice(40, 160), slice(100, 400))

VARIATIONS = {
    "local_threshold_deg": [0.04, 0.05, 0.06, 0.08, 0.10],
    "global_threshold_deg": [-1.0, 0.20, 0.30, 0.40],
    "footprint_tolerance": [0.70, 0.85, 1.00],
    "min_grain_size": [10, 20, 50],
    "flood_footprint": ["faces6", "inplane8_plus_z", "full26"],
    "kam_radius_nm": [900.0, 1300.0, 1800.0],
}
SEEDS = [42, 1, 2, 3, 4]


def run_metrics(field, mask, cfg, voxel_um3, *, params=None, seed=None):
    res = segment_volume(field, mask, cfg, random_seed=seed, params=params)
    labels, markers, kam_map = res["labels"], res["markers"], res["kam"]
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    n_valid = int(mask.sum())
    spread = disell.inner_cell_spread(labels, field)
    e_bd = disell.boundary_band_kam(labels, kam_map, r_bd=1, connectivity=1)
    return {
        "n_markers": int(markers.max()),
        "unassigned_fraction": float(((markers == 0) & mask).sum() / n_valid),
        "n_cells": int(ids.size),
        "median_volume_um3": float(np.median(counts * voxel_um3)) if ids.size else np.nan,
        "largest_cell_fraction": float(counts.max() / n_valid) if ids.size else np.nan,
        "median_sigma_deg": float(np.nanmedian(np.sqrt(list(spread.values())))) if spread else np.nan,
        "median_boundary_kam_deg": float(np.nanmedian(list(e_bd.values()))) if e_bd else np.nan,
    }, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = out_dir_for(cfg, "sensitivity", args.out_root)
    vol_dir = out_dir_for(cfg, "volume", args.out_root)

    field_full, mask_full, _, _ = load_registered_volume(
        vol_dir / "volume_registered.h5"
    )
    field = np.ascontiguousarray(field_full[ROI])
    mask = np.ascontiguousarray(mask_full[ROI])
    spacing = spacing_from_config(cfg)
    voxel_um3 = spacing.voxel_volume_nm3() / 1e9
    print(f"ROI shape {field.shape}, valid fraction {mask.mean():.3f}")

    rows = []
    base = {k: cfg["segmentation"][k] for k in VARIATIONS}

    for pname, values in VARIATIONS.items():
        for v in values:
            m, _ = run_metrics(field, mask, cfg, voxel_um3, params={pname: v})
            m.update({"parameter": pname, "value": v,
                      "is_base": v == base[pname]})
            rows.append(m)
            print(f"{pname}={v}: cells={m['n_cells']} "
                  f"medvol={m['median_volume_um3']:.2f} "
                  f"unassigned={m['unassigned_fraction']:.3f}")

    # --- seed repeats at the base operating point ---------------------------
    seed_labels = {}
    for s in SEEDS:
        m, labels = run_metrics(field, mask, cfg, voxel_um3, seed=s)
        m.update({"parameter": "random_seed", "value": s,
                  "is_base": s == cfg["segmentation"]["random_seed"]})
        rows.append(m)
        seed_labels[s] = labels
        print(f"seed={s}: cells={m['n_cells']}")

    seed_pairs = []
    for a, b in itertools.combinations(SEEDS, 2):
        seed_pairs.append({
            "seed_a": a, "seed_b": b,
            "vi_bits": disell.variation_of_information(
                seed_labels[a], seed_labels[b], mask=mask),
            "matched_overlap": disell.matched_overlap(
                seed_labels[a], seed_labels[b], mask=mask),
        })
    with open(out / "seed_agreement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seed_pairs[0].keys())
        w.writeheader()
        w.writerows(seed_pairs)

    with open(out / "sensitivity_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # --- figure --------------------------------------------------------------
    metrics_to_plot = ["n_cells", "median_volume_um3",
                       "unassigned_fraction", "median_boundary_kam_deg"]
    params = list(VARIATIONS) + ["random_seed"]
    fig, axes = plt.subplots(
        len(metrics_to_plot), len(params),
        figsize=(3.0 * len(params), 2.4 * len(metrics_to_plot)),
        squeeze=False,
    )
    for j, pname in enumerate(params):
        sub = [r for r in rows if r["parameter"] == pname]
        x = np.arange(len(sub))
        for i, mname in enumerate(metrics_to_plot):
            ax = axes[i, j]
            ax.plot(x, [r[mname] for r in sub], "o-")
            for k, r in enumerate(sub):
                if r["is_base"]:
                    ax.axvline(k, color="0.8", lw=4, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([str(r["value"]) for r in sub],
                               rotation=45, fontsize=7)
            if i == 0:
                ax.set_title(pname, fontsize=9)
            if j == 0:
                ax.set_ylabel(mname, fontsize=8)
    fig.suptitle("Parameter and seed sensitivity (ROI), 6.2% volume")
    fig.tight_layout()
    fig.savefig(out / "fig_sensitivity.png", dpi=220)
    fig.savefig(out / "fig_sensitivity.pdf")
    plt.close(fig)

    med_vi = float(np.median([p["vi_bits"] for p in seed_pairs]))
    med_ov = float(np.median([p["matched_overlap"] for p in seed_pairs]))
    print(f"seed agreement: median VI {med_vi:.3f} bits, "
          f"median matched overlap {med_ov:.3f}")

    write_parameters_json(out, {
        "stage": "sensitivity",
        "config": cfg,
        "roi_zyx": [[s.start, s.stop] if s.start is not None else None
                    for s in ROI],
        "variations": VARIATIONS,
        "seeds": SEEDS,
        "seed_agreement_median_vi_bits": med_vi,
        "seed_agreement_median_overlap": med_ov,
        "provenance": provenance(),
    })
    print(f"outputs written to {out}")


if __name__ == "__main__":
    main()

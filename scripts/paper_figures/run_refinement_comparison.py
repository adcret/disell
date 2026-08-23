#!/usr/bin/env python
"""Stage 8 (optional): refinement-method comparison on the sensitivity ROI.

Grows the same flood-fill markers to the full mask with (a) KAM-guided
marker watershed and (b) mean-feature region growing
(``region_grow_minimum_cell_orientation_differences``), and reports the
manuscript's selection metrics for both: inner-cell spread sigma_k and
boundary-band KAM E_bd, plus mutual agreement. Graph cut is not implemented
in the package and is out of scope.

Usage:
    python run_refinement_comparison.py --config config_6_2pct.json
"""

from __future__ import annotations

import csv

import argparse

import numpy as np

from common import (
    load_config,
    load_registered_volume,
    out_dir_for,
    provenance,
    write_parameters_json,
)
from run_segment_3d import segment_volume
from run_sensitivity import ROI

import disell


def refinement_metrics(labels, field, kam_map, mask):
    ids = np.unique(labels[labels > 0])
    spread = disell.inner_cell_spread(labels, field)
    e_bd = disell.boundary_band_kam(labels, kam_map, r_bd=1, connectivity=1)
    return {
        "n_cells": int(ids.size),
        "assigned_fraction": float((labels > 0).sum() / mask.sum()),
        "median_sigma_deg": float(np.nanmedian(np.sqrt(list(spread.values())))),
        "median_boundary_kam_deg": float(np.nanmedian(list(e_bd.values()))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = out_dir_for(cfg, "refinement_comparison", args.out_root)
    vol_dir = out_dir_for(cfg, "volume", args.out_root)

    field_full, mask_full, _, _ = load_registered_volume(
        vol_dir / "volume_registered.h5"
    )
    field = np.ascontiguousarray(field_full[ROI])
    mask = np.ascontiguousarray(mask_full[ROI])

    res = segment_volume(field, mask, cfg)
    markers, kam_map = res["markers"], res["kam"]
    labels_ws = res["labels"]

    # mean-feature region growing from the identical markers
    labels_rg = disell.region_grow_minimum_cell_orientation_differences(
        markers.copy(), field.astype(np.float64), mask
    )
    labels_rg = np.asarray(labels_rg, dtype=np.int32)

    rows = []
    for name, labels in (("watershed", labels_ws),
                         ("mean_feature_region_growing", labels_rg)):
        m = refinement_metrics(labels, field, kam_map, mask)
        m["method"] = name
        rows.append(m)
        print(name, m)

    agreement = {
        "vi_bits": disell.variation_of_information(labels_ws, labels_rg, mask=mask),
        "matched_overlap": disell.matched_overlap(labels_ws, labels_rg, mask=mask),
    }
    print("agreement watershed vs region growing:", agreement)

    with open(out / "refinement_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    write_parameters_json(out, {
        "stage": "refinement_comparison",
        "config": cfg,
        "roi_zyx": [[s.start, s.stop] if s.start is not None else None
                    for s in ROI],
        "results": rows,
        "agreement": agreement,
        "provenance": provenance(),
    })
    print(f"outputs written to {out}")


if __name__ == "__main__":
    main()

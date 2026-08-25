"""Finalise figures and summaries from the completed KAM sweep."""

import csv
import json
from pathlib import Path

import numpy as np

import kam3d_6p2_extended_analysis as analysis


def numeric_row(row):
    result = {}
    for key, value in row.items():
        if key == "dataset_variant":
            result[key] = value
        elif key in {"retained_markers", "completed_regions"}:
            result[key] = int(value)
        else:
            result[key] = float(value)
    return result


output = analysis.OUTPUT
config = json.load(open(
    analysis.PACKAGE_ROOT / "scripts" / "paper_figures" / "config_6_2pct.json"
))
volume = analysis.disell.load_layer_volume(
    Path("~/Documents/Data/4dcells/111_cells_6-2pct_mosa_2x_raw").expanduser(),
    spacing_nm=config["spacing_nm_zyx"],
    angle_unit=config["angle_unit"],
    channel_names=config["channel_names"],
    crop=config["crop_rows_cols"],
    pattern=config["layer_pattern"],
    expect_n_layers=config["expect_n_layers"],
    expect_shape_yx=tuple(config["expect_shape_yx"]),
)
field = volume.field.astype(np.float32)
mask = volume.mask.copy()
field[~mask] = np.nan
spacing_um = (0.5, 1.24, 0.4)

registered_field, registered_mask, transforms = (
    analysis.registered_sensitivity_field(field, mask)
)
footprint = analysis.isotropic_physical_footprint(
    analysis.VoxelSpacing(500, 1240, 400), radius_nm=500, ndim=3
)
registered_kam = analysis.kam_lowmem(
    registered_field, registered_mask, footprint
)
_, registered_labels = analysis.run_flood_fill(
    registered_field, registered_mask, 10, registered_kam
)
registered_flood_summary = analysis.notebook_spread_summary(
    registered_field, registered_labels, registered_mask,
    float(np.prod(spacing_um)),
)

registration = json.load(open(
    output / "registration_sensitivity_summary.json"
))
registration["registered_flood_fill_domains"] = int(
    np.unique(registered_labels[registered_labels > 0]).size
)
registration["registered_flood_fill_summary"] = registered_flood_summary
with open(output / "registration_sensitivity_summary.json", "w") as handle:
    json.dump(registration, handle, indent=2)

rows = [
    numeric_row(row)
    for row in csv.DictReader(open(output / "kam_radius_threshold_sweep.csv"))
]
best_rows = [
    numeric_row(row)
    for row in csv.DictReader(open(output / "best_kam_threshold_by_radius.csv"))
]
native_best = max(
    (row for row in best_rows if row["dataset_variant"] == "native"),
    key=lambda row: (
        row["matched_overlap"], -row["vi_bits"],
        -row["largest_marker_fraction"],
    ),
)
registered_best = max(
    (row for row in best_rows
     if row["dataset_variant"] == "registered_sensitivity"),
    key=lambda row: (
        row["matched_overlap"], -row["vi_bits"],
        -row["largest_marker_fraction"],
    ),
)
baseline = list(csv.DictReader(open(
    output / "baseline_reproduction_6p2.csv"
)))
baseline_15 = next(
    row for row in baseline if float(row["percentile"]) == 15
)
baseline_best = max(
    baseline, key=lambda row: float(row["matched_overlap"])
)
roi = json.load(open(output / "roi_candidates_top5.json"))[0]
arrays = np.load(output / "selected_segmentations_and_roi.npz")

low = np.array([np.nanmin(field[..., channel]) for channel in range(2)])
high = np.array([np.nanmax(field[..., channel]) for channel in range(2)])
normalisation = np.stack(
    [low - 0.001 * (high - low), high + 0.001 * (high - low)], axis=1
)
rgb = np.stack([
    analysis.darling.transforms.rgb(field[z], norm=normalisation)[0]
    for z in range(field.shape[0])
])
rgb[~mask] = 1
native_rows = [
    row for row in rows if row["dataset_variant"] == "native"
]
analysis.make_proposed_figure(
    native_rows, native_best, rgb, roi,
    arrays["labels_flood"], arrays["labels_kam"], spacing_um,
    output / "proposed_main_figure",
)

final = {
    "baseline_reproduced": True,
    "notebook_baseline_displayed_15pct": baseline_15,
    "notebook_radius_05_best_coarse_threshold": baseline_best,
    "native_global_best": native_best,
    "registered_global_best": registered_best,
    "registered_flood_fill_summary": registered_flood_summary,
    "selected_roi": roi,
    "registration": registration,
    "sweep_rows": len(rows),
}
with open(output / "analysis_summary.json", "w") as handle:
    json.dump(final, handle, indent=2)
print(json.dumps(final, indent=2))

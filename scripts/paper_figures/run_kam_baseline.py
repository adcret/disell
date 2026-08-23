#!/usr/bin/env python
"""Fine 3D KAM-threshold comparison and manuscript Figure 4.

This is the implementation source used to update the existing
scripts/paper_figures/run_kam_baseline.py workflow. It consumes the registered
volume and deterministic flood-fill result produced by run_preprocess.py and
run_segment_3d.py; it does not duplicate loading or preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.65,
    "lines.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage
from skimage.segmentation import find_boundaries

HERE = Path(__file__).resolve().parent
PAPER_SCRIPTS = HERE.parent / "scripts" / "paper_figures"
if not PAPER_SCRIPTS.exists():
    PAPER_SCRIPTS = Path(
        "/home/adam/Documents/Scripts/packages/disell/scripts/paper_figures"
    )
sys.path.insert(0, str(PAPER_SCRIPTS))

from common import (  # noqa: E402
    load_config,
    load_registered_volume,
    masked_kam,
    spacing_from_config,
    watershed_feature_from_kam,
)

import darling  # noqa: E402
import disell  # noqa: E402


ACCENT = "#b2182b"
MINOR = "#777777"


def positive_label_count(labels):
    return int(np.unique(labels[labels > 0]).size)


def threshold_cores(kam, mask, threshold, min_size):
    interior = mask & np.isfinite(kam) & (kam < threshold)
    components, n_raw = ndimage.label(
        interior, structure=ndimage.generate_binary_structure(3, 1)
    )
    sizes_raw = np.bincount(components.ravel(), minlength=n_raw + 1)
    keep = np.flatnonzero(sizes_raw >= min_size)
    keep = keep[keep > 0]
    cores = np.where(np.isin(components, keep), components, 0).astype(np.int32)
    kept_sizes = sizes_raw[keep]
    return cores, kept_sizes


def contingency_diagnostics(reference, candidate, mask, fraction=0.10):
    """Merge/fragment fractions using contributions >= fraction of a region.

    A candidate region is counted as a merge when at least two reference
    regions each occupy >=10% of its voxels. Fragmentation is the symmetric
    calculation for reference regions. Zero labels are excluded.
    """
    valid = mask & (reference > 0) & (candidate > 0)
    a = reference[valid]
    b = candidate[valid]
    if not a.size:
        return {
            "merge_region_fraction": np.nan,
            "merge_voxel_fraction": np.nan,
            "fragmented_reference_fraction": np.nan,
            "fragmented_reference_voxel_fraction": np.nan,
        }
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((ua.size, ub.size), dtype=np.int64)
    np.add.at(joint, (ia, ib), 1)

    size_a = joint.sum(axis=1)
    size_b = joint.sum(axis=0)
    contributors_b = (joint >= fraction * size_b[None, :]).sum(axis=0)
    contributors_a = (joint >= fraction * size_a[:, None]).sum(axis=1)
    merged = contributors_b >= 2
    fragmented = contributors_a >= 2
    return {
        "merge_region_fraction": float(merged.mean()),
        "merge_voxel_fraction": float(size_b[merged].sum() / size_b.sum()),
        "fragmented_reference_fraction": float(fragmented.mean()),
        "fragmented_reference_voxel_fraction": float(
            size_a[fragmented].sum() / size_a.sum()
        ),
    }


def summary(labels, field, mask, voxel_um3):
    selected = mask & (labels > 0)
    lab = labels[selected]
    values = field[selected].astype(np.float64)
    if not lab.size:
        return {
            "n_labels": 0,
            "median_volume_um3": np.nan,
            "median_equivalent_diameter_um": np.nan,
            "median_rms_deg": np.nan,
        }
    counts = np.bincount(lab)
    sum_sq = np.zeros(counts.size, dtype=float)
    for channel in range(values.shape[-1]):
        means = (
            np.bincount(lab, weights=values[:, channel], minlength=counts.size)
            / np.maximum(counts, 1)
        )
        residual = values[:, channel] - means[lab]
        sum_sq += (
            np.bincount(lab, weights=residual**2, minlength=counts.size)
            / np.maximum(counts, 1)
        )
    used = counts > 0
    used[0] = False
    volume = counts[used] * voxel_um3
    rms = np.sqrt(sum_sq[used] / values.shape[-1])
    diameter = (6.0 * volume / np.pi) ** (1.0 / 3.0)
    return {
        "n_labels": int(used.sum()),
        "median_volume_um3": float(np.median(volume)),
        "median_equivalent_diameter_um": float(np.median(diameter)),
        "median_rms_deg": float(np.median(rms)),
    }


def row_for_threshold(percentile, threshold, cores, completed, core_sizes,
                      labels_ff, field, mask, voxel_um3):
    n_valid = int(mask.sum())
    completed_summary = summary(completed, field, mask, voxel_um3)
    row = {
        "percentile": float(percentile),
        "threshold_deg": float(threshold),
        "raw_n_regions": int(core_sizes.size),
        "raw_assigned_fraction": float(core_sizes.sum() / n_valid)
        if core_sizes.size else 0.0,
        "raw_largest_region_fraction": float(core_sizes.max() / n_valid)
        if core_sizes.size else 0.0,
        "raw_matched_overlap": disell.matched_overlap(
            labels_ff, cores, mask=mask, ignore_background=False
        ),
        "raw_vi_bits": disell.variation_of_information(
            labels_ff, cores, mask=mask, ignore_background=False
        ),
        "completed_matched_overlap": disell.matched_overlap(
            labels_ff, completed, mask=mask
        ),
        "completed_vi_bits": disell.variation_of_information(
            labels_ff, completed, mask=mask
        ),
        **{f"completed_{k}": v for k, v in completed_summary.items()},
    }
    row.update({
        f"completed_{k}": v for k, v in contingency_diagnostics(
            labels_ff, completed, mask
        ).items()
    })
    return row


def select_roi(labels_ff, labels_kam, mask, spacing_um, window_um=(24.8, 20.0)):
    """Choose a deterministic interior ROI from merge and boundary disagreement."""
    diagnostics = contingency_diagnostics(labels_ff, labels_kam, mask)
    del diagnostics  # full-map values are recorded separately

    valid = mask & (labels_ff > 0) & (labels_kam > 0)
    a = labels_ff[valid]
    b = labels_kam[valid]
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((ua.size, ub.size), dtype=np.int64)
    np.add.at(joint, (ia, ib), 1)
    size_b = joint.sum(axis=0)
    merged_b = (joint >= 0.10 * size_b[None, :]).sum(axis=0) >= 2
    merged_ids = ub[merged_b]
    merge_map = np.isin(labels_kam, merged_ids) & valid

    b_ff = np.stack([
        find_boundaries(labels_ff[z], mode="inner") for z in range(mask.shape[0])
    ])
    b_kam = np.stack([
        find_boundaries(labels_kam[z], mode="inner") for z in range(mask.shape[0])
    ])
    disagreement = np.logical_xor(
        ndimage.binary_dilation(b_ff, iterations=1),
        ndimage.binary_dilation(b_kam, iterations=1),
    ) & mask

    _, dy_um, dx_um = spacing_um
    wy = max(8, int(round(window_um[0] / dy_um)))
    wx = max(8, int(round(window_um[1] / dx_um)))
    wy = min(wy, mask.shape[1] - 10)
    wx = min(wx, mask.shape[2] - 10)
    kernel = np.ones((wy, wx), dtype=float)

    best = None
    for z in range(1, mask.shape[0] - 1):
        score_map = (
            2.0 * ndimage.convolve(merge_map[z].astype(float), kernel, mode="constant")
            + ndimage.convolve(disagreement[z].astype(float), kernel, mode="constant")
        ) / kernel.size
        valid_fraction = (
            ndimage.convolve(mask[z].astype(float), kernel, mode="constant")
            / kernel.size
        )
        score_map[valid_fraction < 0.995] = -np.inf
        margin_y = wy // 2 + 2
        margin_x = wx // 2 + 2
        score_map[:margin_y] = -np.inf
        score_map[-margin_y:] = -np.inf
        score_map[:, :margin_x] = -np.inf
        score_map[:, -margin_x:] = -np.inf
        flat = int(np.argmax(score_map))
        score = float(score_map.ravel()[flat])
        cy, cx = np.unravel_index(flat, score_map.shape)
        candidate = (score, -z, -cy, -cx, z, cy, cx)
        if best is None or candidate > best:
            best = candidate

    if best is None or not np.isfinite(best[0]):
        raise RuntimeError("no valid interior ROI found")
    score, _, _, _, z, cy, cx = best
    y0 = cy - wy // 2
    x0 = cx - wx // 2
    y1, x1 = y0 + wy, x0 + wx
    return {
        "slice_index_zero_based": int(z),
        "pixel_bounds_yx": [int(y0), int(y1), int(x0), int(x1)],
        "physical_bounds_um_xy": [
            float(x0 * dx_um), float(x1 * dx_um),
            float(y0 * dy_um), float(y1 * dy_um),
        ],
        "window_shape_yx": [int(wy), int(wx)],
        "window_size_um_yx": [float(wy * dy_um), float(wx * dx_um)],
        "selection_score": float(score),
        "valid_fraction": float(mask[z, y0:y1, x0:x1].mean()),
        "merge_voxel_fraction": float(merge_map[z, y0:y1, x0:x1].mean()),
        "boundary_disagreement_fraction": float(
            disagreement[z, y0:y1, x0:x1].mean()
        ),
        "selection_rule": (
            "maximum interior sliding-window mean of 2*many-to-one merge "
            "membership plus dilated boundary XOR; >=99.5% valid; lexicographic tie-break"
        ),
    }


def add_scale_bar(ax, length_um, dx_um, roi_width, roi_height):
    length_px = length_um / dx_um
    x0 = 0.07 * roi_width
    y0 = 0.91 * roi_height
    ax.plot([x0, x0 + length_px], [y0, y0], color="k", lw=1.5,
            solid_capstyle="butt")
    ax.text(x0 + length_px / 2, y0 - 0.035 * roi_height,
            rf"{length_um:g} $\mu$m", ha="center", va="top", fontsize=7)


def make_main_figure(rows, selected, field, mask, labels_ff, labels_kam,
                     roi, spacing_um, out_pdf, out_png):
    p = np.array([r["percentile"] for r in rows])
    overlap = np.array([r["completed_matched_overlap"] for r in rows])
    largest = np.array([r["raw_largest_region_fraction"] for r in rows])

    z = roi["slice_index_zero_based"]
    y0, y1, x0, x1 = roi["pixel_bounds_yx"]
    crop = np.s_[z, y0:y1, x0:x1]

    lo = np.array([np.nanmin(field[..., c]) for c in range(2)])
    hi = np.array([np.nanmax(field[..., c]) for c in range(2)])
    norm = np.stack([lo - 0.001 * (hi - lo), hi + 0.001 * (hi - lo)], axis=1)
    rgb = darling.transforms.rgb(field[z], norm=norm)[0]
    rgb[~mask[z]] = 1.0
    rgb_crop = rgb[y0:y1, x0:x1]
    b_ff = find_boundaries(labels_ff[crop], mode="inner")
    b_kam = find_boundaries(labels_kam[crop], mode="inner")

    fig = plt.figure(figsize=(7.0, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.08, 1, 1), wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(p, overlap, color="k", label="matched overlap")
    ax.scatter([selected["percentile"]],
               [selected["completed_matched_overlap"]],
               s=24, facecolor=ACCENT, edgecolor="none", zorder=4)
    ax.set_xlabel("KAM threshold (percentile)")
    ax.set_ylabel("matched overlap")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(p, largest, color=MINOR, ls="--")
    ax2.set_ylabel("largest core / valid volume", color=MINOR)
    ax2.tick_params(axis="y", colors=MINOR)
    ax2.set_ylim(0, 1)
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, va="top",
            fontweight="bold")

    for j, (labels_boundary, panel) in enumerate(((b_ff, "(b)"), (b_kam, "(c)")), 1):
        image_ax = fig.add_subplot(gs[0, j])
        image_ax.imshow(rgb_crop, origin="upper", interpolation="nearest")
        image_ax.contour(labels_boundary.astype(float), levels=[0.5],
                         colors="k", linewidths=0.38, origin="upper")
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        image_ax.text(0.025, 0.975, panel, transform=image_ax.transAxes,
                      va="top", ha="left", fontweight="bold",
                      bbox=dict(facecolor="white", edgecolor="none", pad=0.8,
                                alpha=0.75))
        for spine in image_ax.spines.values():
            spine.set_linewidth(0.55)
        if j == 2:
            add_scale_bar(image_ax, 5.0, spacing_um[2],
                          rgb_crop.shape[1], rgb_crop.shape[0])

    fig.subplots_adjust(left=0.075, right=0.945, bottom=0.19, top=0.97)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def make_supplement(rows, selected, out_path):
    p = np.array([r["percentile"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    axes[0, 0].plot(p, [r["raw_n_regions"] for r in rows], color="k")
    axes[0, 0].set_ylabel("retained KAM cores")
    axes[0, 1].plot(p, [r["raw_assigned_fraction"] for r in rows],
                    color="k", label="assigned")
    axes[0, 1].plot(p, [r["raw_largest_region_fraction"] for r in rows],
                    color=MINOR, ls="--", label="largest")
    axes[0, 1].set_ylabel("fraction of valid volume")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(p, [r["completed_matched_overlap"] for r in rows],
                    color="k", label="overlap")
    axes[1, 0].set_ylabel("matched overlap")
    axes[1, 1].plot(p, [r["completed_vi_bits"] for r in rows], color="k")
    axes[1, 1].set_ylabel("variation of information (bits)")
    for ax in axes.ravel():
        ax.axvline(selected["percentile"], color=ACCENT, lw=0.8)
        ax.spines[["top"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("KAM threshold (percentile)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--manuscript-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.out_root)
    out = out_root / "kam_baseline"
    out.mkdir(parents=True, exist_ok=True)
    manuscript = Path(args.manuscript_dir)
    figures = manuscript / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    field, mask, transforms, attrs = load_registered_volume(
        out_root / "volume" / "volume_registered.h5"
    )
    with h5py.File(
        out_root / "segmentation_3d" / "labels_3d.h5", "r"
    ) as handle:
        labels_ff = handle["labels"][...].astype(np.int32)

    expected_shape = (11, 200, 500, 2)
    if field.shape != expected_shape or mask.shape != expected_shape[:3]:
        raise ValueError(f"unexpected volume shapes: {field.shape}, {mask.shape}")
    if cfg["angle_unit"] != "deg" or cfg["channel_names"] != ["chi", "phi"]:
        raise ValueError("unexpected channel metadata")
    if tuple(cfg["spacing_nm_zyx"]) != (500.0, 1240.0, 400.0):
        raise ValueError("unexpected physical spacing")
    if not np.isfinite(field[mask]).all() or (labels_ff[~mask] > 0).any():
        raise ValueError("invalid mask/label relationship")

    spacing = spacing_from_config(cfg)
    spacing_um = tuple(v / 1e3 for v in spacing.as_tuple_nm())
    voxel_um3 = np.prod(spacing_um)
    pseg = cfg["segmentation"]
    if not (
        pseg["local_threshold_deg"] == 0.06
        and pseg["global_threshold_deg"] == 0.30
        and pseg["footprint_tolerance"] == 0.85
        and pseg["min_grain_size"] == 20
        and pseg["random_seed"] == 42
        and pseg["kam_radius_nm"] == 1300.0
    ):
        raise ValueError("configuration differs from manuscript parameters")

    from pf_io import isotropic_physical_footprint
    footprint = isotropic_physical_footprint(
        spacing, radius_nm=pseg["kam_radius_nm"], ndim=3
    )
    kam = masked_kam(field, mask, footprint)
    feature = watershed_feature_from_kam(kam, mask)
    valid_kam = kam[mask & np.isfinite(kam)]

    percentiles = np.arange(1.0, 99.0 + 0.001, 0.5)
    thresholds = np.percentile(valid_kam, percentiles)
    rows = []
    best = None
    best_labels = None
    best_cores = None
    for percentile, threshold in zip(percentiles, thresholds):
        cores, core_sizes = threshold_cores(
            kam, mask, float(threshold), pseg["min_grain_size"]
        )
        completed = disell.region_grow_watershed(
            cores, mask, feature, connectivity=pseg["watershed_connectivity"]
        ).astype(np.int32)
        row = row_for_threshold(
            percentile, threshold, cores, completed, core_sizes,
            labels_ff, field, mask, voxel_um3,
        )
        rows.append(row)
        key = (
            -np.inf if not np.isfinite(row["completed_matched_overlap"])
            else row["completed_matched_overlap"],
            np.inf if not np.isfinite(row["completed_vi_bits"])
            else -row["completed_vi_bits"],
            -percentile,
        )
        if best is None or key > best[0]:
            best = (key, dict(row))
            best_labels = completed.copy()
            best_cores = cores.copy()

    selected = best[1]
    # Explicit label-invariance check with a deterministic permutation.
    ids = np.unique(best_labels)
    positive = ids[ids > 0]
    permuted = best_labels.copy()
    for old, new in zip(positive, positive[::-1]):
        permuted[best_labels == old] = new
    invariance = {
        "self_overlap_after_label_permutation": disell.matched_overlap(
            best_labels, permuted, mask=mask
        ),
        "self_vi_after_label_permutation_bits": disell.variation_of_information(
            best_labels, permuted, mask=mask
        ),
    }
    if not (
        np.isclose(invariance["self_overlap_after_label_permutation"], 1.0)
        and np.isclose(invariance["self_vi_after_label_permutation_bits"], 0.0)
    ):
        raise AssertionError("agreement metrics are not label-invariant")

    roi = select_roi(labels_ff, best_labels, mask, spacing_um)
    ff_summary = summary(labels_ff, field, mask, voxel_um3)
    selected_diag = contingency_diagnostics(labels_ff, best_labels, mask)
    selected_raw_diag = contingency_diagnostics(labels_ff, best_cores, mask)

    with open(out / "threshold_sweep_fine.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out / "selected_comparison.npz",
        labels_flood_fill=labels_ff,
        labels_kam=best_labels,
        cores_kam=best_cores,
        mask=mask,
    )
    metadata = {
        "dataset": cfg["dataset_name"],
        "volume_shape_zyxc": list(field.shape),
        "valid_voxels": int(mask.sum()),
        "valid_fraction": float(mask.mean()),
        "layer_names": [f"layer_{i}_1" for i in range(1, 12)],
        "channel_names": cfg["channel_names"],
        "angle_unit": cfg["angle_unit"],
        "spacing_um_zyx": list(spacing_um),
        "registration_transforms_yx": transforms.tolist(),
        "flood_fill_parameters": pseg,
        "flood_fill_summary": ff_summary,
        "flood_fill_labels": positive_label_count(labels_ff),
        "kam_radius_um": pseg["kam_radius_nm"] / 1e3,
        "kam_footprint_shape": list(footprint.shape),
        "kam_footprint_voxels_including_centre": int(footprint.sum()),
        "kam_max_contributing_neighbours": int(footprint.sum() - 1),
        "sweep_percentile_start": float(percentiles[0]),
        "sweep_percentile_stop": float(percentiles[-1]),
        "sweep_percentile_step": 0.5,
        "sweep_threshold_deg_start": float(thresholds[0]),
        "sweep_threshold_deg_stop": float(thresholds[-1]),
        "selection_rule": (
            "maximum completed matched overlap with flood-fill partition; "
            "lower VI then lower percentile as tie-breakers"
        ),
        "selected": selected,
        "selected_merge_fragment_diagnostics": selected_diag,
        "selected_raw_merge_fragment_diagnostics": selected_raw_diag,
        "label_invariance_check": invariance,
        "roi": roi,
    }
    with open(out / "analysis_summary.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    make_main_figure(
        rows, selected, field, mask, labels_ff, best_labels, roi, spacing_um,
        figures / "kam3d_threshold_response_6-2pct.pdf",
        figures / "kam3d_threshold_response_6-2pct.png",
    )
    make_supplement(
        rows, selected, figures / "kam3d_threshold_diagnostics_6-2pct.pdf"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

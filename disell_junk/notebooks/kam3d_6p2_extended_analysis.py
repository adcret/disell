#!/usr/bin/env python
"""Reproduce and extend the 6.2% notebook KAM comparison.

The baseline definitions are copied directly from inspect_segmentations.ipynb.
The extension varies only KAM physical radius and threshold percentile, plus
an explicit registration sensitivity analysis. Flood-fill and KAM watershed
completion retain the notebook mask, physical sampling, minimum size and
connectivity.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
})
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from skimage.segmentation import find_boundaries

PACKAGE_ROOT = Path("/home/adam/Documents/Scripts/packages/disell")
NOTEBOOK = PACKAGE_ROOT / "notebooks" / "inspect_segmentations.ipynb"
OUTPUT = PACKAGE_ROOT / "notebooks" / "kam3d_6p2_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PACKAGE_ROOT / "scripts" / "paper_figures"))

import darling  # noqa: E402
import disell  # noqa: E402
from common import flood_footprint, watershed_feature_from_kam  # noqa: E402
from pf_io import VoxelSpacing, isotropic_physical_footprint  # noqa: E402


RADII_UM = (0.5, 0.8, 1.0, 1.25, 1.3, 1.5, 2.0)
INITIAL_PERCENTILES = np.arange(1.0, 100.0, 1.0)
MINIMUM_VOLUME_UM3 = 2.5
SEED = 42
TAU_LOC = 0.02
TAU_GLOB = 0.10
TAU_FP = 0.35
MAX_ITERATIONS = 5000
STAGNATION_TOLERANCE = 2000
MERGE_FRACTION = 0.10


def kam_lowmem(field, mask, footprint):
    """Exact low-memory KAM implementation from notebook cell 13."""
    f = np.where(mask[..., None], field, np.nan).astype(np.float32)
    channels = f.shape[-1]
    offsets = np.argwhere(footprint) - (np.array(footprint.shape) // 2)
    offsets = offsets[~(offsets == 0).all(axis=1)]
    accumulator = np.zeros(f.shape[:-1], np.float32)
    counts = np.zeros(f.shape[:-1], np.int32)

    def pair(delta, length):
        if delta >= 0:
            return slice(0, length - delta), slice(delta, length)
        return slice(-delta, length), slice(0, length + delta)

    for offset in offsets:
        centre, neighbour = zip(
            *[pair(int(delta), length)
              for delta, length in zip(offset, f.shape[:-1])]
        )
        distance_sq = np.zeros(f[centre].shape[:-1], np.float32)
        for channel in range(channels):
            difference = f[neighbour + (channel,)] - f[centre + (channel,)]
            distance_sq += difference * difference
        magnitude = np.sqrt(distance_sq / channels)
        finite = np.isfinite(magnitude)
        accumulator[centre] += np.where(finite, magnitude, 0.0).astype(np.float32)
        counts[centre] += finite

    result = np.divide(
        accumulator, counts,
        out=np.full(accumulator.shape, np.nan, np.float32),
        where=counts > 0,
    )
    result[~mask] = np.nan
    return result


def run_flood_fill(field, mask, min_voxels, kam_map):
    result, _ = disell.flood_fill_dfxm_two_stage(
        np.nan_to_num(field, nan=0.0).astype(np.float32),
        footprint=flood_footprint(
            {"segmentation": {"flood_footprint": "inplane8_plus_z"}}
        ),
        local_misorientation_threshold=TAU_LOC,
        global_threshold=TAU_GLOB,
        footprint_tolerance=TAU_FP,
        mask=mask.astype(np.uint8),
        max_iterations=MAX_ITERATIONS,
        min_grain_size=min_voxels,
        stagnation_tolerance=STAGNATION_TOLERANCE,
        random_seed=SEED,
    )
    markers = np.asarray(result["segmentation"], dtype=np.int32)
    feature = watershed_feature_from_kam(kam_map, mask)
    labels = np.asarray(
        disell.region_grow_watershed(markers, mask, feature, connectivity=1),
        dtype=np.int32,
    )
    return markers, labels


def segment_kam(kam_map, mask, percentile, min_voxels):
    finite_values = kam_map[mask & np.isfinite(kam_map)]
    threshold = float(np.percentile(finite_values, percentile))
    interior = mask & np.isfinite(kam_map) & (kam_map < threshold)
    components, _ = ndimage.label(
        interior, structure=ndimage.generate_binary_structure(mask.ndim, 1)
    )
    sizes = np.bincount(components.ravel())
    keep = np.flatnonzero(sizes >= min_voxels)
    keep = keep[keep > 0]
    markers = np.where(np.isin(components, keep), components, 0).astype(np.int32)
    feature = watershed_feature_from_kam(kam_map, mask)
    labels = np.asarray(
        disell.region_grow_watershed(markers, mask, feature, connectivity=1),
        dtype=np.int32,
    )
    return threshold, markers, labels


def notebook_spread_summary(field, labels, mask, voxel_volume_um3):
    """Notebook's sqrt(var_chi + var_phi), retained without correction."""
    selected = mask & (labels > 0)
    lab = labels[selected]
    values = field[selected].astype(np.float64)
    length = int(lab.max()) + 1
    counts = np.bincount(lab, minlength=length)
    variance_sum = np.zeros(length)
    for channel in range(values.shape[-1]):
        means = (
            np.bincount(lab, weights=values[:, channel], minlength=length)
            / np.maximum(counts, 1)
        )
        residual = values[:, channel] - means[lab]
        variance_sum += (
            np.bincount(lab, weights=residual * residual, minlength=length)
            / np.maximum(counts, 1)
        )
    keep = counts > 0
    keep[0] = False
    volumes = counts[keep] * voxel_volume_um3
    aggregate_spread = np.sqrt(variance_sum[keep])
    per_channel_rms = np.sqrt(variance_sum[keep] / values.shape[-1])
    diameters = (6.0 * volumes / np.pi) ** (1.0 / 3.0)
    return {
        "n_regions": int(keep.sum()),
        "median_volume_um3": float(np.median(volumes)),
        "median_equivalent_diameter_um": float(np.median(diameters)),
        "median_notebook_aggregate_spread_deg": float(
            np.median(aggregate_spread)
        ),
        "median_per_channel_rms_deg": float(np.median(per_channel_rms)),
        "largest_completed_region_fraction": float(
            counts[keep].max() / mask.sum()
        ),
    }


def contingency_metrics(reference, candidate, mask):
    """Label-invariant metrics plus declared many-to-one diagnostics."""
    valid = mask & (reference > 0) & (candidate > 0)
    a = reference[valid]
    b = candidate[valid]
    if not a.size:
        return {
            "matched_overlap": np.nan,
            "vi_bits": np.nan,
            "merge_region_fraction": np.nan,
            "merge_voxel_fraction": np.nan,
            "split_reference_region_fraction": np.nan,
            "split_reference_voxel_fraction": np.nan,
        }
    labels_a, inverse_a = np.unique(a, return_inverse=True)
    labels_b, inverse_b = np.unique(b, return_inverse=True)
    joint = np.zeros((labels_a.size, labels_b.size), dtype=np.int64)
    np.add.at(joint, (inverse_a, inverse_b), 1)
    size_a = joint.sum(axis=1).astype(float)
    size_b = joint.sum(axis=0).astype(float)
    total = joint.sum()

    union = size_a[:, None] + size_b[None, :] - joint
    iou = np.divide(
        joint, union, out=np.zeros_like(union, dtype=float), where=union > 0
    )
    matched = 0.5 * (
        (iou.max(axis=1) * size_a).sum() / size_a.sum()
        + (iou.max(axis=0) * size_b).sum() / size_b.sum()
    )

    probability = joint / total
    marginal_a = probability.sum(axis=1)
    marginal_b = probability.sum(axis=0)
    nonzero = probability > 0
    joint_entropy = -np.sum(
        probability[nonzero] * np.log2(probability[nonzero])
    )
    entropy_a = -np.sum(marginal_a * np.log2(marginal_a))
    entropy_b = -np.sum(marginal_b * np.log2(marginal_b))
    vi = 2 * joint_entropy - entropy_a - entropy_b

    contributors_to_b = (
        joint >= MERGE_FRACTION * size_b[None, :]
    ).sum(axis=0)
    contributors_to_a = (
        joint >= MERGE_FRACTION * size_a[:, None]
    ).sum(axis=1)
    merged = contributors_to_b >= 2
    split = contributors_to_a >= 2
    return {
        "matched_overlap": float(matched),
        "vi_bits": float(vi),
        "merge_region_fraction": float(merged.mean()),
        "merge_voxel_fraction": float(size_b[merged].sum() / size_b.sum()),
        "split_reference_region_fraction": float(split.mean()),
        "split_reference_voxel_fraction": float(
            size_a[split].sum() / size_a.sum()
        ),
    }


def validate_metric_implementation(reference, candidate, mask, calculated):
    expected_overlap = disell.matched_overlap(reference, candidate, mask=mask)
    expected_vi = disell.variation_of_information(reference, candidate, mask=mask)
    if not np.isclose(calculated["matched_overlap"], expected_overlap, atol=1e-12):
        raise AssertionError("matched-overlap helper differs from disell")
    if not np.isclose(calculated["vi_bits"], expected_vi, atol=1e-12):
        raise AssertionError("VI helper differs from disell")

    positive = np.unique(candidate[candidate > 0])
    permuted = candidate.copy()
    for old, new in zip(positive, positive[::-1]):
        permuted[candidate == old] = new
    if not np.isclose(
        disell.matched_overlap(candidate, permuted, mask=mask), 1.0
    ):
        raise AssertionError("matched overlap is not label-invariant")
    if not np.isclose(
        disell.variation_of_information(candidate, permuted, mask=mask), 0.0,
        atol=1e-10,
    ):
        raise AssertionError("VI is not label-invariant")


def raw_marker_metrics(reference, markers, mask):
    assigned = mask & (markers > 0)
    if not assigned.any():
        return {
            "raw_matched_overlap": np.nan,
            "raw_vi_bits": np.nan,
            "raw_assigned_fraction": 0.0,
            "largest_marker_fraction": 0.0,
        }
    sizes = np.bincount(markers[assigned])
    sizes = sizes[1:]
    return {
        "raw_matched_overlap": float(
            disell.matched_overlap(reference, markers, mask=mask)
        ),
        "raw_vi_bits": float(
            disell.variation_of_information(reference, markers, mask=mask)
        ),
        "raw_assigned_fraction": float(assigned.sum() / mask.sum()),
        "largest_marker_fraction": float(sizes.max() / mask.sum()),
    }


def footprint_record(radius_um, footprint, spacing_um):
    centre = np.array(footprint.shape) // 2
    offsets = np.argwhere(footprint) - centre
    offsets = offsets[~(offsets == 0).all(axis=1)]
    records = []
    for offset in offsets:
        physical = offset * np.asarray(spacing_um)
        records.append({
            "offset_zyx": [int(value) for value in offset],
            "physical_offset_um_zyx": [float(value) for value in physical],
            "distance_um": float(np.linalg.norm(physical)),
        })
    return {
        "radius_um": float(radius_um),
        "bounding_box_zyx": list(footprint.shape),
        "included_neighbours": int(len(records)),
        "z_offset_values": sorted({record["offset_zyx"][0] for record in records}),
        "y_offset_values": sorted({record["offset_zyx"][1] for record in records}),
        "x_offset_values": sorted({record["offset_zyx"][2] for record in records}),
        "maximum_distance_um": max(record["distance_um"] for record in records),
        "offsets": records,
    }


def standardised_channel(image, valid):
    values = image[valid]
    result = np.zeros_like(image, dtype=np.float32)
    if values.size:
        result[valid] = (
            (values - values.mean()) / (values.std() + 1e-8)
        ).astype(np.float32)
    return result


def adjacent_shift_diagnostics(field, mask, spacing_um):
    rows = []
    for z in range(field.shape[0] - 1):
        common = mask[z] & mask[z + 1]
        channel_shifts = []
        for channel, name in enumerate(("chi", "phi")):
            reference = standardised_channel(field[z, ..., channel], common)
            moving = standardised_channel(field[z + 1, ..., channel], common)
            shift, error, _ = phase_cross_correlation(
                reference, moving, upsample_factor=10, normalization=None
            )
            channel_shifts.append(shift)
            rows.append({
                "layer_from": z,
                "layer_to": z + 1,
                "channel": name,
                "shift_row_pixels": float(shift[0]),
                "shift_col_pixels": float(shift[1]),
                "shift_magnitude_pixels": float(np.linalg.norm(shift)),
                "shift_row_um": float(shift[0] * spacing_um[1]),
                "shift_col_um": float(shift[1] * spacing_um[2]),
                "shift_magnitude_um": float(np.linalg.norm(
                    shift * np.asarray(spacing_um[1:])
                )),
                "registration_error": float(error),
            })
        mean_shift = np.mean(channel_shifts, axis=0)
        rows.append({
            "layer_from": z,
            "layer_to": z + 1,
            "channel": "mean_chi_phi",
            "shift_row_pixels": float(mean_shift[0]),
            "shift_col_pixels": float(mean_shift[1]),
            "shift_magnitude_pixels": float(np.linalg.norm(mean_shift)),
            "shift_row_um": float(mean_shift[0] * spacing_um[1]),
            "shift_col_um": float(mean_shift[1] * spacing_um[2]),
            "shift_magnitude_um": float(np.linalg.norm(
                mean_shift * np.asarray(spacing_um[1:])
            )),
            "registration_error": np.nan,
        })
    return rows


def registered_sensitivity_field(field, mask):
    registration_input = field.astype(np.float32).copy()
    registration_input[~mask] = np.nan
    transforms = disell.register(
        registration_input, registration_channel=0,
        upsample_factor=10, normalization=None,
    )
    registered = disell.apply_transforms(registration_input, transforms)
    shifted_mask = disell.apply_transforms(
        mask.astype(np.float32)[..., None], transforms
    )[..., 0]
    registered_mask = (
        np.isfinite(shifted_mask)
        & (shifted_mask >= 0.999)
        & np.isfinite(registered).all(axis=-1)
    )
    registered[~registered_mask] = np.nan
    return registered, registered_mask, transforms


def sweep_dataset(name, field, mask, spacing_um, min_voxels):
    spacing_nm = VoxelSpacing(*(value * 1e3 for value in spacing_um))
    voxel_volume_um3 = float(np.prod(spacing_um))
    rows = []
    radius_state = {}
    footprint_rows = []

    # Flood-fill marker identification is independent of KAM radius. The
    # notebook completes the markers using the 0.5-um KAM field.
    baseline_footprint = isotropic_physical_footprint(
        spacing_nm, radius_nm=500.0, ndim=3
    )
    baseline_kam = kam_lowmem(field, mask, baseline_footprint)
    flood_markers, labels_flood = run_flood_fill(
        field, mask, min_voxels, baseline_kam
    )

    for radius_um in RADII_UM:
        footprint = isotropic_physical_footprint(
            spacing_nm, radius_nm=radius_um * 1e3, ndim=3
        )
        footprint_rows.append(
            footprint_record(radius_um, footprint, spacing_um)
        )
        kam_map = kam_lowmem(field, mask, footprint)
        radius_rows = []
        best_arrays = None
        best_array_key = None

        def evaluate(percentile):
            threshold, markers, labels = segment_kam(
                kam_map, mask, percentile, min_voxels
            )
            stats = notebook_spread_summary(
                field, labels, mask, voxel_volume_um3
            )
            agreement = contingency_metrics(labels_flood, labels, mask)
            raw = raw_marker_metrics(labels_flood, markers, mask)
            row = {
                "dataset_variant": name,
                "radius_um": float(radius_um),
                "percentile": float(percentile),
                "threshold_deg": threshold,
                "retained_markers": int(np.unique(markers[markers > 0]).size),
                "completed_regions": stats["n_regions"],
                "median_completed_volume_um3": stats["median_volume_um3"],
                "median_completed_equivalent_diameter_um":
                    stats["median_equivalent_diameter_um"],
                "median_notebook_aggregate_spread_deg":
                    stats["median_notebook_aggregate_spread_deg"],
                "median_per_channel_rms_deg":
                    stats["median_per_channel_rms_deg"],
                "largest_completed_region_fraction":
                    stats["largest_completed_region_fraction"],
                **raw,
                **agreement,
            }
            return row, markers, labels

        for percentile in INITIAL_PERCENTILES:
            row, markers, labels = evaluate(percentile)
            radius_rows.append(row)
            key = (
                row["matched_overlap"], -row["vi_bits"],
                -row["largest_marker_fraction"],
            )
            if best_array_key is None or key > best_array_key:
                best_array_key = key
                best_arrays = (markers.copy(), labels.copy())

        best_initial = max(
            radius_rows, key=lambda item: item["matched_overlap"]
        )
        largest = np.array([
            item["largest_marker_fraction"] for item in radius_rows
        ])
        transition_index = int(np.argmax(np.abs(np.diff(largest)))) + 1
        transition_percentile = radius_rows[transition_index]["percentile"]
        refinement_centres = (
            best_initial["percentile"], transition_percentile
        )
        refined = set()
        for centre in refinement_centres:
            refined.update(np.arange(
                max(1.0, centre - 2.0),
                min(99.0, centre + 2.0) + 0.001,
                0.25,
            ))
        existing = {item["percentile"] for item in radius_rows}
        for percentile in sorted(refined - existing):
            row, markers, labels = evaluate(float(percentile))
            radius_rows.append(row)
            key = (
                row["matched_overlap"], -row["vi_bits"],
                -row["largest_marker_fraction"],
            )
            if key > best_array_key:
                best_array_key = key
                best_arrays = (markers.copy(), labels.copy())

        radius_rows.sort(key=lambda item: item["percentile"])
        best_radius = max(
            radius_rows, key=lambda item: (
                item["matched_overlap"], -item["vi_bits"],
                -item["largest_marker_fraction"],
            )
        )
        radius_state[radius_um] = {
            "kam": kam_map,
            "best": best_radius,
            "markers": best_arrays[0],
            "labels": best_arrays[1],
        }
        rows.extend(radius_rows)
        print(
            f"{name}: radius {radius_um:g} um best "
            f"p={best_radius['percentile']:.2f}, "
            f"overlap={best_radius['matched_overlap']:.4f}, "
            f"VI={best_radius['vi_bits']:.4f}"
        )

    global_best = max(
        rows, key=lambda item: (
            item["matched_overlap"], -item["vi_bits"],
            -item["largest_marker_fraction"],
        )
    )
    state = radius_state[global_best["radius_um"]]
    validate_metric_implementation(
        labels_flood, state["labels"], mask,
        contingency_metrics(labels_flood, state["labels"], mask),
    )
    return {
        "rows": rows,
        "footprints": footprint_rows,
        "labels_flood": labels_flood,
        "markers_flood": flood_markers,
        "global_best": global_best,
        "best_kam": state["kam"],
        "best_markers": state["markers"],
        "best_labels": state["labels"],
        "mask": mask,
        "field": field,
    }


def physical_boundary_dilation(boundary, tolerance_um, dy_um, dx_um):
    ry = int(np.ceil(tolerance_um / dy_um))
    rx = int(np.ceil(tolerance_um / dx_um))
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    structure = (yy * dy_um) ** 2 + (xx * dx_um) ** 2 <= tolerance_um**2
    return ndimage.binary_dilation(boundary, structure=structure)


def select_roi_candidates(
    rgb, labels_flood, labels_kam, mask, spacing_um,
    window_um_yx=(29.76, 30.0), tolerance_um=1.3,
):
    _, dy_um, dx_um = spacing_um
    height = int(round(window_um_yx[0] / dy_um))
    width = int(round(window_um_yx[1] / dx_um))
    kernel = np.ones((height, width), dtype=np.float32)
    candidates = []

    for z in range(1, mask.shape[0] - 1):
        boundary_flood = find_boundaries(
            labels_flood[z], mode="inner"
        ) & ndimage.binary_erosion(mask[z])
        boundary_kam = find_boundaries(
            labels_kam[z], mode="inner"
        ) & ndimage.binary_erosion(mask[z])
        matched_by_kam = physical_boundary_dilation(
            boundary_kam, tolerance_um, dy_um, dx_um
        )
        matched_by_flood = physical_boundary_dilation(
            boundary_flood, tolerance_um, dy_um, dx_um
        )
        unmatched_flood = boundary_flood & ~matched_by_kam
        unmatched_kam = boundary_kam & ~matched_by_flood

        count_flood = ndimage.convolve(
            boundary_flood.astype(np.float32), kernel, mode="constant"
        )
        count_kam = ndimage.convolve(
            boundary_kam.astype(np.float32), kernel, mode="constant"
        )
        miss_flood = ndimage.convolve(
            unmatched_flood.astype(np.float32), kernel, mode="constant"
        )
        miss_kam = ndimage.convolve(
            unmatched_kam.astype(np.float32), kernel, mode="constant"
        )
        valid_fraction = ndimage.convolve(
            mask[z].astype(np.float32), kernel, mode="constant"
        ) / kernel.size
        score = 0.5 * (
            np.divide(
                miss_flood, count_flood,
                out=np.zeros_like(miss_flood), where=count_flood > 0,
            )
            + np.divide(
                miss_kam, count_kam,
                out=np.zeros_like(miss_kam), where=count_kam > 0,
            )
        )

        margin_y = height // 2 + 2
        margin_x = width // 2 + 2
        allowed = valid_fraction >= 0.999
        allowed[:margin_y] = False
        allowed[-margin_y:] = False
        allowed[:, :margin_x] = False
        allowed[:, -margin_x:] = False
        allowed &= count_flood >= 20
        allowed &= count_kam >= 20
        score[~allowed] = -np.inf

        work = score.copy()
        for _ in range(8):
            flat = int(np.argmax(work))
            value = float(work.ravel()[flat])
            if not np.isfinite(value):
                break
            cy, cx = np.unravel_index(flat, work.shape)
            y0, x0 = cy - height // 2, cx - width // 2
            y1, x1 = y0 + height, x0 + width
            labels_ff_roi = np.unique(labels_flood[z, y0:y1, x0:x1])
            labels_kam_roi = np.unique(labels_kam[z, y0:y1, x0:x1])
            labels_ff_roi = labels_ff_roi[labels_ff_roi > 0]
            labels_kam_roi = labels_kam_roi[labels_kam_roi > 0]
            if labels_ff_roi.size > 1 and labels_kam_roi.size > 1:
                candidates.append({
                    "layer_index": int(z),
                    "array_bounds_yx": [int(y0), int(y1), int(x0), int(x1)],
                    "physical_size_um_yx": [
                        float(height * dy_um), float(width * dx_um)
                    ],
                    "boundary_disagreement_score": value,
                    "unmatched_flood_boundary_fraction": float(
                        miss_flood[cy, cx] / count_flood[cy, cx]
                    ),
                    "unmatched_kam_boundary_fraction": float(
                        miss_kam[cy, cx] / count_kam[cy, cx]
                    ),
                    "flood_regions_intersecting": int(labels_ff_roi.size),
                    "kam_regions_intersecting": int(labels_kam_roi.size),
                    "valid_fraction": float(valid_fraction[cy, cx]),
                })
            work[
                max(0, cy - height):min(work.shape[0], cy + height + 1),
                max(0, cx - width):min(work.shape[1], cx + width + 1),
            ] = -np.inf

    candidates.sort(
        key=lambda item: (
            -item["boundary_disagreement_score"],
            item["layer_index"],
            item["array_bounds_yx"],
        )
    )
    selected = []
    for candidate in candidates:
        y0, y1, x0, x1 = candidate["array_bounds_yx"]
        overlaps_existing = False
        for existing in selected:
            if candidate["layer_index"] != existing["layer_index"]:
                continue
            ey0, ey1, ex0, ex1 = existing["array_bounds_yx"]
            intersection = max(0, min(y1, ey1) - max(y0, ey0)) * max(
                0, min(x1, ex1) - max(x0, ex0)
            )
            if intersection > 0.25 * (y1 - y0) * (x1 - x0):
                overlaps_existing = True
        if not overlaps_existing:
            selected.append(candidate)
        if len(selected) == 5:
            break
    return selected


def make_roi_previews(
    rgb, candidates, labels_flood, labels_kam, output_path
):
    figure, axes = plt.subplots(5, 2, figsize=(7.0, 8.2))
    for row, candidate in enumerate(candidates):
        z = candidate["layer_index"]
        y0, y1, x0, x1 = candidate["array_bounds_yx"]
        image = rgb[z, y0:y1, x0:x1]
        for column, (labels, name) in enumerate((
            (labels_flood, "flood fill"), (labels_kam, "KAM")
        )):
            ax = axes[row, column]
            ax.imshow(image, interpolation="nearest", origin="upper")
            boundary = find_boundaries(
                labels[z, y0:y1, x0:x1], mode="inner"
            )
            ax.contour(
                boundary.astype(float), levels=[0.5], colors="k",
                linewidths=0.45, origin="upper",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name)
            if column == 0:
                ax.set_ylabel(
                    f"#{row + 1}, z={z}\n"
                    f"score={candidate['boundary_disagreement_score']:.2f}"
                )
    figure.tight_layout(pad=0.5)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_proposed_figure(
    rows, best, rgb, roi, labels_flood, labels_kam, spacing_um, output_base
):
    radius_rows = [
        row for row in rows
        if row["radius_um"] == best["radius_um"]
        and float(row["percentile"]).is_integer()
    ]
    radius_rows.sort(key=lambda row: row["percentile"])
    percentiles = [row["percentile"] for row in radius_rows]
    overlaps = [row["matched_overlap"] for row in radius_rows]
    largest = [row["largest_marker_fraction"] for row in radius_rows]

    z = roi["layer_index"]
    y0, y1, x0, x1 = roi["array_bounds_yx"]
    image = rgb[z, y0:y1, x0:x1]
    figure = plt.figure(figsize=(7.1, 2.45))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.08, 1, 1), wspace=0.08)
    ax = figure.add_subplot(grid[0, 0])
    ax.plot(percentiles, overlaps, color="k")
    ax.plot(
        percentiles, largest, color="0.55", ls="--",
        label="largest marker",
    )
    ax.scatter(
        [best["percentile"]], [best["matched_overlap"]],
        s=22, color="#b2182b", zorder=4,
    )
    ax.set_xlabel("KAM threshold (percentile)")
    ax.set_ylabel("overlap or volume fraction")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.legend(
        ["matched overlap", "largest marker / valid volume"],
        loc="upper left", bbox_to_anchor=(0.02, 0.82),
        frameon=False, fontsize=6.5, handlelength=2.0,
    )
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, va="top",
            fontweight="bold")

    for index, (labels, panel) in enumerate((
        (labels_flood, "(b)"), (labels_kam, "(c)")
    ), start=1):
        image_ax = figure.add_subplot(grid[0, index])
        image_ax.imshow(image, interpolation="nearest", origin="upper")
        boundary = find_boundaries(
            labels[z, y0:y1, x0:x1], mode="inner"
        )
        image_ax.contour(
            boundary.astype(float), levels=[0.5], colors="k",
            linewidths=0.42, origin="upper",
        )
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        image_ax.text(
            0.025, 0.975, panel, transform=image_ax.transAxes,
            va="top", fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.75, "pad": 0.8},
        )
        if index == 2:
            length_um = 5.0
            length_pixels = length_um / spacing_um[2]
            x_start = 0.07 * image.shape[1]
            y_start = 0.88 * image.shape[0]
            image_ax.plot(
                [x_start, x_start + length_pixels],
                [y_start, y_start], color="k", lw=1.4,
                solid_capstyle="butt",
            )
            image_ax.text(
                x_start + length_pixels / 2,
                y_start - 0.055 * image.shape[0],
                r"5 $\mu$m", ha="center", va="top",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.7, "pad": 0.3},
            )
    figure.subplots_adjust(
        left=0.075, right=0.94, bottom=0.19, top=0.98
    )
    figure.savefig(
        output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02
    )
    figure.savefig(
        output_base.with_suffix(".png"), dpi=600,
        bbox_inches="tight", pad_inches=0.02,
    )
    plt.close(figure)


def make_supplementary_figure(rows, best, output_path):
    integer_rows = [
        row for row in rows if float(row["percentile"]).is_integer()
    ]
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    colours = plt.cm.viridis(np.linspace(0.08, 0.92, len(RADII_UM)))
    for radius, colour in zip(RADII_UM, colours):
        subset = sorted(
            [row for row in integer_rows if row["radius_um"] == radius],
            key=lambda row: row["percentile"],
        )
        percentile = [row["percentile"] for row in subset]
        axes[0, 0].plot(
            percentile, [row["retained_markers"] for row in subset],
            color=colour, label=f"{radius:g}",
        )
        axes[0, 1].plot(
            percentile, [row["largest_marker_fraction"] for row in subset],
            color=colour,
        )
        axes[1, 0].plot(
            percentile, [row["matched_overlap"] for row in subset],
            color=colour,
        )
        axes[1, 1].plot(
            percentile, [row["vi_bits"] for row in subset],
            color=colour,
        )
    axes[0, 0].set_ylabel("retained markers")
    axes[0, 1].set_ylabel("largest marker / valid volume")
    axes[1, 0].set_ylabel("matched overlap")
    axes[1, 1].set_ylabel("variation of information (bits)")
    for ax in axes[1]:
        ax.set_xlabel("KAM threshold (percentile)")
    axes[0, 0].legend(
        title=r"radius ($\mu$m)", ncol=2, frameon=False, fontsize=7
    )
    axes[1, 0].scatter(
        [best["percentile"]], [best["matched_overlap"]],
        color="#b2182b", s=20, zorder=5,
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = json.load(open(
        PACKAGE_ROOT / "scripts" / "paper_figures" / "config_6_2pct.json"
    ))
    data_root = Path(
        "~/Documents/Data/4dcells/111_cells_6-2pct_mosa_2x_raw"
    ).expanduser()
    volume = disell.load_layer_volume(
        data_root,
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
    spacing_um = (0.500, 1.240, 0.400)
    field[~mask] = np.nan
    voxel_volume_um3 = float(np.prod(spacing_um))
    min_voxels = max(
        1, int(round(MINIMUM_VOLUME_UM3 / voxel_volume_um3))
    )

    if field.shape != (11, 200, 500, 2):
        raise AssertionError(field.shape)
    if int(mask.sum()) != 1_100_000:
        raise AssertionError(mask.sum())

    # Exact notebook baseline.
    footprint_05 = isotropic_physical_footprint(
        VoxelSpacing(*(value * 1e3 for value in spacing_um)),
        radius_nm=500.0, ndim=3,
    )
    kam_05 = kam_lowmem(field, mask, footprint_05)
    markers_flood, labels_flood = run_flood_fill(
        field, mask, min_voxels, kam_05
    )
    baseline_rows = []
    baseline_segmentations = {}
    for percentile in np.arange(5, 51, 5):
        threshold, markers, labels = segment_kam(
            kam_05, mask, float(percentile), min_voxels
        )
        stats = notebook_spread_summary(
            field, labels, mask, voxel_volume_um3
        )
        agreement = contingency_metrics(labels_flood, labels, mask)
        validate_metric_implementation(
            labels_flood, labels, mask, agreement
        )
        baseline_rows.append({
            "percentile": float(percentile),
            "threshold_deg": threshold,
            "markers": int(np.unique(markers[markers > 0]).size),
            "labels": stats["n_regions"],
            "median_volume_um3": stats["median_volume_um3"],
            "median_notebook_aggregate_spread_deg":
                stats["median_notebook_aggregate_spread_deg"],
            "median_per_channel_rms_deg":
                stats["median_per_channel_rms_deg"],
            "vi_bits": agreement["vi_bits"],
            "matched_overlap": agreement["matched_overlap"],
        })
        baseline_segmentations[float(percentile)] = labels

    expected = {
        5: (0.00248, 531, 3.2520, 0.3105),
        10: (0.00338, 952, 2.6900, 0.4103),
        15: (0.00419, 1241, 2.4213, 0.4672),
        20: (0.00501, 1445, 2.2522, 0.5063),
        25: (0.00592, 1661, 2.1651, 0.5231),
        30: (0.00696, 1811, 2.1418, 0.5278),
        35: (0.00825, 1882, 2.1731, 0.5163),
        40: (0.00998, 1955, 2.2124, 0.5080),
        45: (0.01270, 1868, 2.3992, 0.4704),
        50: (0.01961, 1543, 2.9252, 0.3844),
    }
    checks = []
    for row in baseline_rows:
        threshold, count, vi, overlap = expected[int(row["percentile"])]
        checks.append({
            "percentile": row["percentile"],
            "threshold_matches_printed": bool(np.isclose(
                row["threshold_deg"], threshold, atol=5e-6
            )),
            "count_matches": row["labels"] == count,
            "vi_matches_printed": bool(np.isclose(
                row["vi_bits"], vi, atol=5e-5
            )),
            "overlap_matches_printed": bool(np.isclose(
                row["matched_overlap"], overlap, atol=5e-5
            )),
        })
    flood_stats = notebook_spread_summary(
        field, labels_flood, mask, voxel_volume_um3
    )
    displayed_stats = notebook_spread_summary(
        field, baseline_segmentations[15.0], mask, voxel_volume_um3
    )
    baseline_passed = (
        field.shape == (11, 200, 500, 2)
        and tuple(spacing_um) == (0.5, 1.24, 0.4)
        and int(mask.sum()) == 1_100_000
        and int(np.unique(labels_flood[labels_flood > 0]).size) == 3411
        and all(all(value for key, value in check.items()
                    if key != "percentile") for check in checks)
        and np.isclose(
            flood_stats["median_volume_um3"], 37.696, atol=0.001
        )
        and np.isclose(
            flood_stats["median_notebook_aggregate_spread_deg"],
            0.1836, atol=5e-5,
        )
        and np.isclose(
            displayed_stats["median_volume_um3"], 133.424, atol=0.001
        )
        and np.isclose(
            displayed_stats["median_notebook_aggregate_spread_deg"],
            0.3463, atol=5e-5,
        )
    )
    baseline_report = {
        "status": "passed" if baseline_passed else "failed",
        "source_notebook": str(NOTEBOOK),
        "feature_shape_zyxc": list(field.shape),
        "spacing_um_zyx": list(spacing_um),
        "valid_voxels": int(mask.sum()),
        "layer_order": volume.layer_names,
        "flood_fill_domains": int(
            np.unique(labels_flood[labels_flood > 0]).size
        ),
        "flood_fill_summary": flood_stats,
        "displayed_15pct_summary": displayed_stats,
        "sweep_checks": checks,
    }
    write_csv(OUTPUT / "baseline_reproduction_6p2.csv", baseline_rows)
    with open(OUTPUT / "baseline_reproduction_6p2.json", "w") as handle:
        json.dump(baseline_report, handle, indent=2)
    if not baseline_passed:
        raise AssertionError("baseline did not reproduce")

    # Fixed volume-wide RGB normalisation from notebook cell 10.
    low = np.array([np.nanmin(field[..., channel]) for channel in range(2)])
    high = np.array([np.nanmax(field[..., channel]) for channel in range(2)])
    normalisation = np.stack(
        [low - 0.001 * (high - low), high + 0.001 * (high - low)],
        axis=1,
    )
    rgb = np.stack([
        darling.transforms.rgb(field[z], norm=normalisation)[0]
        for z in range(field.shape[0])
    ])
    rgb[~mask] = 1.0

    # Preserve regenerated middle-layer notebook plots.
    layer = field.shape[0] // 2
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)
    for ax, labels, title in (
        (axes[0], None, "angular RGB"),
        (axes[1], labels_flood, "flood fill"),
        (axes[2], baseline_segmentations[15.0], "KAM threshold (15th percentile)"),
    ):
        image = rgb[layer].copy()
        if labels is not None:
            inner = ndimage.binary_erosion(mask[layer])
            boundary = find_boundaries(
                labels[layer], mode="inner"
            ) & inner
            image[boundary] = 0.0
        ax.imshow(image, interpolation="nearest")
        ax.set_title(title)
    figure.tight_layout()
    figure.savefig(
        OUTPUT / "baseline_middle_layer_overlays.png", dpi=250
    )
    plt.close(figure)

    shift_rows = adjacent_shift_diagnostics(field, mask, spacing_um)
    write_csv(OUTPUT / "adjacent_layer_shift_diagnostics.csv", shift_rows)
    registered_field, registered_mask, transforms = (
        registered_sensitivity_field(field, mask)
    )

    native = sweep_dataset(
        "native", field, mask, spacing_um, min_voxels
    )
    registered = sweep_dataset(
        "registered_sensitivity", registered_field, registered_mask,
        spacing_um, min_voxels,
    )
    all_rows = native["rows"] + registered["rows"]
    write_csv(OUTPUT / "kam_radius_threshold_sweep.csv", all_rows)

    best_by_radius = []
    for variant, result in (("native", native), ("registered_sensitivity", registered)):
        for radius in RADII_UM:
            subset = [
                row for row in result["rows"] if row["radius_um"] == radius
            ]
            best = max(
                subset, key=lambda row: (
                    row["matched_overlap"], -row["vi_bits"],
                    -row["largest_marker_fraction"],
                )
            )
            best_by_radius.append(best)
    write_csv(OUTPUT / "best_kam_threshold_by_radius.csv", best_by_radius)

    with open(OUTPUT / "footprint_offset_summary.json", "w") as handle:
        json.dump(native["footprints"], handle, indent=2)
    flat_footprints = []
    for footprint in native["footprints"]:
        for offset in footprint["offsets"]:
            flat_footprints.append({
                "radius_um": footprint["radius_um"],
                "bounding_box_zyx": "x".join(map(
                    str, footprint["bounding_box_zyx"]
                )),
                "included_neighbours": footprint["included_neighbours"],
                "offset_z": offset["offset_zyx"][0],
                "offset_y": offset["offset_zyx"][1],
                "offset_x": offset["offset_zyx"][2],
                "physical_offset_z_um": offset["physical_offset_um_zyx"][0],
                "physical_offset_y_um": offset["physical_offset_um_zyx"][1],
                "physical_offset_x_um": offset["physical_offset_um_zyx"][2],
                "distance_um": offset["distance_um"],
            })
    write_csv(OUTPUT / "footprint_offset_summary.csv", flat_footprints)

    mean_shift_rows = [
        row for row in shift_rows if row["channel"] == "mean_chi_phi"
    ]
    magnitudes_pixels = np.array([
        row["shift_magnitude_pixels"] for row in mean_shift_rows
    ])
    magnitudes_um = np.array([
        row["shift_magnitude_um"] for row in mean_shift_rows
    ])
    registration_summary = {
        "adjacent_shift_definition": (
            "phase cross-correlation of standardised neighbouring chi and phi "
            "fields; reported mean of the two channel shift vectors"
        ),
        "median_shift_pixels": float(np.median(magnitudes_pixels)),
        "maximum_shift_pixels": float(np.max(magnitudes_pixels)),
        "median_shift_um": float(np.median(magnitudes_um)),
        "maximum_shift_um": float(np.max(magnitudes_um)),
        "centre_reference_transforms_yx": [
            None if transform is None else [float(value) for value in transform]
            for transform in transforms
        ],
        "native_valid_voxels": int(mask.sum()),
        "registered_valid_voxels": int(registered_mask.sum()),
        "native_global_best": native["global_best"],
        "registered_global_best": registered["global_best"],
    }
    with open(OUTPUT / "registration_sensitivity_summary.json", "w") as handle:
        json.dump(registration_summary, handle, indent=2)

    roi_candidates = select_roi_candidates(
        rgb, native["labels_flood"], native["best_labels"], mask, spacing_um
    )
    with open(OUTPUT / "roi_candidates_top5.json", "w") as handle:
        json.dump(roi_candidates, handle, indent=2)
    make_roi_previews(
        rgb, roi_candidates, native["labels_flood"], native["best_labels"],
        OUTPUT / "roi_candidates_top5.png",
    )

    selected_roi = roi_candidates[0]
    make_proposed_figure(
        native["rows"], native["global_best"], rgb, selected_roi,
        native["labels_flood"], native["best_labels"], spacing_um,
        OUTPUT / "proposed_main_figure",
    )
    make_supplementary_figure(
        native["rows"], native["global_best"],
        OUTPUT / "supplementary_radius_threshold_diagnostics.pdf",
    )
    np.savez_compressed(
        OUTPUT / "selected_segmentations_and_roi.npz",
        labels_flood=native["labels_flood"],
        labels_kam=native["best_labels"],
        markers_kam=native["best_markers"],
        mask=mask,
        roi_bounds_yx=np.array(selected_roi["array_bounds_yx"]),
        roi_layer=np.array(selected_roi["layer_index"]),
    )
    final_summary = {
        "baseline_reproduced": bool(baseline_passed),
        "notebook_baseline_displayed_15pct": next(
            row for row in baseline_rows if row["percentile"] == 15
        ),
        "notebook_radius_05_best_coarse_threshold": max(
            baseline_rows, key=lambda row: row["matched_overlap"]
        ),
        "native_global_best": native["global_best"],
        "registered_global_best": registered["global_best"],
        "selected_roi": selected_roi,
        "registration": registration_summary,
    }
    with open(OUTPUT / "analysis_summary.json", "w") as handle:
        json.dump(final_summary, handle, indent=2)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()

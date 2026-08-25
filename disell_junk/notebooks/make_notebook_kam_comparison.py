#!/usr/bin/env python
"""Reproduce the notebook's existing 30th-percentile KAM comparison figure."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts" / "paper_figures"))

import darling
import disell
from common import flood_footprint, watershed_feature_from_kam
from pf_io import VoxelSpacing, isotropic_physical_footprint

OUTPUT = Path(__file__).with_name("notebook_kam_comparison")
PERCENTILE = 30.0
LAYER = 1
Y_SLICE = slice(68, 92)
X_SLICE = slice(241, 316)


def kam_lowmem(field, mask, footprint):
    """The KAM calculation used in inspect_segmentations.ipynb."""
    f = np.where(mask[..., None], field, np.nan).astype(np.float32)
    offsets = np.argwhere(footprint) - np.array(footprint.shape) // 2
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
        for channel in range(f.shape[-1]):
            difference = f[neighbour + (channel,)] - f[centre + (channel,)]
            distance_sq += difference * difference
        magnitude = np.sqrt(distance_sq / f.shape[-1])
        finite = np.isfinite(magnitude)
        accumulator[centre] += np.where(finite, magnitude, 0).astype(np.float32)
        counts[centre] += finite
    result = np.divide(
        accumulator, counts, out=np.full(accumulator.shape, np.nan, np.float32),
        where=counts > 0,
    )
    result[~mask] = np.nan
    return result


def segment_kam(kam_map, mask, watershed_feature, min_voxels):
    values = kam_map[mask & np.isfinite(kam_map)]
    threshold = float(np.percentile(values, PERCENTILE))
    interior = mask & np.isfinite(kam_map) & (kam_map < threshold)
    components, _ = ndimage.label(
        interior, structure=ndimage.generate_binary_structure(mask.ndim, 1)
    )
    sizes = np.bincount(components.ravel())
    keep = np.flatnonzero(sizes >= min_voxels)
    keep = keep[keep > 0]
    markers = np.where(np.isin(components, keep), components, 0).astype(np.int32)
    labels = np.asarray(
        disell.region_grow_watershed(
            markers, mask, watershed_feature, connectivity=1
        ),
        dtype=np.int32,
    )
    return threshold, labels


def statistics(field, labels, mask, voxel_volume):
    selected = mask & (labels > 0)
    lab = labels[selected]
    values = field[selected].astype(np.float64)
    counts = np.bincount(lab)
    variance = np.zeros_like(counts, dtype=float)
    for channel in range(values.shape[-1]):
        means = (
            np.bincount(lab, weights=values[:, channel], minlength=counts.size)
            / np.maximum(counts, 1)
        )
        residual = values[:, channel] - means[lab]
        variance += (
            np.bincount(lab, weights=residual**2, minlength=counts.size)
            / np.maximum(counts, 1)
        )
    used = counts > 0
    used[0] = False
    volumes = counts[used] * voxel_volume
    diameters = (6 * volumes / np.pi) ** (1 / 3)
    return {
        "domains": int(used.sum()),
        "median_volume_um3": float(np.median(volumes)),
        "median_equivalent_diameter_um": float(np.median(diameters)),
        "median_angular_spread_deg": float(np.median(np.sqrt(variance[used]))),
    }


def main():
    OUTPUT.mkdir(exist_ok=True)
    config = json.load(open(
        PACKAGE_ROOT / "scripts" / "paper_figures" / "config_6_2pct.json"
    ))
    volume = disell.load_layer_volume(
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
    spacing_um = (0.500, 1.240, 0.400)
    voxel_volume = float(np.prod(spacing_um))
    min_voxels = max(1, int(round(2.5 / voxel_volume)))

    footprint = isotropic_physical_footprint(
        VoxelSpacing(*(value * 1000 for value in spacing_um)),
        radius_nm=500.0, ndim=3,
    )
    kam_map = kam_lowmem(field, mask, footprint)
    result, _ = disell.flood_fill_dfxm_two_stage(
        np.nan_to_num(field, nan=0).astype(np.float32),
        footprint=flood_footprint(
            {"segmentation": {"flood_footprint": "inplane8_plus_z"}}
        ),
        local_misorientation_threshold=0.02,
        global_threshold=0.10,
        footprint_tolerance=0.35,
        mask=mask.astype(np.uint8),
        max_iterations=5000,
        min_grain_size=min_voxels,
        stagnation_tolerance=2000,
        random_seed=42,
    )
    markers = np.asarray(result["segmentation"], dtype=np.int32)
    watershed_feature = watershed_feature_from_kam(kam_map, mask)
    flood = np.asarray(
        disell.region_grow_watershed(
            markers, mask, watershed_feature, connectivity=1
        ),
        dtype=np.int32,
    )
    threshold, kam = segment_kam(
        kam_map, mask, watershed_feature, min_voxels
    )

    metrics = {
        "source": "inspect_segmentations.ipynb",
        "feature_shape": list(field.shape),
        "valid_voxels": int(mask.sum()),
        "spacing_um_zyx": list(spacing_um),
        "kam_percentile": PERCENTILE,
        "kam_threshold_deg": threshold,
        "flood_fill": statistics(field, flood, mask, voxel_volume),
        "kam": statistics(field, kam, mask, voxel_volume),
        "matched_overlap": float(
            disell.matched_overlap(flood, kam, mask=mask)
        ),
        "variation_of_information_bits": float(
            disell.variation_of_information(flood, kam, mask=mask)
        ),
        "roi": {
            "layer_index_zero_based": LAYER,
            "layer_name": volume.layer_names[LAYER],
            "y_pixels": [Y_SLICE.start, Y_SLICE.stop],
            "x_pixels": [X_SLICE.start, X_SLICE.stop],
            "height_um": (Y_SLICE.stop - Y_SLICE.start) * spacing_um[1],
            "width_um": (X_SLICE.stop - X_SLICE.start) * spacing_um[2],
        },
    }
    with open(OUTPUT / "notebook_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    low = np.array([np.nanmin(field[..., c]) for c in range(2)])
    high = np.array([np.nanmax(field[..., c]) for c in range(2)])
    norm = np.stack(
        [low - 0.001 * (high - low), high + 0.001 * (high - low)], axis=1
    )
    rgb = darling.transforms.rgb(field[LAYER], norm=norm)[0]
    rgb[~mask[LAYER]] = 1
    crop = rgb[Y_SLICE, X_SLICE]

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.1, 1.62))
    grid = fig.add_gridspec(
        1, 3, width_ratios=(1, 1, 0.88), wspace=0.08
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax, labels, panel in zip(
        axes[:2], (flood, kam), ("(a)", "(b)")
    ):
        ax.imshow(crop, interpolation="nearest", aspect="equal")
        boundary = find_boundaries(
            labels[LAYER, Y_SLICE, X_SLICE], mode="inner"
        )
        ax.contour(
            boundary.astype(float), levels=[0.5], colors="black",
            linewidths=0.45,
        )
        ax.text(
            0.02, 0.97, panel, transform=ax.transAxes, va="top", ha="left",
            fontweight="bold", color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1,
                  "alpha": 0.75},
        )
        ax.set_axis_off()
    bar_um = 10
    bar_pixels = bar_um / spacing_um[2]
    y_bar = crop.shape[0] - 3.2
    x_bar = crop.shape[1] - bar_pixels - 3
    axes[0].plot(
        [x_bar, x_bar + bar_pixels], [y_bar, y_bar],
        color="white", lw=2.5, solid_capstyle="butt",
    )
    axes[0].plot(
        [x_bar, x_bar + bar_pixels], [y_bar, y_bar],
        color="black", lw=1.1, solid_capstyle="butt",
    )
    axes[0].text(
        x_bar + bar_pixels / 2, y_bar - 1.5, "10 µm",
        ha="center", va="bottom", color="black", fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5,
              "alpha": 0.75},
    )

    ax = axes[2]
    ax.set_axis_off()
    ax.text(0, 0.98, "(c)", va="top", fontweight="bold")
    rows = [
        ("Domains", f"{metrics['flood_fill']['domains']:,}",
         f"{metrics['kam']['domains']:,}"),
        ("Median volume (µm³)",
         f"{metrics['flood_fill']['median_volume_um3']:.1f}",
         f"{metrics['kam']['median_volume_um3']:.1f}"),
        ("Median diameter (µm)",
         f"{metrics['flood_fill']['median_equivalent_diameter_um']:.2f}",
         f"{metrics['kam']['median_equivalent_diameter_um']:.2f}"),
        ("Spread (°, median)",
         f"{metrics['flood_fill']['median_angular_spread_deg']:.4f}",
         f"{metrics['kam']['median_angular_spread_deg']:.4f}"),
    ]
    table = ax.table(
        cellText=[[label, flood_value, kam_value]
                  for label, flood_value, kam_value in rows],
        colLabels=("", "Flood fill", "KAM"),
        colWidths=(0.56, 0.24, 0.20),
        cellLoc="right", colLoc="right",
        bbox=(0, 0.27, 1, 0.61),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor("none")
        if column == 0:
            cell.get_text().set_ha("left")
        if row == 0:
            cell.get_text().set_fontweight("bold")
    ax.text(
        0, 0.07,
        f"Matched overlap: {metrics['matched_overlap']:.4f}\n"
        f"Variation of information: "
        f"{metrics['variation_of_information_bits']:.4f} bits",
        ha="left", va="bottom",
    )
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.025, top=0.985)
    fig.savefig(OUTPUT / "notebook_kam_comparison.pdf")
    fig.savefig(OUTPUT / "notebook_kam_comparison.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()

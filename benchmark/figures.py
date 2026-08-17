#!/usr/bin/env python3
"""The single benchmark figure.

Five panels sharing one background: the angular feature map on its own, then the
same map with the ground-truth, KAM and flood-fill boundaries drawn on it, then
the KAM field.  Boundaries are opaque black and exactly one output pixel wide.

Getting "exactly one output pixel" right needs two things:

1. the boundaries are found on the *display* raster, after nearest-neighbour
   upsampling, and written straight into it -- not drawn as line artists, which
   would be antialiased into grey; and
2. the axes are placed with explicit pixel arithmetic so one raster pixel maps
   to one device pixel.  ``constrained_layout`` and ``bbox_inches="tight"``
   both rescale the axes and would break that, so neither is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

RC_PARAMS: dict[str, Any] = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

INVALID_GREY = 0.86


def feature_rgb(field: np.ndarray, reference_deg: float) -> np.ndarray:
    """Two-channel angular feature field as an RGB mosaicity map.

    Hue is the direction of ``(chi, phi)`` about the field median, saturation
    its magnitude relative to ``reference_deg``.  Value stays high so opaque
    black boundaries read against every hue.
    """

    from matplotlib.colors import hsv_to_rgb

    centred = np.stack(
        [field[..., c] - np.median(field[..., c]) for c in range(field.shape[-1])],
        axis=-1,
    )
    magnitude = np.clip(
        np.linalg.norm(centred, axis=-1) / max(float(reference_deg), 1e-9), 0.0, 1.0
    )
    hue = (np.arctan2(centred[..., 1], centred[..., 0]) / (2.0 * np.pi)) % 1.0
    return hsv_to_rgb(
        np.stack([hue, 0.15 + 0.85 * magnitude, 1.0 - 0.25 * magnitude], axis=-1)
    )


def feature_reference_deg(field: np.ndarray, percentile: float = 96.0) -> float:
    """Robust magnitude used to normalise the colour map, in degrees."""

    centred = np.stack(
        [field[..., c] - np.median(field[..., c]) for c in range(field.shape[-1])],
        axis=-1,
    )
    return float(
        max(np.percentile(np.linalg.norm(centred, axis=-1), percentile), 1e-6)
    )


def upsample_nearest(array: np.ndarray, factors_yx: tuple[int, int]) -> np.ndarray:
    """Integer nearest-neighbour upsampling of the first two axes."""

    out = np.repeat(array, int(factors_yx[0]), axis=0)
    return np.repeat(out, int(factors_yx[1]), axis=1)


def single_pixel_boundaries(labels: np.ndarray) -> np.ndarray:
    """One-pixel-wide interfaces of a 2-D label image.

    A pixel is marked where the label changes towards its right or lower
    neighbour, so each interface yields exactly one line of pixels rather than
    the two a symmetric boundary operator would give.
    """

    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return boundary


def overlay(rgb: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Write fully opaque black into an RGB raster.  No alpha, no blending."""

    out = rgb.copy()
    out[boundary] = 0.0
    return out


def isotropic_display_factors(
    spacing_yx: Sequence[float], upsample: int
) -> tuple[int, int]:
    """Integer upsampling that makes the display pixels physically square."""

    spacing = np.asarray(spacing_yx, dtype=np.float64)
    ratio = spacing / spacing.min()
    return (
        max(int(round(upsample * ratio[0])), 1),
        max(int(round(upsample * ratio[1])), 1),
    )


def _scale_bar(ax, raster_shape, factors, spacing_yx, length_um: float) -> None:
    """Opaque black scale bar in raster-pixel coordinates."""

    from matplotlib.patches import Rectangle

    height, width = raster_shape
    px_per_um = factors[1] / float(spacing_yx[1])
    bar = length_um * px_per_um
    x0, y0 = 0.045 * width, 0.045 * height
    thickness = max(0.012 * height, 2.0)
    ax.add_patch(
        Rectangle((x0, y0), bar, thickness, facecolor="black",
                  edgecolor="white", linewidth=0.4, zorder=5)
    )
    ax.text(
        x0 + 0.5 * bar, y0 + 2.6 * thickness, f"{length_um:g} µm",
        ha="center", va="bottom", color="black", fontsize=6.0, zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )


def render_figure(
    out_path: Path,
    *,
    field: np.ndarray,
    kam: np.ndarray,
    truth_labels: np.ndarray,
    kam_labels: np.ndarray,
    flood_fill_labels: np.ndarray,
    spacing_um_zyx: Sequence[float],
    layer: int | None = None,
    upsample: int = 4,
    scale_bar_um: float = 10.0,
    colour_reference_deg: float = 0.75,
    dpi: int = 300,
) -> dict[str, Any]:
    """Render the benchmark figure and report exactly what it shows.

    ``colour_reference_deg`` is a **fixed physical** normalisation for the
    angular colour map, not a percentile of the data.  Two runs of this figure
    are therefore directly comparable: a colour means the same number of
    degrees whatever the phantom does.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spacing = tuple(float(v) for v in spacing_um_zyx)
    z = int(field.shape[0] // 2) if layer is None else int(layer)
    factors = isotropic_display_factors(spacing[1:], upsample)

    reference = float(colour_reference_deg)
    background = upsample_nearest(feature_rgb(field[z], reference), factors)

    panels: list[tuple[str, np.ndarray]] = [("Angular feature map", background)]
    for title, labels in (
        ("Ground truth", truth_labels),
        ("KAM threshold", kam_labels),
        ("Flood fill", flood_fill_labels),
    ):
        display = upsample_nearest(labels[z][..., None], factors)[..., 0]
        panels.append((title, overlay(background, single_pixel_boundaries(display))))

    kam_slice = kam[z]
    finite = kam_slice[np.isfinite(kam_slice)]
    high = float(np.percentile(finite, 99)) if finite.size else 1.0
    kam_display = upsample_nearest(kam_slice[..., None], factors)[..., 0]

    height, width = background.shape[:2]

    # Pixel-exact layout: every axes box is exactly ``width`` x ``height``
    # device pixels, so imshow maps one raster pixel to one output pixel.  The
    # margins are given in inches and converted here, so text keeps its room
    # whatever dpi is asked for.
    inch = float(dpi)
    left = right = int(round(0.05 * inch))
    gutter = int(round(0.07 * inch))
    bottom = int(round(0.05 * inch))
    top = int(round(0.24 * inch))  # title plus panel letter
    total_w = left + 5 * width + 4 * gutter + right
    total_h = bottom + height + top

    with plt.rc_context(RC_PARAMS):
        fig = plt.figure(figsize=(total_w / dpi, total_h / dpi), dpi=dpi)
        images = [(title, raster, None) for title, raster in panels]
        images.append(("KAM", kam_display, high))

        panel_boxes = []
        for index, (title, raster, vmax) in enumerate(images):
            x0 = left + index * (width + gutter)
            panel_boxes.append([x0, top, width, height])
            ax = fig.add_axes(
                [x0 / total_w, bottom / total_h, width / total_w, height / total_h]
            )
            if vmax is None:
                ax.imshow(raster, origin="lower", interpolation="nearest")
            else:
                colormap = plt.get_cmap("Greys").copy()
                colormap.set_bad(str(INVALID_GREY))
                artist = ax.imshow(
                    np.where(np.isfinite(raster), raster, np.nan),
                    origin="lower", interpolation="nearest",
                    cmap=colormap, vmin=0.0, vmax=vmax,
                )
                cax = ax.inset_axes([0.80, 0.12, 0.022, 0.30])
                bar = fig.colorbar(artist, cax=cax)
                bar.outline.set_linewidth(0.4)
                bar.ax.tick_params(width=0.4, length=1.8, pad=1.2, labelsize=5.5)
                bar.set_ticks([0.0, vmax])
                bar.ax.set_yticklabels(["0", f"{vmax:.2f}°"])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            _scale_bar(ax, (height, width), factors, spacing[1:], scale_bar_um)
            ax.set_title(title, pad=4.0, fontsize=7.0)
            ax.text(
                0.0, 1.02, "(" + "abcde"[index] + ")", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.0,
            )

        out_path = Path(out_path)
        # No bbox_inches: it would rescale the axes and break the 1:1 mapping.
        fig.savefig(out_path.with_suffix(".png"), dpi=dpi)
        fig.savefig(out_path.with_suffix(".pdf"), dpi=dpi)
        plt.close(fig)

    return {
        "layer": z,
        "panel_titles": [title for title, _, _ in images],
        "panel_boxes_ltwh": panel_boxes,
        "display_factors_yx": list(factors),
        "raster_shape_yx": [int(height), int(width)],
        "figure_pixels_wh": [int(total_w), int(total_h)],
        "colour_reference_deg": reference,
        "kam_colour_max_deg": high,
        "scale_bar_um": float(scale_bar_um),
        "field_of_view_um_yx": [
            field.shape[1] * spacing[1],
            field.shape[2] * spacing[2],
        ],
    }


def render_failure_modes_figure(
    out_path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    n_truth_cells: int,
    flood_fill: dict[str, Any] | None = None,
    dpi: int = 300,
) -> None:
    """Percolation and fragmentation of KAM markers against the threshold.

    Two panels, both against the KAM threshold percentile: how markers leak
    across weak walls, and how ground-truth cells end up split or unseeded.
    The flood-fill markers at the selected parameters are drawn as horizontal
    reference lines, since they do not depend on a KAM threshold.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percentile = np.array([row["kam_percentile"] for row in rows], dtype=float)
    series = {
        name: np.array([row[name] for row in rows], dtype=float)
        for name in (
            "marker_count",
            "percolating_markers",
            "max_cells_per_marker",
            "cells_split",
            "cells_unseeded",
        )
    }

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3), layout="constrained")

        panels = (
            (
                axes[0],
                (
                    ("marker_count", "markers", "#111111", "-", "o"),
                    ("percolating_markers", "percolating markers", "#111111", (0, (4, 1.6)), "s"),
                    ("max_cells_per_marker", "cells in worst marker", "#888888", (0, (1, 1.4)), "^"),
                ),
                "markers",
            ),
            (
                axes[1],
                (
                    ("cells_unseeded", "cells with no marker", "#111111", "-", "o"),
                    ("cells_split", "cells split by markers", "#888888", (0, (4, 1.6)), "s"),
                ),
                f"ground-truth cells (of {n_truth_cells})",
            ),
        )
        for ax, entries, ylabel in panels:
            for name, label, colour, style, marker in entries:
                ax.plot(
                    percentile, series[name], label=label, color=colour,
                    linestyle=style, marker=marker, linewidth=0.9,
                    markersize=2.4, markeredgewidth=0.0,
                )
                if flood_fill is not None and name in flood_fill:
                    ax.axhline(
                        float(flood_fill[name]), color=colour, linewidth=0.6,
                        linestyle=(0, (1, 2)), alpha=0.7,
                    )
            ax.set_xlabel("KAM threshold percentile")
            ax.set_ylabel(ylabel)
            ax.set_yscale("symlog", linthresh=1.0)
            # Nothing here is negative; symlog would otherwise spend half the
            # axis on the mirrored decades.
            ax.set_ylim(bottom=0.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, loc="best", fontsize=5.8)

        if flood_fill is not None:
            fig.text(
                0.5, -0.02,
                "dotted horizontal lines: flood fill at the selected parameters",
                ha="center", va="top", fontsize=5.8,
            )
        out_path = Path(out_path)
        fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def render_intradomain_cdf_figure(
    out_path: Path,
    spread_deg: np.ndarray,
    *,
    angular_range_deg: np.ndarray | None = None,
    target_spread_deg: tuple[float, float] = (0.025, 0.035),
    dpi: int = 300,
) -> dict[str, float]:
    """Cumulative distributions of the per-cell intradomain angular spread.

    A CDF rather than a histogram: the distribution is deliberately broad and
    right-skewed, and a CDF shows the median and the strongly varying minority
    without a bin-width choice.

    Two curves are drawn because two different statistics are in circulation and
    they must not be read off one another: the variance-based ``s_k`` of the
    manuscript's Eq. A2, and the peak-to-peak range-like span.  On this phantom
    the range is about 4.5x ``s_k``.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.sort(np.asarray(spread_deg, dtype=float))
    fraction = (np.arange(values.size) + 1) / values.size
    median = float(np.median(values))

    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(3.2, 2.4), layout="constrained")
        ax.axvspan(*target_spread_deg, color="0.90", linewidth=0.0, zorder=0)
        ax.step(
            values, fraction, where="post", color="#111111", linewidth=1.0,
            label=r"$s_k=\sqrt{\mathrm{Var}\chi+\mathrm{Var}\phi}$",
        )
        ax.axvline(median, color="#888888", linewidth=0.7, linestyle=(0, (3, 1.6)))
        ax.text(
            median, 0.03, f"  {median:.3f}°", color="#444444",
            fontsize=6.0, ha="left", va="bottom",
        )
        if angular_range_deg is not None:
            spans = np.sort(np.asarray(angular_range_deg, dtype=float))
            ax.step(
                spans, (np.arange(spans.size) + 1) / spans.size, where="post",
                color="#888888", linewidth=1.0, linestyle=(0, (4, 1.6)),
                label="peak-to-peak range",
            )
            ax.legend(frameon=False, loc="upper left", fontsize=6.0)
        ax.set_xscale("log")
        ax.set_xlabel("intradomain angular spread (deg)")
        ax.set_ylabel("cumulative fraction of cells")
        ax.set_ylim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        out_path = Path(out_path)
        fig.savefig(out_path.with_suffix(".png"), dpi=dpi)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)

    return {
        "n_cells": int(values.size),
        "median_deg": median,
        "range_median_deg": (
            float(np.median(angular_range_deg))
            if angular_range_deg is not None else float("nan")
        ),
        "p5_deg": float(np.percentile(values, 5)),
        "p25_deg": float(np.percentile(values, 25)),
        "p75_deg": float(np.percentile(values, 75)),
        "p95_deg": float(np.percentile(values, 95)),
        "p95_over_median": float(np.percentile(values, 95) / median),
        "fraction_above_twice_median": float(np.mean(values > 2.0 * median)),
    }


def render_facet_recovery_figure(
    out_path: Path,
    recoveries: dict[str, dict[str, np.ndarray]],
    *,
    n_bins: int = 14,
    dpi: int = 300,
) -> None:
    """Fraction of each ground-truth facet recovered, against its misorientation.

    This is what separates a parameter-search failure from an unrecoverable
    boundary: a method that misses strong facets is mis-configured, whereas
    every method must miss facets whose angular contrast is near zero.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "flood_fill": ("flood fill", "#111111", "-", "o"),
        "kam_threshold": ("KAM threshold", "#888888", (0, (4, 1.6)), "s"),
    }
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(3.6, 2.5), layout="constrained")
        edges = None
        for name, recovery in recoveries.items():
            misorientation = recovery["misorientation_deg"]
            recovered = recovery["cells_separated"]
            if misorientation.size == 0:
                continue
            if edges is None:
                positive = misorientation[misorientation > 0]
                edges = np.geomspace(
                    max(positive.min(), 1e-3), misorientation.max(), n_bins + 1
                )
            index = np.clip(np.digitize(misorientation, edges) - 1, 0, n_bins - 1)
            centres, medians = [], []
            for b in range(n_bins):
                selected = index == b
                if selected.sum() < 3:
                    continue
                centres.append(np.sqrt(edges[b] * edges[b + 1]))
                medians.append(float(np.mean(recovered[selected])))
            label, colour, style, marker = styles.get(
                name, (name, "#444444", "-", "o")
            )
            ax.plot(
                centres, medians, label=label, color=colour, linestyle=style,
                marker=marker, linewidth=0.9, markersize=2.6, markeredgewidth=0.0,
            )
        ax.set_xscale("log")
        ax.set_xlabel("ground-truth facet misorientation (deg)")
        ax.set_ylabel("fraction of cell pairs separated")
        ax.set_ylim(0.0, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=6.0)
        out_path = Path(out_path)
        fig.savefig(out_path.with_suffix(".png"), dpi=dpi)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


def read_panel(png_path: Path, box: Sequence[int]) -> np.ndarray:
    """Read one panel back out of a saved PNG, as RGB in array orientation.

    ``box`` is ``[left, top, width, height]`` in PNG pixel coordinates, as
    returned by :func:`render_figure`.  The result is flipped back to match the
    raster that was handed to ``imshow`` with ``origin="lower"``, so a test can
    compare the two directly.  If they agree pixel for pixel, the one-pixel
    boundaries in the raster are one pixel in the output.
    """

    import matplotlib.image as mpimg

    left, top, width, height = (int(v) for v in box)
    image = mpimg.imread(str(png_path))[..., :3]
    return image[top : top + height, left : left + width][::-1]

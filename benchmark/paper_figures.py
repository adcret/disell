#!/usr/bin/env python3
"""Manuscript figures for the strain-calibrated synthetic benchmark.

Two figures sharing one visual system, so they read as a pair:

``synthetic_benchmark.pdf``      main text.  What the phantom looks like, what
                                 each method does to it, and how the two
                                 compare across strain.
``synthetic_kam_interior.pdf``   supplementary.  Why KAM merges cells: the
                                 interior a cell must contain for the method to
                                 isolate it shrinks as cells do.

Both are drawn at one column of the IUCr two-column layout (86 mm).  The class
redefines ``figure`` and does not support ``figure*``, so a full-width figure
would break the numbering; the layout is built for the narrow column instead.

Conventions are applied through :func:`panel`, :func:`scale_bar` and the module
constants, so the two figures cannot drift apart:

* one accent per method, used nowhere else;
* outcome classes ordered good-to-bad and shaded light-to-dark, so the message
  survives greyscale printing;
* image panels carry no axes -- a scale bar instead, since the maps are
  qualitative;
* panel letters set in the same place, at the same size, in every panel.

Usage::

    python paper_figures.py --out <manuscript>/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COLUMN_IN = 3.386          # 86 mm, one IUCr column

FLOOD = "#1B5E8C"          # method accents
KAM = "#A63A2B"
RECOVERED = "#7FB2D4"      # outcome classes, light to dark
MERGED = "#C1503F"
NEITHER = "#E2E2E2"
INK = "#1A1A1A"
GRID = "#CCCCCC"


def style():
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.6,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 1.0,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
    })


def panel(ax, letter, title=None, *, image=False):
    """Label a panel identically wherever it appears."""

    text = f"({letter})" if title is None else f"({letter}) {title}"
    ax.set_title(text, fontsize=7.0, loc="left", pad=2.6, color=INK)
    if image:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor(INK)
    else:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.4, alpha=0.9)
        ax.set_axisbelow(True)


def scale_bar(ax, length_um, extent, label):
    span_x = extent[1] - extent[0]
    span_y = extent[2] - extent[3]
    x0 = extent[0] + 0.07 * span_x
    y0 = extent[3] + 0.90 * span_y
    ax.plot([x0, x0 + length_um], [y0, y0], color="white", linewidth=1.7,
            solid_capstyle="butt", zorder=5)
    ax.text(x0 + length_um / 2, y0 - 0.035 * span_y, label, color="white",
            fontsize=6.0, ha="center", va="bottom", zorder=5)


def outcome_map(truth, prediction, tau=0.9):
    """Each true cell classified as recovered, merged with a neighbour, or neither."""

    import strict_recovery as sr

    rows, cols, counts, tsz, psz = sr.contingency(truth, prediction)
    recovered = np.unique(rows[(counts / psz[cols] >= tau) &
                               (counts / tsz[rows] >= tau)])
    substantial = (counts >= sr.SUBSTANTIAL_VOXELS) & (
        counts >= sr.SUBSTANTIAL_FRACTION * tsz[rows])
    per_pred = np.bincount(cols[substantial], minlength=psz.size)
    merged = np.setdiff1d(np.unique(rows[substantial & (per_pred[cols] >= 2)]),
                          recovered)
    code = np.zeros(tsz.size, np.uint8)
    code[recovered] = 1
    code[merged] = 2
    code[0] = 0
    return code[truth], int(recovered.size), int(merged.size)


def outlines(labels):
    edge = np.zeros(labels.shape, bool)
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    rgba = np.zeros(edge.shape + (4,))
    rgba[edge] = (0, 0, 0, 1)
    return rgba


def best_settings():
    """Each arm's optimum, preferring the refined sweep when it has finished."""

    import capped_search as cs

    refined = HERE / "runs" / "refined" / "refined_optima.json"
    if refined.exists():
        rows = json.loads(refined.read_text())
        pick = {r["arm"]: r for r in rows if r["phantom"] == "primary"}
        if {"flood fill", "KAM threshold"} <= set(pick):
            f, k = pick["flood fill"], pick["KAM threshold"]
            return (
                {"footprint_radius_um": f["footprint_radius_um"],
                 "footprint_tolerance": f["footprint_tolerance"],
                 "local_threshold_deg": f["local_threshold_deg"],
                 "global_threshold_deg": -1.0,
                 "min_cell_size": int(f["min_cell_size"]),
                 "kam_radius_um": 1.2},
                {"percentile": k["percentile"], "kam_radius_um": k["kam_radius_um"],
                 "min_cell_size": int(k["min_cell_size"]),
                 "connectivity": int(k["connectivity"])},
                "refined")
    flood = dict(cs.DEFAULTS)
    flood.pop("merge_size_voxels"); flood.pop("merge_threshold_deg")
    flood["min_cell_size"] = 10
    flood["local_threshold_deg"] = cs.DEFAULT_LOCAL_THRESHOLD_DEG["flood fill"]
    return flood, {"percentile": 22.5, "kam_radius_um": 1.08,
                   "min_cell_size": 3, "connectivity": 2}, "coarse"


# -------------------------------------------------------------- main figure


def main_figure(out: Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    import phantom_lab as lab

    ph = lab.load_phantom()
    flood, kam, provenance = best_settings()
    ff = lab.segment(ph, **flood, seed=0)
    km = lab.segment_kam_baseline(ph, **kam)

    ff_map, ff_rec, _ = outcome_map(ph.labels, ff.labels)
    km_map, km_rec, _ = outcome_map(ph.labels, km.labels)
    n_true = int(np.unique(ph.labels[ph.labels > 0]).size)

    # Centre of the volume, fixed before looking at any result.
    z = ph.shape[0] // 2
    half = 36
    mid = ph.labels.shape[1] // 2
    ys = xs = slice(mid - half, mid + half)
    extent = (0, 2 * half * ph.spacing_um_zyx[2], 2 * half * ph.spacing_um_zyx[1], 0)
    edges = outlines(ph.labels[z, ys, xs])
    cmap = ListedColormap([NEITHER, RECOVERED, MERGED])

    fig = plt.figure(figsize=(COLUMN_IN, 3.62))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.34],
                            hspace=0.44, wspace=0.09,
                            left=0.145, right=0.985, top=0.945, bottom=0.10)

    for column, (letter, title, image) in enumerate((
            ("a", "true cells", None),
            ("b", "flood fill", ff_map),
            ("c", "KAM", km_map))):
        ax = fig.add_subplot(grid[0, column])
        if image is None:
            ax.imshow(ph.field[z, ys, xs, 0], cmap="twilight", extent=extent,
                      interpolation="nearest")
        else:
            ax.imshow(image[z, ys, xs], cmap=cmap, vmin=0, vmax=2,
                      extent=extent, interpolation="nearest")
        ax.imshow(edges, extent=extent, interpolation="nearest")
        panel(ax, letter, title, image=True)
        if image is not None:
            share = 100 * (ff_rec if column == 1 else km_rec) / n_true
            ax.text(0.5, -0.015, f"{share:.0f}% recovered", transform=ax.transAxes,
                    fontsize=6.4, ha="center", va="top", color=INK)
        if column == 0:
            scale_bar(ax, 10, extent, r"10 $\mu$m")

    fig.legend(handles=[
        Patch(facecolor=RECOVERED, edgecolor=INK, linewidth=0.4, label="recovered"),
        Patch(facecolor=MERGED, edgecolor=INK, linewidth=0.4, label="merged"),
        Patch(facecolor=NEITHER, edgecolor=INK, linewidth=0.4, label="neither"),
    ], loc="upper center", bbox_to_anchor=(0.565, 0.660), ncol=3, frameon=False,
        handlelength=0.85, handleheight=0.85, handletextpad=0.35,
        columnspacing=1.0, fontsize=6.6)

    # (d) recovery against strain
    ax = fig.add_subplot(grid[1, :])
    summary = json.loads((HERE / "analysis" / "replicates.json").read_text())["summary"]
    strains = np.array([2.4, 3.5, 4.6, 6.2])
    keys = ["2p4", "3p5", "4p6", "6p2"]

    for arm, colour, marker, label in (
            ("flood fill", FLOOD, "o", "flood fill"),
            ("KAM threshold", KAM, "s", "KAM thresholding")):
        mean = np.array([summary[arm][k]["mean"] for k in keys]) * 100
        sd = np.array([summary[arm][k]["sd"] for k in keys]) * 100
        ax.fill_between(strains, mean - sd, mean + sd, color=colour, alpha=0.16,
                        linewidth=0, zorder=1)
        ax.plot(strains[:3], mean[:3], color=colour, marker=marker, markersize=3.2,
                markeredgewidth=0.0, zorder=3, label=label)
        ax.plot(strains[2:], mean[2:], color=colour, linestyle=(0, (2.6, 1.8)),
                zorder=2)
        ax.plot(strains[3], mean[3], color=colour, marker=marker, markersize=3.8,
                markerfacecolor="white", markeredgewidth=0.9, zorder=3)

    ax.set_xlabel("tensile strain (%)")
    ax.set_ylabel("true cells recovered (%)")
    ax.set_xlim(2.05, 6.6)
    ax.set_ylim(-4, 92)
    ax.set_xticks(strains)
    ax.set_yticks([0, 25, 50, 75])
    panel(ax, "d", "recovery against strain")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", marker="o", linestyle="none",
                          markerfacecolor="white", markeredgewidth=0.9,
                          markersize=3.8, label="extrapolated"))
    ax.legend(handles=handles, loc="center right", frameon=False,
              handlelength=1.5, borderpad=0.1, labelspacing=0.35, fontsize=6.4)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"  {out.name}: flood fill {ff_rec}/{n_true} "
          f"({100*ff_rec/n_true:.0f}%), KAM {km_rec}/{n_true} "
          f"({100*km_rec/n_true:.0f}%)  [{provenance} settings]")
    return ff_rec, km_rec, n_true


# ------------------------------------------------------- supplementary figure


def interior_figure(out: Path):
    import matplotlib.pyplot as plt
    from scipy.ndimage import distance_transform_edt

    import phantom_lab as lab

    interior = json.loads(
        (HERE / "analysis" / "why_flood_fill.json").read_text())["interior"]
    radius = float(interior["kernel_radius_um"])

    fig = plt.figure(figsize=(COLUMN_IN, 3.35))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.20],
                            hspace=0.52, wspace=0.09,
                            left=0.155, right=0.985, top=0.945, bottom=0.115)

    # Both ends of the *measured* strain range: the least deformed volume has
    # the largest cells the material actually shows, and the most deformed the
    # smallest.  There is no large-cell comparison outside that range.
    for column, (loader, letter, title) in enumerate((
            (lambda: lab.load_strain_phantom("2p4"), "a", r"5.0 $\mu$m cells"),
            (lab.load_phantom, "b", r"4.2 $\mu$m cells"))):
        ph = loader()
        z = ph.shape[0] // 2
        half = 36
        mid = ph.labels.shape[1] // 2
        ys = xs = slice(mid - half, mid + half)
        volume = ph.labels
        far = np.zeros(volume.shape, np.float32)
        for cell in np.unique(volume[volume > 0]):
            mask = volume == cell
            box = np.argwhere(mask)
            lo = np.maximum(box.min(0) - 1, 0)
            hi = np.minimum(box.max(0) + 2, volume.shape)
            window = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            d = distance_transform_edt(window, sampling=ph.spacing_um_zyx)
            region = far[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            far[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = np.maximum(region, d)

        extent = (0, 2 * half * ph.spacing_um_zyx[2],
                  2 * half * ph.spacing_um_zyx[1], 0)
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(np.where(far[z, ys, xs] > radius, 0.0, 1.0), cmap="Greys",
                  vmin=-0.35, vmax=1.0, extent=extent, interpolation="nearest")
        ax.imshow(outlines(volume[z, ys, xs]), extent=extent,
                  interpolation="nearest")
        panel(ax, letter, title, image=True)
        if column == 0:
            scale_bar(ax, 10, extent, r"10 $\mu$m")

    ax = fig.add_subplot(grid[1, :])
    order = sorted(interior["phantoms"].items(),
                   key=lambda kv: -kv[1]["mean_cell_diameter_um"])
    diameter = np.array([e["mean_cell_diameter_um"] for _, e in order])
    fraction = np.array([100 * e["median_interior_fraction"] for _, e in order])
    none = np.array([100 * e["fraction_of_cells_with_no_interior"] for _, e in order])

    ax.plot(diameter, fraction, "o-", color=INK, markersize=3.0,
            markeredgewidth=0, label="interior per cell")
    ax.plot(diameter, none, "s--", color=KAM, markersize=3.0,
            markeredgewidth=0, label="cells with none")
    ax.set_xlabel(r"mean cell diameter ($\mu$m)")
    ax.set_ylabel("per cent")
    ax.set_xlim(8.5, 3.9)
    ax.set_ylim(-3, 47)
    ax.set_yticks([0, 15, 30, 45])
    panel(ax, "c", f"kernel radius {radius:g} " + r"$\mu$m")
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.5,
              borderpad=0.1, columnspacing=1.2, fontsize=6.4)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"  {out.name}: interior {fraction[0]:.0f}% -> {fraction[-1]:.0f}%, "
          f"cells with none {none[0]:.0f}% -> {none[-1]:.0f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    style()
    print("manuscript figures:")
    main_figure(args.out / "synthetic_benchmark.pdf")
    interior_figure(args.out / "synthetic_kam_interior.pdf")


if __name__ == "__main__":
    main()

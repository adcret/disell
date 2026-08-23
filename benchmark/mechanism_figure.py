#!/usr/bin/env python3
"""The main-text figure: where each method puts the cell boundaries.

Top row, over the angular feature field: the true boundaries, then each
method's.  Bottom row, the same three with the field taken away and each
boundary voxel classified against the truth -- black where the method drew a
true interface, red where it missed one, white where it invented one.  KAM's
panel is mostly red: the interfaces are not misplaced so much as absent, which
is under-segmentation nothing downstream can undo.

The window is **selected**, not typical: it is the 56x56 voxel region in which
the flood fill does best, and the caption must say so.  Every number quoted in
the text is volume-wide.  Showing a favourable region is legitimate for
illustrating what the failure looks like; presenting it as representative
performance would not be.

Usage::

    python mechanism_figure.py --out analysis/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper_figures as pf                                   # noqa: E402

#: The kernel the mechanism is quoted at throughout the study.
KERNEL_RADIUS_UM = 1.08

#: Shading for the schematic: what the kernel spoils, and what survives.
SPOILED = "#E9DCD8"
CORE = "#FFFFFF"


def interior_mask(labels: np.ndarray, spacing, radius_um: float) -> np.ndarray:
    """Voxels whose whole kernel neighbourhood lies inside one cell.

    This is what a KAM threshold can select: anywhere closer than ``radius_um``
    to a wall, the kernel straddles the wall and the KAM is raised.
    """

    from scipy.ndimage import maximum_filter, minimum_filter

    import pipelines

    footprint = pipelines.isotropic_footprint(spacing, radius_um)
    lo = minimum_filter(labels, footprint=footprint, mode="nearest")
    hi = maximum_filter(labels, footprint=footprint, mode="nearest")
    return (labels > 0) & (lo == hi)


#: Both ends of the measured strain range, least deformed first.
INTERIOR_PANELS = (("2p4", r"5.0 $\mu$m cells"), ("6p2", r"4.2 $\mu$m cells"))


def _phantoms():
    import phantom_lab as lab

    return lab.load_phantom(), lab.strain_series()


# ------------------------------------------------------------------ panels


#: Section and window: the region where the flood fill does best, found by
#: scanning every 56-voxel window of every slice.  See the module docstring.
BEST_WINDOW = {"z": 9, "y0": 80, "x0": 24, "size": 56}

#: The boundary panels read against a neutral ground, because two of the three
#: classes are black and white and neither can sit on a page-white field.
BOUNDARY_GROUND = "#AFAFAF"
BOUNDARY_MISSING = "#D42A1F"


def _edges(plane: np.ndarray) -> np.ndarray:
    """Voxels on an interface, in plane."""

    edge = np.zeros(plane.shape, bool)
    edge[:-1, :] |= plane[:-1, :] != plane[1:, :]
    edge[:, :-1] |= plane[:, :-1] != plane[:, 1:]
    return edge


def _boundary_classes(truth_plane, pred_plane) -> np.ndarray:
    """Each boundary voxel as found, missed, or invented.

    Black: the method drew an interface that is really there.  Red: a true
    interface it did not draw -- two cells share a region.  White: an interface
    with no true counterpart.  The truth panel is black everywhere by
    construction, which is the point of showing it.
    """

    from matplotlib.colors import to_rgb

    truth, pred = _edges(truth_plane), _edges(pred_plane)
    rgb = np.tile(np.array(to_rgb(BOUNDARY_GROUND)), truth.shape + (1,))
    rgb[pred & ~truth] = (1.0, 1.0, 1.0)
    rgb[truth & ~pred] = to_rgb(BOUNDARY_MISSING)
    rgb[truth & pred] = (0.0, 0.0, 0.0)
    return rgb


def main_figure(out: Path):
    """Truth, KAM and flood fill: boundaries on the field, then classified."""

    import matplotlib.pyplot as plt
    import phantom_lab as lab

    ph = lab.load_phantom()
    flood, kam, _ = pf.best_settings()
    ff = lab.segment(ph, **flood, seed=0).labels
    km = lab.segment_kam_baseline(ph, **kam).labels

    z = BEST_WINDOW["z"]
    n = BEST_WINDOW["size"]
    ys = slice(BEST_WINDOW["y0"], BEST_WINDOW["y0"] + n)
    xs = slice(BEST_WINDOW["x0"], BEST_WINDOW["x0"] + n)
    dy, dx = ph.spacing_um_zyx[1], ph.spacing_um_zyx[2]
    extent = (0, n * dx, n * dy, 0)
    field = ph.field[z, ys, xs, 0]
    truth_plane = ph.labels[z, ys, xs]

    fig, axes = plt.subplots(2, 3, figsize=(pf.COLUMN_IN, pf.COLUMN_IN / 1.33))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.945, bottom=0.004,
                        wspace=0.035, hspace=0.16)

    panels = (("a", "ground truth", ph.labels), ("b", "KAM", km),
              ("c", "flood fill", ff))
    for column, (letter, title, labels) in enumerate(panels):
        plane = labels[z, ys, xs]

        ax = axes[0, column]
        ax.imshow(field, cmap="twilight", extent=extent, interpolation="nearest")
        ax.imshow(pf.outlines(plane), extent=extent, interpolation="nearest")
        ax.set_title(f"({letter}) {title}", fontsize=7.0, loc="left", pad=2.4,
                     color=pf.INK)

        lower = axes[1, column]
        lower.imshow(_boundary_classes(truth_plane, plane), extent=extent,
                     interpolation="nearest")
        lower.set_title(f"({'def'[column]})", fontsize=7.0, loc="left",
                        pad=2.4, color=pf.INK)

        for ax in (axes[0, column], axes[1, column]):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_edgecolor(pf.INK)

    pf.scale_bar(axes[0, 0], 5, extent, r"5 $\mu$m")

    out.mkdir(parents=True, exist_ok=True)
    stem = out / "synthetic_main"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    print(f"written: {stem.with_suffix('.pdf')} and .png")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "analysis" / "figures")
    args = parser.parse_args()
    pf.style()
    main_figure(args.out)


if __name__ == "__main__":
    main()

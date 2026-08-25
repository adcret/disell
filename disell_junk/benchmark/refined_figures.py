#!/usr/bin/env python3
"""Figures for the three results the refined sweeps added.

These are the findings the coarse search could not state, drawn in the visual
system :mod:`paper_figures` establishes so the whole set reads as one:

``refined_findability.pdf``   why the comparison is not about peak accuracy.
                              Each arm's whole grid, measured against that
                              arm's own best, so the question is "how likely is
                              an arbitrary configuration to be good" rather than
                              "how good is the best one".
``refined_one_knob.pdf``      the one axis anybody sweeps, before and after the
                              merge.  After it the four strains want the same
                              value, which is what makes the parameter
                              transferable.
``refined_objections.pdf``    the two ways KAM's ceiling could have been an
                              artefact -- a percentile grid too coarse and
                              starting too high, and a kernel floor set by
                              flood-fill reasoning -- shown closed.

Usage::

    python refined_figures.py --out analysis/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper_figures as pf                                   # noqa: E402

REFINED = HERE / "runs" / "refined"

TRUTH = {"primary": 2534, "2p4": 1481, "3p5": 1622, "4p6": 1924, "6p2": 2525}
STRAINS = ("2p4", "3p5", "4p6", "6p2")
STRAIN_LABEL = {"2p4": "2.4 %", "3p5": "3.5 %", "4p6": "4.6 %", "6p2": "6.2 %"}

#: Arms, in the order they are reported.  ``merged`` is the arm as deployed.
ARMS = (("merged", "flood fill + merge", pf.FLOOD, "-"),
        ("flood", "flood fill", pf.FLOOD, "--"),
        ("kam", "KAM threshold", pf.KAM, "-"))

#: One global value serves every strain after the merge; section 5b.
GLOBAL_THRESHOLD_DEG = 0.010483
#: The floor the first search imposed, and this one removed.
RETIRED_RADIUS_FLOOR_UM = 0.9


def rows(stem: str, label: str) -> list[dict]:
    import capped_search as cs

    return [r for r in cs.read_rows(REFINED / f"{stem}_{label}.jsonl")
            if r.get("status") == "ok" and not cs._disqualified(r)]


def best_by(stem: str, label: str, axis: str) -> dict:
    """Best recovery *rate* at each value of one axis."""

    out: dict[float, float] = {}
    for row in rows(stem, label):
        value = row.get(axis)
        if value is None:
            continue
        rate = row["recovered_at_90"] / TRUTH[label]
        out[value] = max(out.get(value, 0.0), rate)
    return out


# ------------------------------------------------------------- findability


def figure_findability(out: Path):
    """How much of each arm's grid is near that arm's own best.

    A survival curve rather than a bar chart: the reader can pick their own
    definition of "good enough" off the x axis instead of taking the three
    thresholds a bar chart would have committed to.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(pf.COLUMN_IN, pf.COLUMN_IN * 0.78))
    grid = np.linspace(0.0, 1.0, 401)

    for stem, name, colour, dash in ARMS:
        fractions = []
        pooled_n = 0
        for label in STRAINS:
            values = np.array([r["recovered_at_90"] for r in rows(stem, label)],
                              dtype=float)
            if values.size == 0:
                continue
            relative = values / values.max()
            fractions.append(np.array([(relative >= g).mean() for g in grid]))
            pooled_n += values.size
        if not fractions:
            continue
        # Each phantom contributes equally: a phantom with a denser admissible
        # set must not dominate the pooled curve.
        curve = np.mean(fractions, axis=0)
        ax.plot(grid, curve, color=colour, linestyle=dash, linewidth=1.1,
                label=f"{name} ({pooled_n:,})", zorder=3)

    for mark in (0.90, 0.95):
        ax.axvline(mark, color=pf.GRID, linewidth=0.4, zorder=1)
        ax.text(mark - 0.008, 0.52, f"{mark:.0%}", fontsize=6.0, rotation=90,
                color=pf.INK, ha="right", va="center")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("recovery, as a fraction of that arm's own best")
    ax.set_ylabel("fraction of configurations at least this good")
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    # A single-panel figure carries no panel letter, but keeps the rest of the
    # house style: despined, y grid only, axis below the data.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=pf.GRID, linewidth=0.4, alpha=0.9)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    _save(fig, out / "refined_findability")
    return fig


# ----------------------------------------------------------------- one knob


def figure_one_knob(out: Path):
    """The tuned axis, before and after the merge.

    The point is the *alignment* of the four peaks in the lower panel, not
    their height, so both panels share a y axis and the per-strain optima are
    marked individually.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(pf.COLUMN_IN, pf.COLUMN_IN * 1.28),
                             sharex=True, sharey=True)
    shades = np.linspace(0.35, 1.0, len(STRAINS))

    for ax, stem, title in ((axes[0], "flood", "flood fill"),
                            (axes[1], "merged", "flood fill + merge")):
        for shade, label in zip(shades, STRAINS):
            curve = best_by(stem, label, "local_threshold_deg")
            if not curve:
                continue
            x = np.array(sorted(curve))
            y = np.array([curve[v] for v in x])
            colour = plt.matplotlib.colors.to_rgb(pf.FLOOD)
            colour = tuple(1 - shade * (1 - c) for c in colour)
            ax.plot(x, y, color=colour, linewidth=1.0, zorder=3,
                    label=STRAIN_LABEL[label])
            peak = x[int(np.argmax(y))]
            ax.plot([peak], [y.max()], marker="o", markersize=2.4,
                    color=colour, zorder=4)
        ax.axvline(GLOBAL_THRESHOLD_DEG, color=pf.INK, linewidth=0.5,
                   linestyle=":", zorder=2)
        ax.set_xscale("log")
        ax.set_ylabel("recovery rate")
        # Plain values: a threshold in degrees reads worse as 6x10^-3.
        ax.set_xticks([0.006, 0.010, 0.016, 0.024, 0.032])
        ax.set_xticklabels(["0.006", "0.010", "0.016", "0.024", "0.032"])
        ax.minorticks_off()
        ax.set_ylim(0.20, 0.86)

    axes[0].legend(frameon=False, loc="upper left", ncol=4, handlelength=1.2,
                   columnspacing=0.9, bbox_to_anchor=(0.0, 1.03))
    axes[1].set_xlabel("local threshold (deg)")
    # Above the curves, not across the axis label.
    axes[0].annotate("one global value", xy=(GLOBAL_THRESHOLD_DEG, 0.235),
                     xytext=(GLOBAL_THRESHOLD_DEG * 1.09, 0.235),
                     fontsize=6.0, color=pf.INK, ha="left", va="center")
    pf.panel(axes[0], "a", "before the merge")
    pf.panel(axes[1], "b", "after the merge")
    fig.tight_layout(pad=0.4)
    _save(fig, out / "refined_one_knob")
    return fig


# --------------------------------------------------------------- objections


def figure_objections(out: Path):
    """The two ways KAM's ceiling could have been an artefact of the search."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(pf.COLUMN_IN, pf.COLUMN_IN * 1.28))
    shades = np.linspace(0.35, 1.0, len(STRAINS))

    for shade, label in zip(shades, STRAINS):
        colour = plt.matplotlib.colors.to_rgb(pf.KAM)
        colour = tuple(1 - shade * (1 - c) for c in colour)
        for ax, axis in ((axes[0], "percentile"), (axes[1], "kam_radius_um")):
            curve = best_by("kam", label, axis)
            if not curve:
                continue
            x = np.array(sorted(curve))
            y = np.array([curve[v] for v in x])
            ax.plot(x, y, color=colour, linewidth=1.0, zorder=3,
                    label=STRAIN_LABEL[label])
            ax.plot([x[int(np.argmax(y))]], [y.max()], marker="o",
                    markersize=2.4, color=colour, zorder=4)

    # Both panels shade what the first search could not reach, so the two
    # objections are shown being closed in the same visual language.
    axes[0].axvspan(0.0, 20.0, color=pf.NEITHER, zorder=0)
    axes[0].set_xlim(4.0, 60.0)
    axes[0].text(12.0, 0.104, "excluded by the\nold grid floor", fontsize=6.0,
                 color=pf.INK, ha="center", va="top")
    axes[0].set_xlabel("KAM percentile")
    axes[0].set_ylabel("recovery rate")
    axes[0].legend(frameon=False, loc="lower right", ncol=2, handlelength=1.2,
                   columnspacing=1.0)

    axes[1].axvspan(0.0, RETIRED_RADIUS_FLOOR_UM, color=pf.NEITHER, zorder=0)
    axes[1].set_xlim(0.30, 2.55)
    axes[1].text(0.60, 0.104, "excluded by the\nretired floor", fontsize=6.0,
                 color=pf.INK, ha="center", va="top")
    axes[1].set_xlabel("KAM kernel radius (\u00b5m)")
    axes[1].set_ylabel("recovery rate")

    pf.panel(axes[0], "a", "the percentile grid was not clipping it")
    pf.panel(axes[1], "b", "the radius floor cost it nothing")
    fig.tight_layout(pad=0.4)
    _save(fig, out / "refined_objections")
    return fig


def figure_optimisation(out: Path):
    """Both arms searched to their own optimum, on every phantom.

    Two claims in one figure, in the order a sceptic asks them.  (a) flood fill
    is ahead on every volume, and not because KAM was left untuned -- each arm
    is at the best configuration its own exhaustive grid contains.  (b) that
    lead is not a property of the best configuration only: across the whole
    grid, an arbitrarily chosen flood-fill configuration is usually close to
    the best available, and an arbitrarily chosen KAM one almost never is.
    """

    import matplotlib.pyplot as plt

    # The merged arm only, and called simply "flood fill": the merge is part of
    # the method as deployed, so splitting it out here would offer the reader a
    # variant the paper does not recommend.
    shown = (("merged", "flood fill", pf.FLOOD, "-"),
             ("kam", "KAM threshold", pf.KAM, "-"))

    fig, axes = plt.subplots(2, 1, figsize=(pf.COLUMN_IN, pf.COLUMN_IN * 1.36))

    # ---- (a) each arm at its own optimum, every phantom -------------------
    ax = axes[0]
    # The strain series only.  The primary phantom is another realisation at
    # 6.2 %, so beside the 6.2 % bar it adds a category without adding a case.
    ticks = np.arange(len(STRAINS))
    width = 0.34
    for offset, (stem, name, colour, _) in zip((-width / 2, width / 2), shown):
        heights = [max((r["recovered_at_90"] for r in rows(stem, label)),
                       default=0) / TRUTH[label] for label in STRAINS]
        ax.bar(ticks + offset, heights, width, color=colour,
               edgecolor=pf.INK, linewidth=0.4, label=name, zorder=3)
    ax.set_xticks(ticks)
    ax.set_xticklabels([STRAIN_LABEL[k] for k in STRAINS])
    ax.set_ylabel("recovery rate")
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, loc="upper left", handlelength=1.0,
              handleheight=0.9, fontsize=6.0, borderpad=0.1)
    pf.panel(ax, "a")

    # ---- (b) and not only at the optimum ---------------------------------
    ax = axes[1]
    grid = np.linspace(0.0, 1.0, 401)
    for stem, name, colour, dash in shown:
        fractions = []
        for label in STRAINS:
            values = np.array([r["recovered_at_90"] for r in rows(stem, label)],
                              dtype=float)
            if values.size:
                relative = values / values.max()
                fractions.append(np.array([(relative >= g).mean() for g in grid]))
        if fractions:
            ax.plot(grid, np.mean(fractions, axis=0), color=colour,
                    linestyle=dash, linewidth=1.1, zorder=3)
    for mark in (0.90, 0.95):
        ax.axvline(mark, color=pf.GRID, linewidth=0.4, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("fraction of best")
    ax.set_ylabel("fraction of\nconfigurations", fontsize=6.6)
    pf.panel(ax, "b")

    fig.tight_layout(pad=0.4)
    _save(fig, out / "refined_optimisation")
    return fig


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    print(f"written: {stem.with_suffix('.pdf')} and .png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "analysis" / "figures")
    args = parser.parse_args()

    pf.style()
    figure_optimisation(args.out)
    figure_findability(args.out)
    figure_one_knob(args.out)
    figure_objections(args.out)


if __name__ == "__main__":
    main()

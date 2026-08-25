#!/usr/bin/env python
"""Manuscript figure for Section 4.2: 3D KAM-threshold response (6.2% volume).

Two stacked one-column panels from the validated sweep in
paper_outputs/6-2pct/kam_baseline/threshold_sweep.csv:
  (a) number of connected interior regions vs KAM threshold percentile;
  (b) largest connected region as a fraction of the valid volume.
The selected comparison threshold is marked identically in both panels.

Style follows the existing paper figures (Fig. 6): no title, no grid,
black data with a single restrained accent colour, embedded fonts.

Usage:
    python fig_kam3d_manuscript.py --config config_6_2pct.json
"""

from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
})
import matplotlib.pyplot as plt

from common import load_config, out_dir_for

ACCENT = "#b2182b"  # restrained dark red, matches the accent of Fig. 6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    kam_dir = out_dir_for(cfg, "kam_baseline", args.out_root)
    out = out_dir_for(cfg, "manuscript", args.out_root)

    rows = list(csv.DictReader(open(kam_dir / "threshold_sweep.csv")))
    pct = [float(r["percentile"]) for r in rows]
    n_comp = [int(r["n_components"]) for r in rows]
    largest = [float(r["largest_component_fraction"]) for r in rows]
    meta = json.load(open(kam_dir / "parameters_used.json"))
    pct_sel = float(meta["selected_percentile"])

    # compact source data next to the figure
    with open(out / "kam3d_threshold_response_source.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["percentile", "threshold_deg", "n_components",
                    "largest_component_fraction"])
        for r in rows:
            w.writerow([r["percentile"], f'{float(r["threshold"]):.4f}',
                        r["n_components"],
                        f'{float(r["largest_component_fraction"]):.4f}'])

    # IUCr one-column width: 8.8 cm
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(8.8 / 2.54, 8.0 / 2.54), sharex=True
    )

    ax_a.plot(pct, n_comp, "-o", color="k", ms=2.6, lw=0.9)
    ax_a.set_ylabel("interior regions")
    ax_a.set_ylim(0, 1050)

    ax_b.plot(pct, largest, "-o", color="k", ms=2.6, lw=0.9)
    ax_b.set_ylabel("largest region /\nvalid volume")
    ax_b.set_ylim(0, 1.0)
    ax_b.set_xlabel(r"KAM threshold (percentile)")
    ax_b.set_xlim(5, 95)

    for ax, lab in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.axvline(pct_sel, color=ACCENT, ls="--", lw=0.9)
        ax.text(0.02, 0.92, lab, transform=ax.transAxes, va="top",
                fontsize=8.5)

    fig.align_ylabels((ax_a, ax_b))
    fig.subplots_adjust(left=0.21, right=0.97, top=0.97, bottom=0.13,
                        hspace=0.12)
    fig.savefig(out / "kam3d_threshold_response_6-2pct.pdf")
    fig.savefig(out / "kam3d_threshold_response_6-2pct.png", dpi=600)
    plt.close(fig)
    print(f"figure written to {out} (selected percentile {pct_sel:g})")


if __name__ == "__main__":
    main()

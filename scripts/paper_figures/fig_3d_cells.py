#!/usr/bin/env python
"""Stage 7: publication figure — direct 3D segmentation of the 6.2% volume.

Panels:
  (a) registered chi feature map of the central layer with all final 3D cell
      boundaries overlaid and three selected neighbouring cells highlighted;
  (b) 3D rendering of those three cells at physical scale, colours matched to
      panel (a), with the displayed layer indicated.

Cell selection is fully reproducible and criteria-based:
  * the cells intersect the displayed slice with at least ``--min-slice-px``
    pixels;
  * their total volume is at least the median cell volume (no fragments);
  * they are mutually adjacent in 3D (touching under face connectivity);
  * among all qualifying triples the one with the largest minimum pairwise
    contact area is chosen (ties broken by ascending label ids).

Usage:
    python fig_3d_cells.py --config config_6_2pct.json
"""

from __future__ import annotations

import argparse
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from skimage.measure import marching_cubes
from skimage.segmentation import find_boundaries

from common import (
    load_config,
    load_registered_volume,
    out_dir_for,
    provenance,
    spacing_from_config,
    write_parameters_json,
)

CELL_COLORS = ["#d62728", "#1f77b4", "#ff7f0e"]


def contact_area(labels, a, b, struct):
    """Number of face-adjacent voxel pairs between labels a and b."""
    ma = labels == a
    grown = ndimage.binary_dilation(ma, structure=struct)
    return int((grown & (labels == b)).sum())


def select_cells(labels, z0, min_slice_px):
    """Deterministic selection of three neighbouring cells crossing slice z0."""
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    med_vol = np.median(counts)
    sizes = dict(zip(ids.tolist(), counts.tolist()))

    sl = labels[z0]
    slice_ids, slice_counts = np.unique(sl[sl > 0], return_counts=True)
    cands = [int(i) for i, c in zip(slice_ids, slice_counts)
             if c >= min_slice_px and sizes[int(i)] >= med_vol]
    if len(cands) < 3:
        raise RuntimeError(
            f"only {len(cands)} candidate cells on slice {z0}; "
            "relax --min-slice-px"
        )

    struct = ndimage.generate_binary_structure(3, 1)
    contact = {}
    for a, b in itertools.combinations(sorted(cands), 2):
        c = contact_area(labels, a, b, struct)
        if c > 0:
            contact[(a, b)] = c

    best, best_key = None, None
    for a, b, c in itertools.combinations(sorted(cands), 3):
        pairs = [(a, b), (a, c), (b, c)]
        if not all(p in contact for p in pairs):
            continue
        key = (min(contact[p] for p in pairs), sum(contact[p] for p in pairs),
               [-a, -b, -c])
        if best_key is None or key > best_key:
            best_key, best = key, (a, b, c)
    if best is None:
        raise RuntimeError("no mutually adjacent triple found on the slice")
    return list(best), {f"{a}-{b}": c for (a, b), c in contact.items()
                        if a in best and b in best}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--slice", type=int, default=None,
                    help="displayed layer (default: central layer)")
    ap.add_argument("--min-slice-px", type=int, default=150)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = out_dir_for(cfg, "fig_3d_cells", args.out_root)
    vol_dir = out_dir_for(cfg, "volume", args.out_root)
    seg_dir = out_dir_for(cfg, "segmentation_3d", args.out_root)

    field, mask, _, _ = load_registered_volume(vol_dir / "volume_registered.h5")
    import h5py
    with h5py.File(seg_dir / "labels_3d.h5", "r") as f:
        labels = f["labels"][...]

    Z = labels.shape[0]
    z0 = args.slice if args.slice is not None else Z // 2
    selected, contacts = select_cells(labels, z0, args.min_slice_px)
    print(f"selected cells {selected} on slice {z0}; contacts {contacts}")

    spacing = spacing_from_config(cfg)
    dz, dy, dx = [s / 1e3 for s in spacing.as_tuple_nm()]  # um
    ny, nx = labels.shape[1:]

    # bounding box of the selected cells (used by zoom inset and 3D view)
    union = np.isin(labels, selected)
    zz, yy, xx = np.where(union)
    pad = 2
    z_lo, z_hi = max(zz.min() - 1, 0), min(zz.max() + 2, labels.shape[0])
    y_lo, y_hi = max(yy.min() - pad, 0), min(yy.max() + pad + 1, ny)
    x_lo, x_hi = max(xx.min() - pad, 0), min(xx.max() + pad + 1, nx)
    sub = labels[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0],
                          width_ratios=[1.6, 1.0])

    def draw_slice(ax, x_range=None, y_range=None, fill_selected=True):
        extent = [0, nx * dx, ny * dy, 0]
        chi = field[z0, ..., 0]
        im = ax.imshow(chi, cmap="viridis", extent=extent,
                       interpolation="nearest")
        b = find_boundaries(labels[z0], mode="outer") & mask[z0]
        byy, bxx = np.where(b)
        ax.scatter(bxx * dx, byy * dy, s=0.05, c="k", linewidths=0)
        if fill_selected:
            overlay = np.zeros(labels[z0].shape + (4,))
            for color, lbl in zip(CELL_COLORS, selected):
                rgba = matplotlib.colors.to_rgba(color, alpha=0.55)
                overlay[labels[z0] == lbl] = rgba
            ax.imshow(overlay, extent=extent, interpolation="nearest")
        if x_range:
            ax.set_xlim(x_range)
            ax.set_ylim(y_range[1], y_range[0])
        ax.set_xlabel(r"x ($\mu m$)")
        ax.set_ylabel(r"y ($\mu m$)")
        return im

    # --- (a) full slice ------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    im = draw_slice(ax_a)
    # zoom box around the selected cells
    ax_a.add_patch(plt.Rectangle(
        (x_lo * dx, y_lo * dy), (x_hi - x_lo) * dx, (y_hi - y_lo) * dy,
        fill=False, edgecolor="w", lw=1.2))
    bar_um = 50.0
    ax_a.plot([8, 8 + bar_um], [ny * dy - 14, ny * dy - 14], "w-", lw=3)
    ax_a.text(8 + bar_um / 2, ny * dy - 22,
              rf"{bar_um:.0f} $\mu m$", color="w", ha="center", fontsize=9)
    ax_a.set_title(
        rf"(a) layer {z0}, $\chi$ with 3D cell boundaries", fontsize=11
    )
    cb = fig.colorbar(im, ax=ax_a, label=r"$\chi$ (deg)", shrink=0.85)

    # --- (b) zoom ------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    draw_slice(ax_b, x_range=(x_lo * dx, x_hi * dx),
               y_range=(y_lo * dy, y_hi * dy))
    ax_b.plot([x_hi * dx - 12, x_hi * dx - 2],
              [y_hi * dy - 3, y_hi * dy - 3], "w-", lw=3)
    ax_b.text(x_hi * dx - 7, y_hi * dy - 5.5, r"10 $\mu m$", color="w",
              ha="center", fontsize=9)
    ax_b.set_title("(b) selected neighbouring cells", fontsize=11)

    # --- (c) 3D rendering ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :], projection="3d")

    for color, lbl in zip(CELL_COLORS, selected):
        m = np.pad(sub == lbl, 1)
        verts, faces, _, _ = marching_cubes(
            m.astype(float), level=0.5, spacing=(dz, dy, dx)
        )
        verts -= (dz, dy, dx)  # undo the padding offset
        verts_xyz = verts[:, [2, 1, 0]]
        poly = Poly3DCollection(verts_xyz[faces], alpha=0.85)
        poly.set_facecolor(color)
        poly.set_edgecolor("none")
        ax3.add_collection3d(poly)

    x_rng = (x_hi - x_lo) * dx
    y_rng = (y_hi - y_lo) * dy
    z_rng = (z_hi - z_lo) * dz
    ax3.set_xlim(0, x_rng)
    ax3.set_ylim(0, y_rng)
    ax3.set_zlim(0, z_rng)
    ax3.set_box_aspect((x_rng, y_rng, z_rng))

    # indicate the displayed layer
    zs = (z0 - z_lo) * dz
    xxp, yyp = np.meshgrid([0, x_rng], [0, y_rng])
    ax3.plot_surface(xxp, yyp, np.full_like(xxp, zs), alpha=0.15, color="gray")

    ax3.set_xlabel(r"x ($\mu m$)", labelpad=8)
    ax3.set_ylabel(r"y ($\mu m$)", labelpad=8)
    ax3.set_zlabel(r"z ($\mu m$)", labelpad=2)
    ax3.set_zticks([0, 2.5, 5])
    ax3.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax3.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax3.tick_params(axis="both", labelsize=8)
    ax3.view_init(elev=28, azim=-60)
    ax3.set_title("(c) three neighbouring cells (physical scale)",
                  fontsize=11, pad=0)
    handles = [Line2D([0], [0], marker="s", color="none",
                      markerfacecolor=c, markersize=10,
                      label=f"cell {l}") for c, l in zip(CELL_COLORS, selected)]
    ax3.legend(handles=handles, loc="upper left", fontsize=9)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.03,
                        hspace=0.18, wspace=0.22)
    fig.savefig(out / "fig_3d_cells_6-2pct.png", dpi=300)
    fig.savefig(out / "fig_3d_cells_6-2pct.pdf")
    plt.close(fig)

    with open(out / "selection.json", "w") as f:
        json.dump({"slice": z0, "selected_labels": selected,
                   "contact_areas_voxel_faces": contacts,
                   "min_slice_px": args.min_slice_px,
                   "view": {"elev": 28, "azim": -60}}, f, indent=2)

    write_parameters_json(out, {
        "stage": "fig_3d_cells",
        "config": cfg,
        "slice": z0,
        "selected_labels": selected,
        "selection_rule": "mutually adjacent triple, slice intersection >= "
                          f"{args.min_slice_px} px, volume >= median, "
                          "maximising min pairwise contact",
        "provenance": provenance(),
    })
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()

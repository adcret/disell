#!/usr/bin/env python3
"""Generate the publication-quality 3D figure from a labelled volume.

Inputs (produced by ``segment_3d_cells.py``):

* ``labels_3d.h5``   — full 3D segmentation,
* ``kam_3d.h5``      — KAM scalar field (used as a fallback if no
  orientation HDF5 is available; here the volume is reloaded via
  ``darling`` for the actual orientation overlay).

This script does two things, both reproducibly from data:

1. **2D slice panel.** A representative Z slice of the orientation
   field (chi or phi, default chi) with the segmentation boundaries
   overlaid in black, a physical scale bar, and consistent colour
   limits. Output: ``figure_3d_cells.{pdf,png}``.

2. **Selected-cells 3D bundle.** A subset of the largest cells (by
   default the largest 3) is exported to a single VTI / VTK file with
   a ``selected_label`` field, ready to be loaded into ParaView and
   rendered. The script also writes
   ``README_paraview_rendering.md`` (in this folder) with step-by-step
   instructions to reproduce the 3D rendering exactly.

Why a separate ParaView step?
-----------------------------
Matplotlib 3D rendering is not publication-quality for cell volumes;
ParaView is the tool of choice in the DFXM community. Bundling the
data export step with deterministic instructions ensures the figure is
reproducible without anyone needing to write VTK code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _io import (  # noqa: E402
    OrientationVolume,
    VoxelSpacing,
    load_orientation_volume_from_darling,
    load_orientation_volume_from_h5,
    save_vti,
    write_parameters_json,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("data source")
    src.add_argument("--input-h5", type=str, required=True)
    src.add_argument("--scan-ids", nargs="+")
    src.add_argument("--orientation-method", choices=["mean", "peak"],
                     default="mean")
    src.add_argument("--roi", nargs=4, type=int, default=None,
                     metavar=("R0", "R1", "C0", "C1"))
    src.add_argument("--intensity-threshold", type=float, default=None)
    src.add_argument("--no-darling", action="store_true")
    src.add_argument("--field-dataset", type=str, default=None)
    src.add_argument("--mask-dataset", type=str, default=None)
    src.add_argument("--voxel-spacing-nm", nargs=3, type=float, required=True,
                     metavar=("DZ", "DY", "DX"))
    src.add_argument("--angle-unit", choices=["deg", "rad", "mrad"],
                     required=True)

    inp = p.add_argument_group("inputs from segment_3d_cells.py")
    inp.add_argument("--labels-h5", type=str, required=True,
                     help="labels_3d.h5 produced by segment_3d_cells.py")

    fig = p.add_argument_group("figure")
    fig.add_argument("--slice-z", type=int, default=None,
                     help="Z index of the slice rendered in the 2D panel. "
                          "Default: middle slice.")
    fig.add_argument("--orientation-channel", choices=["chi", "phi"],
                     default="chi",
                     help="Which orientation channel to colour the slice "
                          "with.")
    fig.add_argument("--scalebar-um", type=float, default=10.0,
                     help="Physical length of the scale bar in micrometres.")
    fig.add_argument("--n-cells-3d", type=int, default=3,
                     help="Number of largest cells exported for the 3D "
                          "rendering. The same cells are highlighted in the "
                          "2D slice.")
    fig.add_argument("--selected-cell-ids", nargs="+", type=int, default=None,
                     help="Optional explicit cell ids; overrides "
                          "--n-cells-3d.")

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", type=str, required=True)
    return p


def load_volume(args) -> OrientationVolume:
    spacing = VoxelSpacing(*args.voxel_spacing_nm)
    if args.no_darling:
        if args.field_dataset is None:
            raise SystemExit("--no-darling requires --field-dataset HDFPATH")
        return load_orientation_volume_from_h5(
            args.input_h5, args.field_dataset,
            mask_dataset=args.mask_dataset,
            spacing=spacing, angle_unit=args.angle_unit,
        )
    if not args.scan_ids:
        raise SystemExit("--scan-ids required without --no-darling")
    return load_orientation_volume_from_darling(
        args.input_h5, args.scan_ids,
        spacing=spacing, angle_unit=args.angle_unit,
        method=args.orientation_method,
        roi=tuple(args.roi) if args.roi else None,
        intensity_threshold=args.intensity_threshold,
    )


def load_labels(path: Path) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        return np.asarray(f["labels"][...]).astype(np.int32)


def select_cells(labels: np.ndarray, *, n: int,
                 explicit: Optional[List[int]] = None) -> List[int]:
    if explicit:
        return [int(x) for x in explicit]
    sizes = np.bincount(labels.ravel())
    sizes_no_bg = sizes.copy()
    if sizes_no_bg.size:
        sizes_no_bg[0] = 0
    n = min(n, max(sizes_no_bg.size - 1, 0))
    if n == 0:
        return []
    return np.argpartition(sizes_no_bg, -n)[-n:].tolist()


def render_slice_panel(
    out_dir: Path,
    volume: OrientationVolume,
    labels: np.ndarray,
    selected_cells: List[int],
    *,
    slice_z: int,
    orientation_channel: str,
    scalebar_um: float,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.segmentation import find_boundaries

    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    chan_idx = 0 if orientation_channel == "chi" else 1
    arr = volume.field[slice_z, ..., chan_idx]
    arr = np.where(volume.mask[slice_z], arr, np.nan)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise RuntimeError(
            f"slice z={slice_z} has no valid voxels; pick another slice"
        )
    vmin, vmax = np.percentile(finite, [1, 99])

    Y, X = arr.shape
    extent_um_x = X * volume.spacing.dx_nm * 1e-3
    extent_um_y = Y * volume.spacing.dy_nm * 1e-3

    fig, ax = plt.subplots(figsize=(6.0, 6.0 * extent_um_y / extent_um_x))
    im = ax.imshow(
        arr, origin="lower", cmap="viridis",
        vmin=vmin, vmax=vmax, interpolation="nearest",
        extent=[0.0, extent_um_x, 0.0, extent_um_y],
    )

    boundaries = find_boundaries(labels[slice_z], mode="thick")
    ax.contour(
        boundaries.astype(int), levels=[0.5], colors="black", linewidths=0.5,
        extent=[0.0, extent_um_x, 0.0, extent_um_y],
    )

    palette = ["tab:red", "tab:blue", "tab:green", "tab:orange",
               "tab:purple", "tab:brown", "tab:pink"]
    for k, cell_id in enumerate(selected_cells):
        cell_in_slice = labels[slice_z] == cell_id
        if not cell_in_slice.any():
            continue
        ys, xs = np.where(cell_in_slice)
        cy = float(ys.mean()) * volume.spacing.dy_nm * 1e-3
        cx = float(xs.mean()) * volume.spacing.dx_nm * 1e-3
        ax.plot(
            cx, cy, marker="o", markersize=8, mfc="none",
            mec=palette[k % len(palette)], mew=1.8,
            label=f"cell {cell_id}",
        )

    sb_y = 0.05 * extent_um_y
    sb_x = 0.05 * extent_um_x
    ax.add_patch(Rectangle(
        (sb_x, sb_y), scalebar_um, 0.012 * extent_um_y,
        facecolor="white", edgecolor="black", lw=0.6,
    ))
    ax.text(
        sb_x + 0.5 * scalebar_um, sb_y + 0.018 * extent_um_y,
        f"{scalebar_um:g} um", color="white", ha="center",
        va="bottom", fontsize=9, fontweight="bold",
    )

    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(
        f"{orientation_channel} [{volume.angle_unit}], z={slice_z} "
        f"({slice_z * volume.spacing.dz_nm * 1e-3:.2f} um)"
    )
    fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04,
        label=f"{orientation_channel} [{volume.angle_unit}]",
    )
    if selected_cells:
        ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"figure_3d_cells.{ext}", bbox_inches="tight")
    plt.close(fig)


def export_selected_cells_vti(
    out_dir: Path,
    labels: np.ndarray,
    selected_cells: List[int],
    *,
    spacing: VoxelSpacing,
) -> Path:
    """Write a VTI containing the full label volume and a ``selected_label`` mask."""
    selected = np.zeros_like(labels, dtype=np.int32)
    for k, cell_id in enumerate(selected_cells, start=1):
        selected[labels == cell_id] = k

    mask = (labels > 0).astype(np.uint8)

    out_path = out_dir / "selected_cells.vti"
    save_vti(
        out_path,
        {
            "labels_all": labels.astype(np.int32),
            "selected_label": selected,
            "mask": mask,
        },
        spacing=spacing,
    )
    return out_path


PARAVIEW_README = """\
# Reproducing the 3D rendering in ParaView

This file documents how to turn ``selected_cells.vti`` (produced by
``make_3d_paper_figure.py``) into the publication 3D rendering. The
recipe is deterministic and produces the same output every time.

## 1. Open the data

1. Launch ParaView (>= 5.11) from the command line so the working
   directory is the script output folder:

       paraview --data=selected_cells.vti

2. ParaView will open with the dataset selected. Click **Apply**.

## 2. Render only the selected cells

3. Open **Filters → Common → Threshold**. Set:
   - **Scalars**: ``selected_label``
   - **Lower threshold**: 1
   - **Upper threshold**: <number of selected cells>

   Click **Apply**.

4. Open **Filters → Common → Contour** on the threshold output if you
   prefer iso-surfaces of cell membership. Set:
   - **Contour by**: ``selected_label``
   - **Isosurfaces**: one value per selected cell (e.g. 0.5, 1.5,
     2.5, …) so each cell becomes its own surface.

   Click **Apply**.

5. With the contour selected, in the **Coloring** panel:
   - choose ``selected_label`` as the colour-by quantity,
   - open **Edit Color Map** and set a **categorical** colour map
     with one swatch per cell (red / blue / green by default).

## 3. Camera and styling

6. Set the camera to **Reset Camera Closest** (toolbar) and then
   manually rotate to the orientation the paper figure uses.
   ``View → Camera → Save Current Camera Configuration`` is useful
   to make this exactly reproducible.

7. Turn off all axis decorations except the orientation triad
   (**View → Axis Annotations → Orientation Triad**).

8. Add a 10 µm scale bar via **Annotation → Scale Bar**.

## 4. Export

9. **File → Save Screenshot** at 4× the screen resolution, choose PDF
   (vector) or PNG (raster).

The ``parameters_used.json`` written next to ``selected_cells.vti``
records the exact ``--selected-cell-ids`` and ``--slice-z`` values
used by the script, so different runs of ParaView always start from
the same data.

## Tip: identical view as the slice figure

The orientation triad in ParaView is right-handed with X right, Y up,
Z toward the viewer. The slice figure uses the same convention. Set
the camera to the +Z view and rotate by ~30° around X and Y for a
3/4 perspective.
"""


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume(args)
    labels = load_labels(Path(args.labels_h5))
    if labels.shape != volume.shape_zyx:
        raise SystemExit(
            f"Labels shape {labels.shape} differs from data shape "
            f"{volume.shape_zyx}; did you point at the right --labels-h5?"
        )

    selected = select_cells(
        labels, n=int(args.n_cells_3d),
        explicit=args.selected_cell_ids,
    )

    z = (
        int(args.slice_z)
        if args.slice_z is not None
        else volume.shape_zyx[0] // 2
    )
    render_slice_panel(
        out_dir, volume, labels, selected,
        slice_z=z,
        orientation_channel=args.orientation_channel,
        scalebar_um=float(args.scalebar_um),
    )
    vti_path = export_selected_cells_vti(
        out_dir, labels, selected, spacing=volume.spacing,
    )

    readme = out_dir / "README_paraview_rendering.md"
    readme.write_text(PARAVIEW_README)

    write_parameters_json(
        out_dir,
        {
            "script": "make_3d_paper_figure.py",
            "args": vars(args),
            "selected_cell_ids": list(map(int, selected)),
            "slice_z": z,
            "vti_path": str(vti_path.name),
            "voxel_spacing_nm_zyx": list(volume.spacing.as_tuple_nm()),
            "angle_unit": volume.angle_unit,
            "data_source": volume.source,
        },
    )

    print(
        f"[done] wrote figure_3d_cells.pdf/png and {vti_path.name} "
        f"to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

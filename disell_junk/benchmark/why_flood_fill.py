#!/usr/bin/env python3
"""Why flood fill beats KAM thresholding on 3D DFXM volumes.

This is the mechanistic half of the benchmark.  The search establishes *that*
one arm scores higher; this establishes *why*, which is what a reader needs
before accepting the method on experimental data where no ground truth exists.

Three claims are tested, each against the phantom's known ground truth.

1. **KAM needs a closed ridge; flood fill does not.**
   A KAM threshold defines a cell as a connected region of low local
   misorientation, so it can only close a cell where the surrounding wall
   raises KAM everywhere along it.  Where a wall is faint the ridge breaks and
   the two cells leak into one -- an unrecoverable fusion.  The flood fill
   instead grows from a seed under a running-mean constraint, so it can stop at
   an orientation step even where no ridge exists.
   *Test*: the phantom broadens a fraction of its wall area
   (``incomplete_wall_fraction``), recording the local width in
   ``wall_width_um``.  If the claim holds, the interfaces each method fuses
   should have systematically wider (fainter) walls, and the effect should be
   much stronger for the KAM arm.

2. **The advantage is specifically three-dimensional.**
   At 1.0 x 0.4 x 0.4 um spacing a small kernel spans only one z-layer, so a
   "3D" KAM is effectively a stack of 2D operators.  Flood fill propagates
   through z by connectivity rather than by kernel support, so it should lose
   far less when the volume is segmented slice by slice.
   *Test*: run both arms slice-wise and volumetrically and compare the gain.

3. **Flood fill's optimum is broader.**
   On experimental data there is no ground truth to tune against, so a method
   whose accuracy collapses away from an exactly-tuned point is not usable.
   *Test*: measure the volume of parameter space within a tolerance of each
   arm's own best score.  (Implemented in ``robustness.py``.)

Usage::

    python why_flood_fill.py walls      # claim 1
    python why_flood_fill.py dimension  # claim 2
    python why_flood_fill.py all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "analysis"
PHANTOM = HERE / "phantoms" / "primary_6p2_consistent.npz"

SUBSTANTIAL_FRACTION = 0.10
SUBSTANTIAL_VOXELS = 5


# ---------------------------------------------------------------- interfaces


def interface_table(labels: np.ndarray, wall_width_um: np.ndarray):
    """Every adjacent pair of true cells, with the wall width between them.

    Returns ``(pairs, area_voxels, mean_width_um)``.  A face between two
    differently-labelled voxels contributes the mean of the two widths, so the
    statistic describes the interface rather than either cell's interior.
    """

    pair_widths: dict[tuple[int, int], list] = {}
    for axis in range(labels.ndim):
        lower = np.moveaxis(labels, axis, 0)[:-1]
        upper = np.moveaxis(labels, axis, 0)[1:]
        w_low = np.moveaxis(wall_width_um, axis, 0)[:-1]
        w_high = np.moveaxis(wall_width_um, axis, 0)[1:]
        keep = (lower != upper) & (lower > 0) & (upper > 0)
        if not np.any(keep):
            continue
        a = np.minimum(lower[keep], upper[keep]).astype(np.int64)
        b = np.maximum(lower[keep], upper[keep]).astype(np.int64)
        widths = 0.5 * (w_low[keep] + w_high[keep])
        key = a * (int(labels.max()) + 1) + b
        order = np.argsort(key, kind="stable")
        key, widths = key[order], widths[order]
        edges = np.flatnonzero(np.diff(key)) + 1
        for chunk_key, chunk in zip(np.split(key, edges), np.split(widths, edges)):
            k = int(chunk_key[0])
            pair_widths.setdefault(k, []).append(chunk)

    span = int(labels.max()) + 1
    pairs, areas, means = [], [], []
    for k, chunks in pair_widths.items():
        values = np.concatenate(chunks)
        pairs.append((k // span, k % span))
        areas.append(values.size)
        means.append(float(values.mean()))
    return (np.asarray(pairs, dtype=np.int64),
            np.asarray(areas, dtype=np.int64),
            np.asarray(means, dtype=np.float64))


def fused_pairs(truth: np.ndarray, prediction: np.ndarray) -> set[tuple[int, int]]:
    """Pairs of true cells that share one predicted cell substantially."""

    import strict_recovery as sr

    rows, cols, counts, truth_sizes, pred_sizes = sr.contingency(truth, prediction)
    substantial = (counts >= SUBSTANTIAL_VOXELS) & (
        counts >= SUBSTANTIAL_FRACTION * truth_sizes[rows]
    )
    rows, cols = rows[substantial], cols[substantial]
    order = np.argsort(cols, kind="stable")
    rows, cols = rows[order], cols[order]
    out: set[tuple[int, int]] = set()
    edges = np.flatnonzero(np.diff(cols)) + 1
    for group in np.split(rows, edges):
        if group.size < 2:
            continue
        group = np.unique(group)
        for i in range(group.size):
            for j in range(i + 1, group.size):
                out.add((int(group[i]), int(group[j])))
    return out


def clustered_width_bootstrap(pairs: np.ndarray, widths: np.ndarray,
                              fused: np.ndarray, *, n_boot: int = 2000,
                              seed: int = 20260820) -> dict:
    """Bootstrap wall-width summaries by resampling true cells.

    Interface pairs sharing a cell are not independent observations.  This
    resamples cell ids, then includes interfaces incident to the sampled cells,
    so uncertainty is driven by cell-level clusters rather than treating every
    adjacency edge as an independent draw.  The original edge-level
    Mann--Whitney p-value remains in the report as a diagnostic, but these
    intervals are the computational uncertainty estimate.
    """

    pairs = np.asarray(pairs, dtype=np.int64)
    widths = np.asarray(widths, dtype=float)
    fused = np.asarray(fused, dtype=bool)
    cells = np.unique(pairs)
    if cells.size == 0 or not fused.any() or fused.all():
        return {"n_boot": 0, "n_cluster_cells": int(cells.size)}

    rng = np.random.default_rng(seed)
    ratios, differences = [], []
    for _ in range(int(n_boot)):
        sampled = rng.choice(cells, size=cells.size, replace=True)
        incident = np.isin(pairs[:, 0], sampled) | np.isin(pairs[:, 1], sampled)
        selected_fused = incident & fused
        selected_intact = incident & ~fused
        if not selected_fused.any() or not selected_intact.any():
            continue
        fused_median = float(np.median(widths[selected_fused]))
        intact_median = float(np.median(widths[selected_intact]))
        ratios.append(fused_median / intact_median)
        differences.append(fused_median - intact_median)

    def interval(values):
        if not values:
            return None
        return [float(v) for v in np.percentile(values, (2.5, 50.0, 97.5))]

    return {
        "n_boot": int(len(ratios)),
        "n_cluster_cells": int(cells.size),
        "n_cluster_cells_fused": int(np.unique(pairs[fused]).size),
        "n_cluster_cells_intact": int(np.unique(pairs[~fused]).size),
        "ratio_median_ci95": interval(ratios),
        "difference_median_um_ci95": interval(differences),
    }


def wall_analysis(phantom_path: Path, arms: dict) -> dict:
    """Claim 1: are fused interfaces the faint ones, and for which arm?"""

    with np.load(phantom_path) as data:
        truth = np.ascontiguousarray(data["labels"])
        wall = np.ascontiguousarray(data["wall_width_um"])

    pairs, areas, widths = interface_table(truth, wall)
    lookup = {tuple(p): i for i, p in enumerate(map(tuple, pairs))}
    print(f"  {len(pairs)} adjacent cell pairs; wall width "
          f"median {np.median(widths):.3f} um, p90 {np.percentile(widths, 90):.3f} um")

    report = {
        "n_interfaces": int(len(pairs)),
        "wall_width_um": {
            "median": float(np.median(widths)),
            "p10": float(np.percentile(widths, 10)),
            "p90": float(np.percentile(widths, 90)),
        },
        "arms": {},
    }
    for name, labels in arms.items():
        fused = fused_pairs(truth, labels)
        mask = np.zeros(len(pairs), dtype=bool)
        for pair in fused:
            index = lookup.get(pair)
            if index is not None:
                mask[index] = True
        if not mask.any():
            report["arms"][name] = {"fused_interfaces": 0}
            continue
        intact = ~mask
        # A wider wall is a fainter one: the phantom broadens incomplete wall
        # area, which is exactly what flattens the KAM ridge.
        report["arms"][name] = {
            "fused_interfaces": int(mask.sum()),
            "fused_fraction": float(mask.mean()),
            "median_width_fused_um": float(np.median(widths[mask])),
            "median_width_intact_um": float(np.median(widths[intact])),
            "width_ratio_fused_over_intact": float(
                np.median(widths[mask]) / np.median(widths[intact])
            ),
            "median_area_fused_voxels": float(np.median(areas[mask])),
            "median_area_intact_voxels": float(np.median(areas[intact])),
        }
        report["arms"][name]["cluster_bootstrap"] = clustered_width_bootstrap(
            pairs, widths, mask)
        # Rank-sum test: is the fused population drawn from wider walls?
        from scipy.stats import mannwhitneyu

        statistic, p_value = mannwhitneyu(
            widths[mask], widths[intact], alternative="greater"
        )
        report["arms"][name]["mannwhitney_u"] = float(statistic)
        report["arms"][name]["p_value_fused_walls_are_wider"] = float(p_value)
        entry = report["arms"][name]
        print(f"  {name:22s} fused {entry['fused_interfaces']:4d} interfaces; "
              f"wall width {entry['median_width_fused_um']:.3f} vs "
              f"{entry['median_width_intact_um']:.3f} um "
              f"(ratio {entry['width_ratio_fused_over_intact']:.2f}, "
              f"p={entry['p_value_fused_walls_are_wider']:.2e})")
    return report


# ------------------------------------------------------------ 2D versus 3D


def slicewise(function, phantom, **kwargs) -> np.ndarray:
    """Apply a 2D segmentation to every z slice and stack with unique labels."""

    planes = []
    offset = 0
    for z in range(phantom.labels.shape[0]):
        plane = function(phantom, z, **kwargs)
        plane = np.where(plane > 0, plane + offset, 0)
        offset = int(plane.max())
        planes.append(plane)
    return np.stack(planes).astype(np.int32)


def link_slices(planar: np.ndarray, overlap_fraction: float = 0.5) -> np.ndarray:
    """Join slice-wise labels through z where they overlap substantially.

    Segmenting each slice independently and then linking is what a practitioner
    does without a volumetric algorithm, so this -- not the unlinked stack -- is
    the honest 2D baseline.  An unlinked stack scores zero recovery by
    construction, because no single slice can hold 90 % of a cell that spans
    several layers; that is an artefact of the comparison, not a result.

    Two labels in adjacent slices are joined when their overlap covers at least
    ``overlap_fraction`` of the smaller one.
    """

    planar = np.ascontiguousarray(planar, dtype=np.int32)
    n = int(planar.max()) + 1
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sizes = np.bincount(planar.ravel(), minlength=n)
    for z in range(planar.shape[0] - 1):
        lower, upper = planar[z].ravel(), planar[z + 1].ravel()
        keep = (lower > 0) & (upper > 0)
        if not np.any(keep):
            continue
        key = lower[keep].astype(np.int64) * n + upper[keep].astype(np.int64)
        uniq, counts = np.unique(key, return_counts=True)
        a, b = np.divmod(uniq, n)
        smaller = np.minimum(sizes[a], sizes[b])
        for x, y in zip(a[counts >= overlap_fraction * smaller],
                        b[counts >= overlap_fraction * smaller]):
            rx, ry = find(int(x)), find(int(y))
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    roots[0] = 0
    survivors = np.unique(roots[1:])
    survivors = survivors[survivors > 0]
    remap = np.zeros(n, dtype=np.int32)
    remap[survivors] = np.arange(1, survivors.size + 1, dtype=np.int32)
    return remap[roots][planar]


def _flood_plane(phantom, z, **params):
    import disell
    import pipelines

    spacing2d = phantom.spacing_um_zyx[1:]
    footprint = pipelines.isotropic_footprint(
        spacing2d, params["footprint_radius_um"], ndim=2
    )
    field = np.ascontiguousarray(phantom.field[z], dtype=np.float32)
    mask = np.ascontiguousarray(phantom.mask[z].astype(np.uint8))
    result, _ = disell.flood_fill_dfxm_two_stage(
        field, footprint=np.ascontiguousarray(footprint, dtype=bool),
        local_misorientation_threshold=float(params["local_threshold_deg"]),
        global_threshold=float(params["global_threshold_deg"]),
        footprint_tolerance=float(params["footprint_tolerance"]),
        mask=mask, max_iterations=700_000,
        min_grain_size=max(int(params["min_cell_size"] //
                               max(phantom.labels.shape[0] // 4, 1)), 3),
        recycle_small_grains=False, stagnation_tolerance=2000, random_seed=0,
    )
    markers = np.asarray(result["segmentation"], dtype=np.int32)
    if markers.max() == 0:
        return markers
    kam = pipelines.masked_kam(
        phantom.field[z], phantom.mask[z],
        pipelines.isotropic_footprint(spacing2d, params["kam_radius_um"], ndim=2),
    )
    labels = disell.region_grow_watershed(
        markers, phantom.mask[z],
        pipelines.watershed_elevation(kam, phantom.mask[z]), connectivity=1,
    )
    return np.asarray(labels, dtype=np.int32)


def _kam_plane(phantom, z, **params):
    import pipelines

    spacing2d = phantom.spacing_um_zyx[1:]
    kam = pipelines.masked_kam(
        phantom.field[z], phantom.mask[z],
        pipelines.isotropic_footprint(spacing2d, params["kam_radius_um"], ndim=2),
    )
    try:
        labels, _, _ = pipelines.run_kam_threshold(
            kam, phantom.mask[z], percentile=float(params["percentile"]),
            min_cell_size=max(int(params["min_cell_size"] //
                                  max(phantom.labels.shape[0] // 4, 1)), 3),
            connectivity=int(params.get("connectivity", 1)),
            watershed_connectivity=1,
        )
    except RuntimeError:
        return np.zeros(phantom.labels.shape[1:], dtype=np.int32)
    return np.asarray(labels, dtype=np.int32)


def dimension_analysis(flood_params: dict, kam_params: dict,
                       phantom_path: Path | None = None) -> dict:
    """Claim 2: how much does each arm lose when denied the third dimension?

    ``phantom_path`` defaults to the primary phantom, so the reported figure is
    unchanged; it is a parameter only so the claim can be repeated on
    independent realisations of that phantom.
    """

    import phantom_lab as lab
    import strict_recovery as sr

    ph = lab.load_phantom(phantom_path) if phantom_path else lab.load_phantom()
    report = {"note": "2D+link segments each z slice independently and then "
                      "joins labels through z on >=50 % overlap, which is what "
                      "one does without a volumetric algorithm. 3D runs use the "
                      "volumetric algorithm at the same settings."}

    volumetric = {
        "flood fill": lab.segment(ph, **flood_params, seed=0).labels,
        "KAM threshold": lab.segment_kam_baseline(ph, **kam_params).labels,
    }
    raw = {
        "flood fill": slicewise(_flood_plane, ph, **flood_params),
        "KAM threshold": slicewise(_kam_plane, ph,
                                   **{**kam_params, "connectivity": 1}),
    }
    planar = {arm: link_slices(labels) for arm, labels in raw.items()}
    for arm in volumetric:
        entry = {"unlinked_2d_cells": int(raw[arm].max())}
        for mode, labels in (("3D", volumetric[arm]), ("2D+link", planar[arm])):
            strict = sr.strict_recovery(ph.labels, labels, tau=0.9)
            entry[mode] = {
                "n_cells_pred": strict["n_cells_pred"],
                "recovered_at_90": strict["recovered_at_tau"],
                "recovery_rate": strict["recovery_rate_at_tau"],
                "contamination": strict["contamination"],
                "fused_true_cells": strict["fused_true_cells"],
            }
        gain = entry["3D"]["recovered_at_90"] - entry["2D+link"]["recovered_at_90"]
        entry["gain_from_3d_cells"] = int(gain)
        entry["gain_from_3d_relative"] = float(
            gain / max(entry["2D+link"]["recovered_at_90"], 1)
        )
        report[arm] = entry
        print(f"  {arm:16s} 2D+link {entry['2D+link']['recovered_at_90']:4d} -> "
              f"3D {entry['3D']['recovered_at_90']:4d} recovered "
              f"(gain {gain:+d}, {entry['gain_from_3d_relative']:+.1%})")
    return report


# --------------------------------------------- the KAM interior reservoir


def interior_fraction(labels: np.ndarray, spacing, radius_um: float) -> np.ndarray:
    """Per-cell fraction of voxels further than ``radius_um`` from the cell wall.

    A KAM kernel of radius r raises KAM within r of any boundary, so the only
    place a low-KAM component can form is the inner core of a cell.  This
    measures how much of that core survives.  Where it goes to zero, no closed
    low-KAM region exists and adjacent cells necessarily share one component --
    which is under-segmentation that nothing downstream can undo.

    Returned per cell id (index 0 is background and is NaN).
    """

    from scipy.ndimage import distance_transform_edt

    labels = np.asarray(labels)
    n = int(labels.max()) + 1
    # Distance to the nearest voxel of a *different* label, in physical units.
    # Computing it once for the whole volume is wrong at cell interfaces, so
    # the transform is taken per cell against its own complement.
    interior = np.zeros(n)
    total = np.bincount(labels.ravel(), minlength=n).astype(float)
    # One pass: distance within each cell to the cell's own boundary.
    for cell in range(1, n):
        if total[cell] == 0:
            continue
        mask = labels == cell
        if not mask.any():
            continue
        box = np.argwhere(mask)
        lo = np.maximum(box.min(0) - 1, 0)
        hi = np.minimum(box.max(0) + 2, labels.shape)
        window = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        distance = distance_transform_edt(window, sampling=spacing)
        interior[cell] = float(np.count_nonzero(distance > radius_um))
    out = np.full(n, np.nan)
    valid = total > 0
    out[valid] = interior[valid] / total[valid]
    out[0] = np.nan
    return out


def interior_analysis(phantoms: dict, radius_um: float = 1.2) -> dict:
    """Claim 3: the low-KAM interior collapses as cells shrink.

    ``phantoms`` maps a label to a loaded phantom.  The same kernel radius is
    applied to all of them, so the only thing changing is the cell size.
    """

    report = {"kernel_radius_um": float(radius_um), "phantoms": {}}
    for name, phantom in phantoms.items():
        voxel = float(np.prod(phantom.spacing_um_zyx))
        volumes = np.bincount(phantom.labels.ravel())[1:] * voxel
        volumes = volumes[volumes > 0]
        diameter = 2.0 * (3.0 * volumes.mean() / (4.0 * np.pi)) ** (1.0 / 3.0)
        fraction = interior_fraction(phantom.labels, phantom.spacing_um_zyx, radius_um)
        fraction = fraction[np.isfinite(fraction)]
        entry = {
            "mean_cell_diameter_um": float(diameter),
            "kernel_over_diameter": float(radius_um / diameter),
            "median_interior_fraction": float(np.median(fraction)),
            "mean_interior_fraction": float(np.mean(fraction)),
            "cells_with_no_interior": int(np.count_nonzero(fraction <= 0)),
            "fraction_of_cells_with_no_interior": float(np.mean(fraction <= 0)),
            "n_cells": int(fraction.size),
        }
        report["phantoms"][name] = entry
        print(f"  {name:22s} d {diameter:5.2f} um  r/d {entry['kernel_over_diameter']:.2f}  "
              f"median interior {entry['median_interior_fraction']:6.3f}  "
              f"cells with none {entry['fraction_of_cells_with_no_interior']:6.1%}")
    return report


def best_parameters(store: Path = HERE / "runs/primary"):
    """Read each arm's winning configuration from the finished search."""

    import capped_search as cs

    rows = [r for r in cs.read_rows(store / "final.jsonl") if r.get("status") == "ok"]
    if not rows:
        return None, None
    winners: dict[str, dict] = {}
    for row in rows:
        arm = row.get("arm") or "flood fill"
        if arm not in winners or cs.strict_recovery_key(row) < cs.strict_recovery_key(winners[arm]):
            winners[arm] = row
    flood = winners.get("flood fill + merge") or winners.get("flood fill")
    kam = winners.get("KAM threshold")
    flood_params = None
    if flood:
        flood_params = {k: flood[k] for k in
                        ("local_threshold_deg", "global_threshold_deg",
                         "footprint_tolerance", "footprint_radius_um",
                         "min_cell_size", "kam_radius_um") if k in flood}
    kam_params = None
    if kam:
        kam_params = {k: kam[k] for k in
                      ("percentile", "kam_radius_um", "min_cell_size",
                       "connectivity") if k in kam}
    return flood_params, kam_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["walls", "dimension", "interior", "all"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import phantom_lab as lab

    flood_params, kam_params = best_parameters()
    if flood_params is None:
        flood_params = {k: v for k, v in lab.CAPPED_START.items()
                        if not k.startswith("merge")}
        print("! no finished search found; using the provisional capped start")
    if kam_params is None:
        kam_params = dict(lab.KAM_BASELINE_PARAMS)
    print(f"flood fill : {flood_params}")
    print(f"KAM        : {kam_params}\n")

    report = {"flood_parameters": flood_params, "kam_parameters": kam_params}

    if args.action in ("walls", "all"):
        print("=== claim 1: fusion happens where walls are faint ===")
        ph = lab.load_phantom()
        arms = {
            "flood fill": lab.segment(ph, **flood_params, seed=0).labels,
            "KAM threshold": lab.segment_kam_baseline(ph, **kam_params).labels,
        }
        report["walls"] = wall_analysis(PHANTOM, arms)

    if args.action in ("interior", "all"):
        print("\n=== claim 3: the low-KAM interior collapses as cells shrink ===")
        # Only phantoms calibrated to the measured cell-size and misorientation
        # trend.  The claim is about the microstructure the method will meet,
        # so a volume built to any other cell size cannot support it.
        phantoms = {"primary (4.2 um cells)": lab.load_phantom()}
        for key, phantom in lab.strain_series().items():
            phantoms[f"strain {key}"] = phantom
        radius = float(kam_params.get("kam_radius_um", 1.2))
        report["interior"] = interior_analysis(phantoms, radius)

    if args.action in ("dimension", "all"):
        print("\n=== claim 2: how much of the advantage is three-dimensional ===")
        report["dimension"] = dimension_analysis(flood_params, kam_params)

    (OUT / "why_flood_fill.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(f"\nwritten: {OUT/'why_flood_fill.json'}")


if __name__ == "__main__":
    main()

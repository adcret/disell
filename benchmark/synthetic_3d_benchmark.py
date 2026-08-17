#!/usr/bin/env python3
r"""Systematic flood-fill parameter search on one fixed 3D phantom.

Builds the deterministic phantom of :mod:`phantom`, then searches the
``adam_fix`` ``disell`` flood-fill identification over

* local threshold,
* global threshold,
* footprint tolerance,
* physical footprint radius, and
* minimum cell size,

each followed by the same KAM-guided watershed refinement.  Every point is
scored against the known 3D ground truth by adjusted Rand index, with variation
of information, physical boundary ASSD and cell-count error as secondary
diagnostics.

The search runs in two stages.  Stage 1 evaluates the whole grid with one
flood-fill seed order.  Stage 2 repeats the top finalists over several seed
orders and selects the highest **mean** ARI, because the multi-seed flood fill
is order-dependent and one lucky seed order is not a result.

A KAM-threshold baseline is swept over its threshold percentile and reported at
its own best ARI, so it is not handicapped by an arbitrary choice.

Outputs
-------
``search.csv``                  complete stage-1 and stage-2 results
``selected_parameters.json``    selected parameters and how they were selected
``metrics.json``                phantom properties and the final scores
``benchmark_figure.{png,pdf}``  the main figure
``intradomain_spread_cdf.*``    CDF of the per-cell intradomain angular spread
``phantom_and_labels.npz``      arrays behind the figure
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):  # allow "python benchmark/synthetic_3d_benchmark.py"
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_metrics
import figures
import phantom as phantom_module
import pipelines
from phantom import PhantomConfig, generate_phantom


@dataclass(frozen=True)
class SearchConfig:
    """The parameter grid and how the finalists are repeated."""

    # Only the genuinely active parameters are searched.  A profile through the
    # previous optimum showed the global threshold inactive above ~0.17 deg and
    # the footprint-count test inactive below ~0.04, so both are held at a
    # plateau value and re-verified afterwards rather than being re-optimised.
    local_thresholds_deg: tuple[float, ...] = (0.004, 0.0075, 0.012, 0.02)
    footprint_tolerances: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    footprint_radii_um: tuple[float, ...] = (0.9, 1.3, 1.8)
    min_cell_sizes: tuple[int, ...] = (75, 125, 200, 320)
    #: KAM / watershed footprint radius, now searched rather than assumed.
    kam_radii_um: tuple[float, ...] = (0.6, 1.0, 1.6)

    #: Held fixed; ``inactive_parameter_profile`` re-checks it at the optimum.
    #: The footprint tolerance used to be held here too, but after the
    #: intradomain field was restructured it is worth 0.04 ARI, so it is now
    #: searched rather than assumed.
    global_threshold_deg: float = 0.30

    #: KAM-threshold baseline: swept over both its radius (``kam_radii_um``)
    #: and its threshold percentile.
    kam_percentiles: tuple[float, ...] = (5, 10, 15, 20, 25, 30, 40, 50)
    #: Finer sweep used only for the percolation / fragmentation diagnostic.
    failure_mode_percentiles: tuple[float, ...] = (
        2, 5, 8, 10, 13, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80,
    )

    #: Refinement stops when the best point is interior on every axis and every
    #: immediate neighbouring value has been evaluated, or after this many
    #: rounds.
    max_refinement_rounds: int = 14
    #: An axis is considered fine enough when adjacent values differ by less
    #: than this ratio (continuous axes) or this many elements (min cell size).
    refine_ratio: float = 1.30
    refine_min_size_step: int = 25
    #: Fixed geometric step used when an axis has to grow past its current end.
    extension_factor: float = 2.0

    #: How many candidates are repeated, and over which seed orders.
    n_finalists: int = 15
    repeat_seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    #: Mean-ARI gap within which candidates are treated as tied and separated
    #: by VI, then boundary ASSD, then seed-to-seed ARI spread.
    selection_tolerance: float = 0.002

    #: Acceptance criterion on the phantom itself.  If flood fill at its own
    #: optimum cannot even seed this fraction of the ground-truth cells, the
    #: phantom is pathological rather than merely challenging and the result
    #: should not be reported as a method comparison.
    max_unseeded_fraction: float = 0.10

    max_seed_attempts: int = 8000
    stagnation_tolerance: int = 2000
    watershed_connectivity: int = 1


#: The searched axes.  The global threshold and the footprint tolerance are not
#: here: they are inactive at the optimum and are held fixed.
PARAMETER_NAMES = (
    "local_threshold_deg",
    "footprint_tolerance",
    "footprint_radius_um",
    "min_cell_size",
    "kam_radius_um",
)

#: Hard limits and refinement style per searched axis.
AXIS_SPECS: dict[str, dict[str, Any]] = {
    "local_threshold_deg": {"kind": "continuous", "lo": 1e-4, "hi": 1.0},
    "footprint_tolerance": {"kind": "continuous", "lo": 0.01, "hi": 1.0},
    "footprint_radius_um": {"kind": "continuous", "lo": 0.4, "hi": 4.0},
    "min_cell_size": {"kind": "integer", "lo": 5, "hi": 20000},
    "kam_radius_um": {"kind": "continuous", "lo": 0.3, "hi": 4.0},
}

REPEATED_METRICS = (
    "ari",
    "vi_total_bits",
    "vi_split_bits",
    "vi_merge_bits",
    "boundary_assd_um",
    "cell_count_error",
    "n_cells_pred",
    "percolating_marker_volume_fraction",
    "cells_split",
    "cells_unseeded",
)


def grid_points(config: SearchConfig) -> list[dict[str, float]]:
    """Every combination of the five searched flood-fill parameters."""

    return [
        dict(zip(PARAMETER_NAMES, values))
        for values in itertools.product(
            config.local_thresholds_deg,
            config.footprint_tolerances,
            config.footprint_radii_um,
            config.min_cell_sizes,
            config.kam_radii_um,
        )
    ]


class KamCache:
    """KAM fields keyed by footprint radius, computed on first use.

    The radius is a searched parameter now, so the field cannot be computed
    once up front; caching keeps the refinement loop from recomputing it for
    every point.
    """

    def __init__(self, phantom):
        self._phantom = phantom
        self._cache: dict[float, np.ndarray] = {}

    def __getitem__(self, radius_um: float) -> np.ndarray:
        key = round(float(radius_um), 6)
        if key not in self._cache:
            self._cache[key] = pipelines.masked_kam(
                self._phantom.segmentation_field,
                self._phantom.mask,
                pipelines.isotropic_footprint(self._phantom.spacing_um_zyx, key),
            )
        return self._cache[key]


def run_flood_fill_point(
    phantom,
    kam: KamCache,
    point: dict[str, float],
    config: SearchConfig,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One flood-fill identification plus watershed refinement.

    Returns the refined partition and the identification markers; the markers
    are what the percolation and fragmentation diagnostics act on.
    """

    footprint = pipelines.isotropic_footprint(
        phantom.spacing_um_zyx, point["footprint_radius_um"]
    )
    labels, markers = pipelines.run_flood_fill(
        phantom.segmentation_field,
        phantom.mask,
        kam[point["kam_radius_um"]],
        footprint,
        local_threshold_deg=point["local_threshold_deg"],
        global_threshold_deg=config.global_threshold_deg,
        footprint_tolerance=point["footprint_tolerance"],
        min_cell_size=int(point["min_cell_size"]),
        max_seed_attempts=config.max_seed_attempts,
        stagnation_tolerance=config.stagnation_tolerance,
        random_seed=random_seed,
        watershed_connectivity=config.watershed_connectivity,
    )
    return labels, markers


def _score(phantom, labels: np.ndarray) -> dict[str, float]:
    return bench_metrics.evaluate(phantom.labels, labels, phantom.spacing_um_zyx)


def _axis_values(config: SearchConfig) -> dict[str, list[float]]:
    """Current searched values per axis, sorted."""

    return {
        "local_threshold_deg": sorted(config.local_thresholds_deg),
        "footprint_tolerance": sorted(config.footprint_tolerances),
        "footprint_radius_um": sorted(config.footprint_radii_um),
        "min_cell_size": sorted(int(v) for v in config.min_cell_sizes),
        "kam_radius_um": sorted(config.kam_radii_um),
    }


def _extend_axis(
    values: list[float], best: float, name: str, factor: float
) -> float | None:
    """A value just outside the axis, when the best result sits on an end.

    Returns ``None`` when the best value is interior or the axis has reached its
    hard limit, which is what lets the refinement loop terminate.
    """

    spec = AXIS_SPECS[name]
    if len(values) < 2 or best not in values:
        return None
    index = values.index(best)
    if 0 < index < len(values) - 1:
        return None

    if index == 0:
        candidate = values[0] / factor
        if candidate <= spec["lo"]:
            return None
    else:
        candidate = values[-1] * factor
        if candidate >= spec["hi"]:
            return None
    if spec["kind"] == "integer":
        candidate = int(round(candidate))
        if candidate in values:
            return None
    return candidate


def _bisect_axis(values: list[float], best: float, config: SearchConfig, name: str
                 ) -> list[float]:
    """Midpoints between the best value and its immediate neighbours.

    Empty once both gaps are already finer than the refinement tolerance, so
    "every immediate neighbouring value has been tested" becomes a terminating
    condition rather than an unbounded subdivision.
    """

    spec = AXIS_SPECS[name]
    if best not in values:
        return []
    index = values.index(best)
    out: list[float] = []
    for neighbour in (index - 1, index + 1):
        if not 0 <= neighbour < len(values):
            continue
        low, high = sorted((values[index], values[neighbour]))
        if spec["kind"] == "integer":
            if high - low <= config.refine_min_size_step:
                continue
            candidate = int(round(0.5 * (low + high)))
            if candidate in values or candidate in out:
                continue
        else:
            if low <= 0 or high / low <= config.refine_ratio:
                continue
            candidate = float(np.sqrt(low * high))
            if any(abs(candidate - v) < 1e-12 for v in values + out):
                continue
        out.append(candidate)
    return out


def _neighbourhood(
    axes: dict[str, list[float]], best: dict[str, float]
) -> list[dict[str, float]]:
    """Full product of each axis's best value and its immediate neighbours."""

    per_axis = []
    for name in PARAMETER_NAMES:
        values = axes[name]
        index = values.index(best[name])
        window = [
            values[i] for i in (index - 1, index, index + 1) if 0 <= i < len(values)
        ]
        per_axis.append(window)
    return [dict(zip(PARAMETER_NAMES, combo)) for combo in itertools.product(*per_axis)]


def _point_key(point: dict[str, float]) -> tuple:
    return tuple(round(float(point[name]), 10) for name in PARAMETER_NAMES)


def _evaluate_points(
    phantom,
    kam: KamCache,
    points: list[dict[str, float]],
    config: SearchConfig,
    cache: dict[tuple, dict[str, Any]],
    rows: list[dict[str, Any]],
    round_index: int,
    random_seed: int = 0,
    verbose: bool = True,
) -> None:
    """Evaluate the uncached points, appending one row each."""

    pending = [p for p in points if _point_key(p) not in cache]
    if verbose and pending:
        print(f"  round {round_index}: {len(pending)} new points "
              f"({len(points) - len(pending)} cached)")
    for point in pending:
        started = time.perf_counter()
        row: dict[str, Any] = {
            "stage": "search", "round": round_index,
            "random_seed": random_seed, **point,
        }
        try:
            labels, markers = run_flood_fill_point(
                phantom, kam, point, config, random_seed
            )
            row.update(_score(phantom, labels))
            row.update(bench_metrics.marker_confusion(markers, phantom.labels))
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["runtime_seconds"] = time.perf_counter() - started
        rows.append(row)
        cache[_point_key(point)] = row


def refine_flood_fill(
    phantom,
    kam: KamCache,
    config: SearchConfig,
    *,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    """Coarse-to-fine search around the starting axes.

    Round 1 evaluates the full focused grid.  Each later round evaluates the
    full product of the best value and its immediate neighbours on every axis,
    after extending any axis whose best value sits on an end and bisecting the
    gaps around the best value.  The loop stops when a round adds no points,
    which can only happen once the best point is interior on every axis and
    every immediate neighbour has been evaluated.
    """

    axes = _axis_values(config)
    cache: dict[tuple, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    initial = [
        dict(zip(PARAMETER_NAMES, combo))
        for combo in itertools.product(*(axes[name] for name in PARAMETER_NAMES))
    ]
    if verbose:
        print(f"round 1: focused grid of {len(initial)} points")
    _evaluate_points(phantom, kam, initial, config, cache, rows, 1, verbose=verbose)

    best_row = _best_row(cache)
    converged = False
    for round_index in range(2, int(config.max_refinement_rounds) + 2):
        best = {name: best_row[name] for name in PARAMETER_NAMES}
        extended, bisected = {}, {}
        for name in PARAMETER_NAMES:
            candidate = _extend_axis(
                axes[name], best[name], name, config.extension_factor
            )
            if candidate is not None:
                axes[name] = sorted(axes[name] + [candidate])
                extended[name] = candidate
            for candidate in _bisect_axis(axes[name], best[name], config, name):
                axes[name] = sorted(axes[name] + [candidate])
                bisected.setdefault(name, []).append(candidate)

        points = _neighbourhood(axes, best)
        pending = [p for p in points if _point_key(p) not in cache]
        history.append(
            {
                "round": round_index,
                "best_before": {k: float(v) for k, v in best.items()},
                "best_ari_before": float(best_row["ari"]),
                "extended": {k: float(v) for k, v in extended.items()},
                "bisected": {k: [float(v) for v in vs] for k, vs in bisected.items()},
                "new_points": len(pending),
            }
        )
        if not pending:
            converged = True
            if verbose:
                print(f"round {round_index}: converged, nothing new to test")
            break
        _evaluate_points(
            phantom, kam, points, config, cache, rows, round_index, verbose=verbose
        )
        best_row = _best_row(cache)
        if verbose:
            print(f"  best ARI {float(best_row['ari']):.4f}")

    best = {name: best_row[name] for name in PARAMETER_NAMES}
    # A single-value axis is not "on a boundary", it is simply not searched.
    boundary = {
        name: (
            len(axes[name]) > 1
            and (best[name] == min(axes[name]) or best[name] == max(axes[name]))
        )
        for name in PARAMETER_NAMES
    }
    summary = {
        "axes": {name: [float(v) for v in axes[name]] for name in PARAMETER_NAMES},
        "rounds": history,
        "n_evaluated": len(cache),
        "converged": converged,
        "on_grid_boundary": boundary,
        "any_on_boundary": any(boundary.values()),
        "unsearched_axes": [
            name for name in PARAMETER_NAMES if len(axes[name]) < 2
        ],
    }
    return rows, best, summary


def _best_row(cache: dict[tuple, dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in cache.values() if row.get("status") == "ok"]
    if not ok:
        raise RuntimeError("No flood-fill operating point succeeded.")
    return max(ok, key=lambda row: float(row["ari"]))


def repeat_candidates(
    phantom,
    kam: KamCache,
    candidates: list[dict[str, float]],
    config: SearchConfig,
    rows: list[dict[str, Any]],
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Repeat each candidate over the seed orders and select the best.

    Primary criterion is the mean ARI.  Candidates within ``selection_tolerance``
    of the leader count as tied and are separated by, in order: mean VI, mean
    boundary ASSD, absolute mean cell-count error, leaked marker volume, split
    cells, unseeded cells, and finally seed-to-seed ARI spread.
    """

    if verbose:
        print(
            f"repeats: {len(candidates)} candidates x "
            f"{len(config.repeat_seeds)} seed orders"
        )
    summaries: list[dict[str, Any]] = []
    for point in candidates:
        group: list[dict[str, Any]] = []
        for seed in config.repeat_seeds:
            started = time.perf_counter()
            row: dict[str, Any] = {
                "stage": "repeat", "random_seed": int(seed), **point,
            }
            try:
                labels, markers = run_flood_fill_point(
                    phantom, kam, point, config, int(seed)
                )
                row.update(_score(phantom, labels))
                row.update(bench_metrics.marker_confusion(markers, phantom.labels))
                row["status"] = "ok"
                group.append(row)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["runtime_seconds"] = time.perf_counter() - started
            rows.append(row)
        if not group:
            continue
        entry: dict[str, Any] = {**point, "n_repeats": len(group)}
        for metric in REPEATED_METRICS:
            values = np.array([float(row[metric]) for row in group])
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(
                values.std(ddof=1) if values.size > 1 else 0.0
            )
        summaries.append(entry)
    if not summaries:
        raise RuntimeError("No candidate survived the repeated seed orders.")

    leader = max(entry["ari_mean"] for entry in summaries)
    tied = [
        entry for entry in summaries
        if entry["ari_mean"] >= leader - config.selection_tolerance
    ]
    selected = min(
        tied,
        key=lambda entry: (
            entry["vi_total_bits_mean"],
            entry["boundary_assd_um_mean"],
            abs(entry["cell_count_error_mean"]),
            entry["percolating_marker_volume_fraction_mean"],
            entry["cells_split_mean"],
            entry["cells_unseeded_mean"],
            entry["ari_std"],
        ),
    )
    selected = dict(selected)
    selected["n_tied_within_tolerance"] = len(tied)
    selected["selection_tolerance"] = float(config.selection_tolerance)
    selected["finalist_summaries"] = sorted(
        summaries, key=lambda entry: -entry["ari_mean"]
    )
    return selected


def search_kam_baseline(
    phantom, kam: KamCache, config: SearchConfig, *, verbose: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sweep the KAM footprint radius and threshold percentile, keep the best.

    The baseline gets its own radius search so it is not handicapped by a
    radius chosen for the other arm.
    """

    rows: list[dict[str, Any]] = []
    for radius, percentile in itertools.product(
        config.kam_radii_um, config.kam_percentiles
    ):
        started = time.perf_counter()
        row: dict[str, Any] = {
            "stage": "kam_baseline",
            "kam_radius_um": float(radius),
            "kam_percentile": float(percentile),
        }
        try:
            labels, markers, threshold = pipelines.run_kam_threshold(
                kam[radius],
                phantom.mask,
                percentile=float(percentile),
                min_cell_size=min(config.min_cell_sizes),
                connectivity=1,
                watershed_connectivity=config.watershed_connectivity,
            )
            row["kam_threshold_deg"] = threshold
            row.update(_score(phantom, labels))
            row.update(bench_metrics.marker_confusion(markers, phantom.labels))
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["runtime_seconds"] = time.perf_counter() - started
        rows.append(row)

    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        raise RuntimeError("No KAM-threshold operating point succeeded.")
    best = max(ok, key=lambda row: float(row["ari"]))
    if verbose:
        print(
            f"KAM baseline: best ARI {best['ari']:.3f} at percentile "
            f"{best['kam_percentile']:g}, radius {best['kam_radius_um']:g} um"
        )
    return rows, best


def kam_failure_modes(
    phantom, kam: np.ndarray, config: SearchConfig
) -> list[dict[str, Any]]:
    """Percolation and fragmentation of KAM markers across the whole sweep.

    Reported on the raw connected components of the sub-threshold KAM field,
    before any refinement, over a finer percentile grid than the baseline sweep.
    This is the diagnostic that shows whether a KAM threshold exists at which
    the marker set neither leaks across weak walls nor breaks cells apart.
    """

    from scipy.ndimage import generate_binary_structure, label

    valid = phantom.mask & np.isfinite(kam)
    values = kam[valid]
    structure = generate_binary_structure(kam.ndim, 1)
    minimum = min(config.min_cell_sizes)

    rows = []
    for percentile in config.failure_mode_percentiles:
        threshold = float(np.percentile(values, float(percentile)))
        markers, _ = label(valid & (kam < threshold), structure=structure)
        markers = pipelines._drop_small_labels(markers.astype(np.int32), minimum)
        rows.append(
            {
                "kam_percentile": float(percentile),
                "kam_threshold_deg": threshold,
                **bench_metrics.marker_confusion(markers, phantom.labels),
            }
        )
    return rows


def inactive_parameter_profile(
    phantom,
    kam: KamCache,
    selected_point: dict[str, float],
    config: SearchConfig,
    random_seed: int = 1,
) -> dict[str, Any]:
    """Re-check that the two held-fixed parameters really are inactive.

    They were excluded from the search because a profile through the previous
    optimum showed them flat.  The phantom has changed since, so the claim is
    re-tested here rather than assumed: each is varied one at a time about the
    selected point and the ARI recorded.
    """

    from dataclasses import replace as dc_replace

    out: dict[str, Any] = {}
    for name, values in (
        ("global_threshold_deg", (0.05, 0.10, 0.17, 0.30, 0.60, 1.20)),
    ):
        profile = {}
        for value in values:
            variant = dc_replace(config, **{name: float(value)})
            try:
                labels, _ = run_flood_fill_point(
                    phantom, kam, selected_point, variant, random_seed
                )
                profile[str(value)] = float(
                    _score(phantom, labels)["ari"]
                )
            except Exception as exc:
                profile[str(value)] = f"{type(exc).__name__}: {exc}"
        numeric = [v for v in profile.values() if isinstance(v, float)]
        held = float(getattr(config, name))
        out[name] = {
            "held_at": held,
            "profile": profile,
            "ari_spread_over_profile": (
                float(max(numeric) - min(numeric)) if numeric else float("nan")
            ),
        }
    return out


def phantom_properties(phantom) -> dict[str, Any]:
    """Measured properties of the phantom, so the claims about it are checkable."""

    from scipy import stats
    from scipy.ndimage import distance_transform_edt

    volumes = phantom.cell_volumes_um3
    volumes = volumes[volumes > 0]
    log_volumes = np.log(volumes)
    misorientation = phantom_module.neighbour_misorientations_deg(phantom)
    fitted_k, _, fitted_sigma = stats.chi.fit(misorientation, floc=0)
    spread = phantom_module.intradomain_angular_spread_deg(
        phantom.labels, phantom.segmentation_field, phantom.config.n_cells
    )
    angular_range = phantom_module.intradomain_angular_range_deg(
        phantom.labels, phantom.segmentation_field, phantom.config.n_cells
    )
    offsets = phantom_module.boundary_normal_ridge_offsets_um(
        phantom.labels, phantom.segmentation_field, phantom.spacing_um_zyx
    )

    kam = pipelines.masked_kam(
        phantom.segmentation_field,
        phantom.mask,
        pipelines.isotropic_footprint(phantom.spacing_um_zyx, 1.0),
    )
    boundary = phantom_module.ground_truth_boundaries(phantom.labels)
    distance = distance_transform_edt(~boundary, sampling=phantom.spacing_um_zyx)
    finite = np.isfinite(kam)
    core = (distance > 1.5) & finite
    percentiles = (10, 50, 90)

    return {
        "n_cells": int(volumes.size),
        "cell_volume_um3": {
            str(p): float(v)
            for p, v in zip(percentiles, np.percentile(volumes, percentiles))
        },
        "equivalent_diameter_um": {
            str(p): float((6.0 * v / np.pi) ** (1 / 3))
            for p, v in zip(percentiles, np.percentile(volumes, percentiles))
        },
        "log_volume_sigma": float(log_volumes.std()),
        "log_volume_skew": float(stats.skew(log_volumes)),
        "log_volume_normality_ks_p": float(
            stats.kstest(
                log_volumes, "norm", args=(log_volumes.mean(), log_volumes.std())
            ).pvalue
        ),
        "misorientation_deg_median": float(np.median(misorientation)),
        "misorientation_n_facets": int(misorientation.size),
        "misorientation_fitted_k": float(fitted_k),
        "misorientation_target_k": float(phantom.config.misorientation_k),
        "misorientation_fitted_sigma_deg": float(fitted_sigma),
        "misorientation_target_sigma_deg": float(
            phantom.config.misorientation_sigma_deg
        ),
        "facet_fraction_below_0p10_deg": float(np.mean(misorientation < 0.10)),
        "facet_fraction_below_0p05_deg": float(np.mean(misorientation < 0.05)),
        "ridge_offset_um_median": float(np.median(offsets)) if offsets.size else 0.0,
        "ridge_offset_um_median_abs": (
            float(np.median(np.abs(offsets))) if offsets.size else 0.0
        ),
        "wall_width_um": {
            str(p): float(v)
            for p, v in zip(
                percentiles, np.percentile(phantom.wall_width_um, percentiles)
            )
        },
        "intradomain_spread_definition": (
            "s_k = sqrt(Var(chi) + Var(phi)) per cell, manuscript Eq. A2. The "
            "experimental 0.184 deg is this quantity."
        ),
        "intradomain_spread_p99_deg": float(np.percentile(spread, 99)),
        "intradomain_spread_max_deg": float(np.max(spread)),
        "intradomain_range_definition": (
            "separate diagnostic: per-channel peak-to-peak span combined in "
            "quadrature. Not s_k and not comparable with it or with any s_k "
            "target."
        ),
        "intradomain_range_deg": {
            str(p): float(v)
            for p, v in zip(
                (5, 25, 50, 75, 95), np.percentile(angular_range, [5, 25, 50, 75, 95])
            )
        },
        "crystal_angular_spread_deg": float(
            phantom_module.crystal_angular_spread_deg(
                phantom.segmentation_field, phantom.mask
            )
        ),
        "channel_1_99_ranges_deg": phantom_module.channel_percentile_ranges_deg(
            phantom.segmentation_field, phantom.mask
        ),
        "crystal_spread_over_median_spread": float(
            phantom_module.crystal_angular_spread_deg(
                phantom.segmentation_field, phantom.mask
            )
            / np.median(spread)
        ),
        "median_misorientation_over_median_spread": float(
            np.median(misorientation) / np.median(spread)
        ),
        "intradomain_spread_deg": {
            str(p): float(v)
            for p, v in zip((5, 25, 50, 75, 95), np.percentile(spread, [5, 25, 50, 75, 95]))
        },
        "intradomain_spread_p95_over_median": float(
            np.percentile(spread, 95) / np.median(spread)
        ),
        "intradomain_spread_fraction_above_twice_median": float(
            np.mean(spread > 2.0 * np.median(spread))
        ),
        "kam_core_median_deg": float(np.median(kam[core])),
        "kam_wall_median_deg": float(np.median(kam[boundary & finite])),
        "kam_wall_to_core_ratio": float(
            np.median(kam[boundary & finite]) / np.median(kam[core])
        ),
    }


def _serialisable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_serialisable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialisable(v) for k, v in value.items()}
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialisable(row.get(key, "")) for key in keys})


def run_benchmark(
    phantom_config: PhantomConfig,
    search_config: SearchConfig,
    out_dir: Path,
    *,
    figure_layer: int | None = None,
    figure_upsample: int = 4,
    figure_dpi: int = 300,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate the phantom, run both searches, write every output."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    phantom = generate_phantom(phantom_config)
    kam = KamCache(phantom)

    ff_rows, best_point, refinement = refine_flood_fill(
        phantom, kam, search_config, verbose=verbose
    )
    ok = [row for row in ff_rows if row.get("status") == "ok"]
    ranked = sorted(ok, key=lambda row: -float(row["ari"]))
    candidates, seen = [], set()
    for row in ranked:
        point = {name: row[name] for name in PARAMETER_NAMES}
        key = _point_key(point)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(point)
        if len(candidates) >= search_config.n_finalists:
            break
    selected = repeat_candidates(
        phantom, kam, candidates, search_config, ff_rows, verbose=verbose
    )
    kam_rows, kam_best = search_kam_baseline(
        phantom, kam, search_config, verbose=verbose
    )
    write_csv(out_dir / "search.csv", ff_rows + kam_rows)

    baseline_radius = float(kam_best["kam_radius_um"])
    failure_rows = kam_failure_modes(
        phantom, kam[baseline_radius], search_config
    )
    write_csv(out_dir / "kam_failure_modes.csv", failure_rows)

    selected_point = {name: selected[name] for name in PARAMETER_NAMES}
    with (out_dir / "selected_parameters.json").open("w") as stream:
        json.dump(
            _serialisable(
                {
                    "selected_flood_fill_parameters": selected_point,
                    "selection_criterion": (
                        "highest mean ARI over "
                        f"{len(search_config.repeat_seeds)} flood-fill seed orders"
                    ),
                    "repeat_seeds": list(search_config.repeat_seeds),
                    "ari_mean": selected["ari_mean"],
                    "ari_std": selected["ari_std"],
                    "n_repeats": selected["n_repeats"],
                    "finalist_summaries": selected["finalist_summaries"],
                    "n_tied_within_tolerance": selected["n_tied_within_tolerance"],
                    "tie_breakers": [
                        "mean VI", "mean boundary ASSD", "seed-to-seed ARI std",
                    ],
                    "refinement": refinement,
                    "held_fixed": {
                        "global_threshold_deg": search_config.global_threshold_deg,
                    },
                    "search_config": asdict(search_config),
                    "phantom_config": asdict(phantom_config),
                }
            ),
            stream,
            indent=2,
            sort_keys=True,
        )

    # The figure shows a real member of the repeated set, not a fresh unscored
    # run, so its numbers appear in search.csv too.
    figure_seed = int(search_config.repeat_seeds[0])
    ff_labels, ff_markers = run_flood_fill_point(
        phantom, kam, selected_point, search_config, figure_seed
    )
    ff_metrics = _score(phantom, ff_labels)
    ff_metrics.update(bench_metrics.marker_confusion(ff_markers, phantom.labels))
    kam_labels, kam_markers, kam_threshold = pipelines.run_kam_threshold(
        kam[baseline_radius],
        phantom.mask,
        percentile=float(kam_best["kam_percentile"]),
        min_cell_size=min(search_config.min_cell_sizes),
        connectivity=1,
        watershed_connectivity=search_config.watershed_connectivity,
    )

    figures.render_intradomain_cdf_figure(
        out_dir / "intradomain_spread_cdf",
        phantom_module.intradomain_angular_spread_deg(
            phantom.labels, phantom.segmentation_field, phantom_config.n_cells
        ),
        angular_range_deg=phantom_module.intradomain_angular_range_deg(
            phantom.labels, phantom.segmentation_field, phantom_config.n_cells
        ),
    )

    figures.render_failure_modes_figure(
        out_dir / "kam_failure_modes",
        failure_rows,
        n_truth_cells=int(phantom.labels.max()),
        flood_fill=bench_metrics.marker_confusion(ff_markers, phantom.labels),
    )

    figure_info = figures.render_figure(
        out_dir / "benchmark_figure",
        field=phantom.segmentation_field,
        kam=kam[baseline_radius],
        truth_labels=phantom.labels,
        kam_labels=kam_labels,
        flood_fill_labels=ff_labels,
        spacing_um_zyx=phantom.spacing_um_zyx,
        layer=figure_layer,
        upsample=figure_upsample,
        dpi=figure_dpi,
    )

    region_means = phantom_module.ground_truth_region_means_deg(phantom)
    facets = {
        "flood_fill": bench_metrics.facet_recovery(
            phantom.labels, ff_labels, region_means
        ),
        "kam_threshold": bench_metrics.facet_recovery(
            phantom.labels, kam_labels, region_means
        ),
    }
    facet_bands = {
        name: bench_metrics.facet_recovery_bands(recovery)
        for name, recovery in facets.items()
    }
    figures.render_facet_recovery_figure(
        out_dir / "facet_recovery", facets
    )

    inactive = inactive_parameter_profile(
        phantom, kam, selected_point, search_config
    )

    n_truth_cells = int(phantom.labels.max())
    unseeded_fraction = float(
        ff_metrics["cells_unseeded"] / max(n_truth_cells, 1)
    )
    accepted = unseeded_fraction <= search_config.max_unseeded_fraction

    metrics = {
        "phantom": phantom_properties(phantom),
        "acceptance": {
            "criterion": (
                "flood fill at its optimum must seed at least "
                f"{100 * (1 - search_config.max_unseeded_fraction):.0f}% of the "
                "ground-truth cells"
            ),
            "unseeded_fraction": unseeded_fraction,
            "max_unseeded_fraction": float(search_config.max_unseeded_fraction),
            "phantom_accepted": bool(accepted),
        },
        "flood_fill": {
            "parameters": selected_point,
            "repeated": {
                metric: {
                    "mean": selected[f"{metric}_mean"],
                    "std": selected[f"{metric}_std"],
                }
                for metric in REPEATED_METRICS
            },
            "n_repeats": selected["n_repeats"],
            "figure_run": {"random_seed": figure_seed, **ff_metrics},
        },
        "kam_threshold": {
            "kam_percentile": float(kam_best["kam_percentile"]),
            "kam_threshold_deg": float(kam_threshold),
            "note": "swept over its threshold percentile, reported at its best ARI",
            **{key: kam_best[key] for key in REPEATED_METRICS},
        },
        "figure": figure_info,
        "refinement": refinement,
        "facet_recovery_bands": facet_bands,
        "inactive_parameter_profile": inactive,
        "kam_failure_modes": failure_rows,
        "n_search_points": len(ff_rows),
        "n_failed": sum(1 for row in ff_rows if row.get("status") != "ok"),
        "total_runtime_seconds": time.perf_counter() - started,
        "disell_version": _disell_version(),
    }
    with (out_dir / "metrics.json").open("w") as stream:
        json.dump(_serialisable(metrics), stream, indent=2, sort_keys=True)

    np.savez_compressed(
        out_dir / "phantom_and_labels.npz",
        ground_truth_labels=phantom.labels,
        angular_feature_field_deg=phantom.segmentation_field,
        measured_field_deg=phantom.field,
        kam_deg=kam[baseline_radius],
        flood_fill_labels=ff_labels,
        kam_threshold_labels=kam_labels,
        wall_width_um=phantom.wall_width_um,
        spacing_um_zyx=np.asarray(phantom.spacing_um_zyx),
    )
    return metrics


def _disell_version() -> str:
    try:
        from importlib.metadata import version

        return version("disell")
    except Exception:
        return "unknown"


def search_config_from_args(args: argparse.Namespace) -> SearchConfig:
    """Start from SearchConfig's own axes; override only what was passed."""

    mapping = {
        "local_thresholds": ("local_thresholds_deg", float),
        "footprint_radii_um": ("footprint_radii_um", float),
        "min_cell_sizes": ("min_cell_sizes", int),
        "footprint_tolerances": ("footprint_tolerances", float),
        "kam_radii_um": ("kam_radii_um", float),
        "kam_percentiles": ("kam_percentiles", float),
        "repeat_seeds": ("repeat_seeds", int),
    }
    overrides: dict[str, Any] = {}
    for flag, (field, cast) in mapping.items():
        value = getattr(args, flag)
        if value is not None:
            overrides[field] = tuple(cast(v) for v in value)
    for flag, field, cast in (
        ("global_threshold_deg", "global_threshold_deg", float),
        ("n_finalists", "n_finalists", int),
        ("max_refinement_rounds", "max_refinement_rounds", int),
    ):
        value = getattr(args, flag)
        if value is not None:
            overrides[field] = cast(value)
    return replace(SearchConfig(), **overrides)


def phantom_config_from_args(args: argparse.Namespace) -> PhantomConfig:
    """Start from the phantom's own defaults; override only what was passed."""

    overrides: dict[str, Any] = {}
    if args.shape_zyx is not None:
        overrides["shape_zyx"] = tuple(int(v) for v in args.shape_zyx)
    if args.spacing_um_zyx is not None:
        overrides["spacing_um_zyx"] = tuple(float(v) for v in args.spacing_um_zyx)
    if args.n_cells is not None:
        overrides["n_cells"] = int(args.n_cells)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    return replace(PhantomConfig(), **overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, required=True)

    # The phantom module owns its defaults.  These flags default to None and
    # only override when passed, so the CLI cannot silently pin a stale shape or
    # cell count if PhantomConfig changes.
    group = parser.add_argument_group("phantom (defaults from PhantomConfig)")
    group.add_argument("--shape-zyx", nargs=3, type=int, default=None)
    group.add_argument("--spacing-um-zyx", nargs=3, type=float, default=None)
    group.add_argument("--n-cells", type=int, default=None)
    group.add_argument("--seed", type=int, default=None)

    # As for the phantom, these default to None so SearchConfig stays the single
    # source of truth; a stale CLI default must not silently replace the
    # focused axes the refinement loop starts from.
    group = parser.add_argument_group("flood-fill search (defaults from SearchConfig)")
    group.add_argument("--local-thresholds", nargs="+", type=float, default=None)
    group.add_argument("--footprint-radii-um", nargs="+", type=float, default=None)
    group.add_argument("--min-cell-sizes", nargs="+", type=int, default=None)
    group.add_argument("--kam-radii-um", nargs="+", type=float, default=None)
    group.add_argument("--global-threshold-deg", type=float, default=None)
    group.add_argument("--footprint-tolerances", nargs="+", type=float, default=None)
    group.add_argument("--kam-percentiles", nargs="+", type=float, default=None)
    group.add_argument("--n-finalists", type=int, default=None)
    group.add_argument("--repeat-seeds", nargs="+", type=int, default=None)
    group.add_argument("--max-refinement-rounds", type=int, default=None)

    group = parser.add_argument_group("figure")
    group.add_argument("--figure-layer", type=int, default=None)
    group.add_argument("--figure-upsample", type=int, default=4)
    group.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    phantom_config = phantom_config_from_args(args)
    search_config = search_config_from_args(args)
    metrics = run_benchmark(
        phantom_config,
        search_config,
        args.out_dir,
        figure_layer=args.figure_layer,
        figure_upsample=int(args.figure_upsample),
        figure_dpi=int(args.figure_dpi),
        verbose=not args.quiet,
    )

    selected = metrics["flood_fill"]["parameters"]
    repeated = metrics["flood_fill"]["repeated"]
    print(f"\nWrote outputs to {args.out_dir}")
    print(
        f"{metrics['n_search_points']} search points "
        f"({metrics['n_failed']} failed), "
        f"{metrics['total_runtime_seconds']:.0f} s"
    )
    print("\nSelected flood-fill parameters:")
    for name in PARAMETER_NAMES:
        print(f"  {name:>22} = {selected[name]:g}")
    print(f"\nRepeated over {metrics['flood_fill']['n_repeats']} seed orders:")
    for metric, unit in (
        ("ari", ""),
        ("vi_total_bits", " bits"),
        ("vi_split_bits", " bits"),
        ("vi_merge_bits", " bits"),
        ("boundary_assd_um", " um"),
        ("cell_count_error", " cells"),
        ("n_cells_pred", " cells"),
    ):
        entry = repeated[metric]
        print(f"  {metric:>18} = {entry['mean']:.3f} +/- {entry['std']:.3f}{unit}")
    kam = metrics["kam_threshold"]
    print(
        f"\nKAM baseline (own best ARI, percentile {kam['kam_percentile']:g}): "
        f"ARI {kam['ari']:.3f}, VI {kam['vi_total_bits']:.2f} bits, "
        f"ASSD {kam['boundary_assd_um']:.2f} um, "
        f"cell-count error {int(kam['cell_count_error']):+d}"
    )
    print(f"\nGround truth: {metrics['phantom']['n_cells']} cells")
    acceptance = metrics["acceptance"]
    verdict = "ACCEPTED" if acceptance["phantom_accepted"] else "REJECTED"
    print(
        f"Phantom {verdict}: flood fill leaves "
        f"{100 * acceptance['unseeded_fraction']:.1f}% of cells unseeded at its "
        f"optimum (limit {100 * acceptance['max_unseeded_fraction']:.0f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Staged, resumable oracle search over the flood-fill parameters.

This is an **oracle** search: every configuration is scored against the known
ground-truth labels of the synthetic phantom.  What it measures is the *upper
bound* on what these parameters can do on this microstructure, and how sharply
that bound depends on each of them.  It is not a procedure a user without
ground truth could run, and none of the selected values should be transferred
to experimental data as if they were.

Stages
------
1. **Broad**  Scrambled Sobol over all six parameters, two independent
   scrambles, ~10k unique configurations at one seed order, plus a random
   subset repeated at further seed orders so seed sensitivity is measured
   across the whole space rather than only at the optimum.
2. **Regions**  Promising regions are retained rather than a single best point:
   the top configurations are clustered in normalised parameter space and the
   best few clusters are kept, so a second basin is not thrown away because it
   lost the first round by 0.001 ARI.
3. **Refine**  Each retained region is densely resampled in a shrinking local
   box, round after round, until two consecutive rounds fail to improve any of
   the four objectives by a meaningful margin.
4. **Surfaces**  Pairwise response surfaces through the balanced solution, for
   the parameter pairs the surrogate says interact.
5. **Finalists**  The top candidates under every objective are repeated over 20
   seed orders, which is what the reported numbers come from.

Every evaluated configuration is appended to ``evaluations.jsonl`` and the
search skips anything already there, so it can be stopped and resumed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import oracle_core as oc
import oracle_runner
import oracle_select as osel
from oracle_store import Store

DEFAULT_OUT = BENCHMARK_DIR / "oracle_results"

#: Broad-stage ranges.  Deliberately wider than the previous search on every
#: axis: ``min_cell_size`` reaches 400 (the smallest true cell is 179 voxels, so
#: the range covers and passes that limit), and the global threshold covers both
#: its disabled and its active regime.
SPACE: dict[str, dict[str, Any]] = {
    "local_threshold_deg": {"kind": "log", "low": 5e-4, "high": 0.30},
    "global_threshold_deg": {
        "kind": "log_or_off", "low": 5e-3, "high": 2.0, "off_probability": 0.25
    },
    "footprint_tolerance": {"kind": "linear", "low": 0.01, "high": 0.80},
    "footprint_radius_um": {"kind": "log", "low": 0.45, "high": 2.60},
    "min_cell_size": {"kind": "log_int", "low": 5, "high": 400},
    "kam_radius_um": {"kind": "log", "low": 0.42, "high": 2.60},
}

#: Seed orders.  The broad stage uses one; a subset gets ``BROAD_EXTRA_SEEDS``;
#: the finalists get the full set.
BROAD_SEED = 0
BROAD_EXTRA_SEEDS = (1, 2)
FINAL_SEEDS = tuple(range(1, 21))
SCREEN_SEEDS = tuple(range(5))

#: What counts as a meaningful improvement, per objective, when deciding
#: whether a refinement round achieved anything.
IMPROVEMENT = {
    "ari_mean": 0.0005,
    "vi_total_bits_mean": 0.002,
    "abs_cell_count_error": 1.0,
    "balanced_score": 0.005,
}


# ------------------------------------------------------------------ sampling


def from_unit(unit: Sequence[float]) -> dict[str, Any]:
    """Map a point of the unit cube onto one configuration."""

    out: dict[str, Any] = {}
    for value, (name, spec) in zip(unit, SPACE.items()):
        value = float(np.clip(value, 0.0, 1.0 - 1e-12))
        kind = spec["kind"]
        if kind == "log":
            out[name] = float(
                np.exp(
                    np.log(spec["low"])
                    + value * (np.log(spec["high"]) - np.log(spec["low"]))
                )
            )
        elif kind == "linear":
            out[name] = float(spec["low"] + value * (spec["high"] - spec["low"]))
        elif kind == "log_int":
            out[name] = int(
                round(
                    np.exp(
                        np.log(spec["low"])
                        + value * (np.log(spec["high"]) - np.log(spec["low"]))
                    )
                )
            )
        elif kind == "log_or_off":
            # The disabled regime is a distinct branch of the algorithm, not a
            # limiting value, so it gets its own slice of the axis instead of
            # being approximated by a very large threshold.
            probability = float(spec["off_probability"])
            if value < probability:
                out[name] = oc.GLOBAL_DISABLED
            else:
                rescaled = (value - probability) / (1.0 - probability)
                out[name] = float(
                    np.exp(
                        np.log(spec["low"])
                        + rescaled * (np.log(spec["high"]) - np.log(spec["low"]))
                    )
                )
        else:
            raise ValueError(f"unknown axis kind {kind!r}")
    return out


def sobol_points(n: int, scramble_seed: int) -> np.ndarray:
    from scipy.stats import qmc

    sampler = qmc.Sobol(d=len(SPACE), scramble=True, seed=int(scramble_seed))
    return sampler.random(int(n))


def lhs_points(n: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=len(SPACE), seed=int(seed))
    return sampler.random(int(n))


def canonical_params(
    params: dict[str, Any], workspace: oc.Workspace
) -> dict[str, Any]:
    return oc.canonical(oc.Config(**params), workspace).as_dict()


def unique_configs(
    raw: Iterable[dict[str, Any]], workspace: oc.Workspace
) -> list[dict[str, Any]]:
    """Canonicalise, then drop duplicates.

    Canonicalisation collapses radii that rasterise to the same footprint, so a
    Sobol design of 10k points yields fewer than 10k distinct configurations;
    the search tops the design up until the *unique* count is reached.
    """

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for params in raw:
        canonical = canonical_params(params, workspace)
        key = oc.Config(**canonical).key()
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


# -------------------------------------------------------------------- stages


def run_tasks(
    tasks: list[tuple[dict[str, Any], int]],
    store: Store,
    cache_dir: Path,
    processes: int,
    *,
    label: str,
    stage: str,
    batch: int = 75,
    out_dir: Path | None = None,
) -> int:
    """Evaluate what is not already stored, streaming results to disk."""

    pending = [
        (params, seed)
        for params, seed in tasks
        if not store.has(oc.Config(**params).key(), seed)
    ]
    if not pending:
        print(f"  {label}: nothing new ({len(tasks)} already stored)", flush=True)
        return 0
    print(
        f"  {label}: {len(pending)} new of {len(tasks)} "
        f"({len(tasks) - len(pending)} cached)",
        flush=True,
    )
    written = 0
    completed = 0
    for row in oracle_runner.evaluate_batch(
        pending, cache_dir, processes, label=f"{label} ", progress_every=batch
    ):
        row["stage"] = stage
        # One fsync per completed trial.  This deliberately favours restart
        # safety over a small amount of filesystem throughput.
        written += store.append([row])
        completed += 1
        if completed % batch == 0 or completed == len(pending):
            _write_progress(store, out_dir or cache_dir.parent, label, completed,
                            len(pending))
    return written


def _write_progress(store: Store, out_dir: Path, label: str, completed: int,
                    requested: int) -> None:
    """Atomically publish progress and current objective extrema."""

    snapshot = _objective_snapshot(store)
    payload = {
        "label": label, "completed_this_invocation": int(completed),
        "requested_this_invocation": int(requested), "stored_rows": len(store),
        "best_so_far": snapshot, "updated_unix": time.time(),
    }
    path = Path(out_dir) / "progress.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True))
    temporary.replace(path)


def stage_broad(
    store: Store,
    workspace: oc.Workspace,
    cache_dir: Path,
    processes: int,
    *,
    n_unique: int,
    extra_seed_fraction: float = 0.15,
    rng_seed: int = 20260812,
) -> dict[str, Any]:
    """Sobol and Latin-hypercube coverage of the whole space."""

    print(f"\nstage 1: broad search, target {n_unique} unique configurations")
    designs: list[np.ndarray] = []
    # Two independent Sobol scrambles plus one Latin hypercube: three different
    # ways of filling the cube, so the coverage does not inherit the artefacts
    # of any one of them.
    size = int(2 ** np.ceil(np.log2(max(n_unique, 2))))
    designs.append(sobol_points(size // 2, scramble_seed=11))
    designs.append(sobol_points(size // 2, scramble_seed=97))
    designs.append(lhs_points(size // 2, seed=1234))
    raw = [from_unit(u) for design in designs for u in design]
    configs = unique_configs(raw, workspace)[:n_unique]
    print(f"  {len(configs)} unique configurations after canonicalisation")

    radii = {c["kam_radius_um"] for c in configs}
    made = oracle_runner.precompute_kam(sorted(radii), cache_dir, processes)
    print(f"  KAM fields: {len(radii)} distinct footprints, {made} computed now")

    run_tasks(
        [(c, BROAD_SEED) for c in configs],
        store, cache_dir, processes, label="broad", stage="broad",
    )

    rng = np.random.default_rng(rng_seed)
    subset_size = int(round(extra_seed_fraction * len(configs)))
    subset = [configs[i] for i in rng.choice(len(configs), subset_size, replace=False)]
    run_tasks(
        [(c, seed) for c in subset for seed in BROAD_EXTRA_SEEDS],
        store, cache_dir, processes,
        label="broad seed-repeat", stage="broad_seeds",
    )
    return {
        "n_unique_requested": int(n_unique),
        "n_unique_configs": len(configs),
        "n_distinct_kam_footprints": len(radii),
        "seed_repeat_subset": subset_size,
        "seed_repeat_seeds": list(BROAD_EXTRA_SEEDS),
        "space": SPACE,
    }


def _unit_coordinates(frame) -> np.ndarray:
    """Parameters mapped back into the unit cube, for distance comparisons."""

    columns = []
    for name, spec in SPACE.items():
        values = frame[name].to_numpy(dtype=float)
        if spec["kind"] in ("log", "log_int"):
            values = np.log(np.maximum(values, 1e-12))
            low, high = np.log(spec["low"]), np.log(spec["high"])
        elif spec["kind"] == "log_or_off":
            # "off" is its own category; place it at -1 so it can never be
            # averaged into an active threshold when a region centre is formed.
            active = values > 0
            scaled = np.full(values.shape, -1.0)
            low, high = np.log(spec["low"]), np.log(spec["high"])
            scaled[active] = (np.log(values[active]) - low) / (high - low)
            columns.append(scaled)
            continue
        else:
            low, high = spec["low"], spec["high"]
        columns.append((values - low) / (high - low))
    return np.column_stack(columns)


def retain_regions(
    aggregated,
    *,
    n_regions: int = 6,
    pool: int = 250,
    min_separation: float = 0.12,
) -> list[dict[str, Any]]:
    """Distinct promising regions, not just the single best point.

    The top ``pool`` configurations by ARI are walked in order and a candidate
    is kept as a new region only if it is at least ``min_separation`` away, in
    normalised parameter space, from every region already kept.  A greedy
    farthest-first rule rather than k-means: what matters is that the retained
    regions are *distinct*, and k-means would happily return several centres
    inside one basin.
    """

    frame = aggregated.sort_values("ari_mean", ascending=False).head(pool)
    coordinates = _unit_coordinates(frame)
    regions: list[dict[str, Any]] = []
    for position in range(len(frame)):
        point = coordinates[position]
        if any(
            np.linalg.norm(point - r["coordinates"]) < min_separation
            for r in regions
        ):
            continue
        row = frame.iloc[position]
        regions.append(
            {
                "coordinates": point,
                "params": {name: row[name] for name in osel.PARAMETER_NAMES},
                "ari": float(row["ari_mean"]),
                "vi": float(row["vi_total_bits_mean"]),
                "n_cells": float(row["n_cells_pred_mean"]),
            }
        )
        if len(regions) >= n_regions:
            break
    return regions


def local_design(
    centre: dict[str, Any],
    radius: float,
    n: int,
    rng: np.random.Generator,
    workspace: oc.Workspace,
) -> list[dict[str, Any]]:
    """Sobol box of relative width ``radius`` around one region centre.

    The box is built in the same normalised coordinates as the global space, so
    "half as wide" means the same thing on a log axis as on a linear one.  The
    global-threshold axis keeps its disabled branch: a fraction of every local
    design switches it off, so a region cannot get stuck in the active regime
    simply because its centre was.
    """

    from scipy.stats import qmc

    sampler = qmc.Sobol(d=len(SPACE), scramble=True, seed=int(rng.integers(1 << 30)))
    design = sampler.random(int(n))
    centre_unit = _unit_coordinates(
        _one_row_frame(centre)
    )[0]
    out = []
    for point in design:
        unit = np.clip(centre_unit + radius * (2.0 * point - 1.0), 0.0, 1.0)
        params = from_unit(unit)
        # Preserve the centre's global-threshold regime most of the time, and
        # deliberately flip it the rest of the time.
        if rng.random() < 0.75:
            if centre["global_threshold_deg"] <= 0:
                params["global_threshold_deg"] = oc.GLOBAL_DISABLED
            elif params["global_threshold_deg"] <= 0:
                params["global_threshold_deg"] = float(
                    centre["global_threshold_deg"]
                )
        out.append(params)
    return unique_configs(out, workspace)


def _one_row_frame(params: dict[str, Any]):
    import pandas as pd

    return pd.DataFrame([{name: params[name] for name in SPACE}])


def stage_refine(
    store: Store,
    workspace: oc.Workspace,
    cache_dir: Path,
    processes: int,
    *,
    per_region: int = 256,
    max_rounds: int = 8,
    n_regions: int = 6,
    rng_seed: int = 7,
) -> dict[str, Any]:
    """Dense local refinement of every retained region, until convergence.

    A round is "an improvement" if any of the four objectives moves by more than
    its threshold in :data:`IMPROVEMENT`.  The loop stops after two consecutive
    rounds without one, which is the convergence test the study reports.
    """

    print("\nstage 3: dense refinement of the retained regions")
    rng = np.random.default_rng(rng_seed)
    history: list[dict[str, Any]] = []
    stale = 0
    radius = 0.18
    best = _objective_snapshot(store)
    print(f"  starting from {_format_snapshot(best)}")

    for round_index in range(1, int(max_rounds) + 1):
        aggregated = osel.aggregate(store.ok_rows())
        regions = retain_regions(aggregated, n_regions=n_regions)
        print(
            f"  round {round_index}: {len(regions)} regions, "
            f"box radius {radius:.3f}"
        )
        for index, region in enumerate(regions):
            print(
                f"    region {index}: ARI {region['ari']:.4f} "
                f"cells {region['n_cells']:.0f} "
                f"local={region['params']['local_threshold_deg']:.4g} "
                f"tol={region['params']['footprint_tolerance']:.3g} "
                f"min={int(region['params']['min_cell_size'])} "
                f"g={'off' if region['params']['global_threshold_deg'] <= 0 else format(region['params']['global_threshold_deg'], '.3g')}"
            )
        configs: list[dict[str, Any]] = []
        for region in regions:
            configs.extend(
                local_design(region["params"], radius, per_region, rng, workspace)
            )
        configs = unique_configs(configs, workspace)
        oracle_runner.precompute_kam(
            sorted({c["kam_radius_um"] for c in configs}), cache_dir, processes
        )
        run_tasks(
            [(c, BROAD_SEED) for c in configs],
            store, cache_dir, processes,
            label=f"refine r{round_index}", stage=f"refine_{round_index}",
        )

        snapshot = _objective_snapshot(store)
        improved = _improvement(best, snapshot)
        history.append(
            {
                "round": round_index,
                "box_radius": float(radius),
                "n_regions": len(regions),
                "n_configs": len(configs),
                "objectives": snapshot,
                "improved": improved,
                "regions": [
                    {k: v for k, v in region.items() if k != "coordinates"}
                    for region in regions
                ],
            }
        )
        print(f"    -> {_format_snapshot(snapshot)}  improved={improved}")
        if improved:
            stale = 0
            best = _merge_best(best, snapshot)
        else:
            stale += 1
            if stale >= 2:
                print("  two consecutive rounds without improvement: converged")
                break
        radius = max(radius * 0.6, 0.03)

    return {
        "rounds": history,
        "converged": stale >= 2,
        "final_objectives": _objective_snapshot(store),
        "improvement_thresholds": IMPROVEMENT,
    }


def _objective_snapshot(store: Store) -> dict[str, float]:
    aggregated = osel.aggregate(store.ok_rows())
    selected = osel.select_solutions(aggregated)
    if not selected:
        return {}
    retained = selected["retained"]
    return {
        "ari_mean": float(aggregated["ari_mean"].max()),
        "vi_total_bits_mean": float(aggregated["vi_total_bits_mean"].min()),
        "abs_cell_count_error": float(aggregated["abs_cell_count_error"].min()),
        "balanced_score": float(retained["balanced_score"].min()),
        "n_configs": int(len(aggregated)),
    }


def _improvement(before: dict[str, float], after: dict[str, float]) -> bool:
    if not before:
        return True
    for name, threshold in IMPROVEMENT.items():
        if name not in before or name not in after:
            continue
        direction = -1.0 if name == "ari_mean" else 1.0
        if direction * (after[name] - before[name]) < -threshold:
            return True
    return False


def _merge_best(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    out = dict(after)
    if before:
        out["ari_mean"] = max(before["ari_mean"], after["ari_mean"])
        for name in ("vi_total_bits_mean", "abs_cell_count_error", "balanced_score"):
            out[name] = min(before[name], after[name])
    return out


def _format_snapshot(snapshot: dict[str, float]) -> str:
    if not snapshot:
        return "(nothing yet)"
    return (
        f"max ARI {snapshot['ari_mean']:.4f}  min VI {snapshot['vi_total_bits_mean']:.3f}"
        f"  min |dcells| {snapshot['abs_cell_count_error']:.0f}"
        f"  best balanced {snapshot['balanced_score']:.4f}"
        f"  [{snapshot['n_configs']} configs]"
    )


def stage_finalists(
    store: Store,
    cache_dir: Path,
    processes: int,
    *,
    n_finalists: int = 30,
    seeds: Sequence[int] = FINAL_SEEDS,
) -> dict[str, Any]:
    """Repeat the leading candidates under every objective over many seeds."""

    print(f"\nstage 5: {n_finalists} finalists x {len(seeds)} seed orders")
    aggregated = osel.aggregate(store.ok_rows())
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(frame, column: str, ascending: bool, count: int) -> None:
        ordered = frame.sort_values(column, ascending=ascending)
        for _, row in ordered.head(count).iterrows():
            key = row["config_key"]
            if key in seen:
                continue
            seen.add(key)
            chosen.append(
                {name: row[name] for name in osel.PARAMETER_NAMES}
            )

    band = aggregated[
        aggregated["ari_mean"] >= float(aggregated["ari_mean"].max()) - osel.ARI_BAND
    ].copy()
    band["balanced_score"] = osel.balanced_score(band)
    take(aggregated, "ari_mean", False, 10)
    take(aggregated, "vi_total_bits_mean", True, 8)
    take(aggregated, "abs_cell_count_error", True, 6)
    take(band, "balanced_score", True, 12)
    chosen = chosen[:n_finalists]

    run_tasks(
        [
            ({**params, "min_cell_size": int(params["min_cell_size"])}, int(seed))
            for params in chosen
            for seed in seeds
        ],
        store, cache_dir, processes, label="finalists", stage="final",
    )
    return {
        "n_finalists": len(chosen),
        "seeds": [int(s) for s in seeds],
        "finalists": [
            {k: (int(v) if k == "min_cell_size" else float(v)) for k, v in p.items()}
            for p in chosen
        ],
    }


def stage_screening(
    store: Store,
    cache_dir: Path,
    processes: int,
    *,
    n_candidates: int = 50,
    seeds: Sequence[int] = SCREEN_SEEDS,
) -> dict[str, Any]:
    """Five-seed evaluation of a diverse multi-objective Pareto set."""

    aggregated = osel.aggregate(store.ok_rows())
    objectives = (
        ("ari_mean", -1), ("vi_total_bits_mean", +1),
        ("excess_fragments_total_mean", +1),
        ("abs_cell_count_error", +1),
        ("boundary_f1_at_0p4um_mean", -1),
    )
    front = osel.pareto_front(aggregated, objectives).copy()
    front["balanced_score"] = osel.balanced_score(front)
    # Walk several rankings in round-robin order.  This preserves extremes and
    # avoids filling all 50 slots with one dense neighbourhood of the front.
    rankings = [
        front.sort_values(name, ascending=direction > 0)
        for name, direction in objectives
    ] + [front.sort_values("balanced_score")]
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank in range(max((len(frame) for frame in rankings), default=0)):
        for frame in rankings:
            if rank >= len(frame):
                continue
            row = frame.iloc[rank]
            if row["config_key"] in seen:
                continue
            seen.add(row["config_key"])
            chosen.append({name: row[name] for name in osel.PARAMETER_NAMES})
            if len(chosen) >= n_candidates:
                break
        if len(chosen) >= n_candidates:
            break
    run_tasks(
        [({**p, "min_cell_size": int(p["min_cell_size"])}, int(seed))
         for p in chosen for seed in seeds],
        store, cache_dir, processes, label="screening", stage="screening",
    )
    return {"n_candidates": len(chosen), "seeds": list(map(int, seeds))}


# ----------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--processes", type=int, default=1,
                        help="must be 1; retained only for command compatibility")
    parser.add_argument("--memory-limit-gib", type=float, default=None,
                        help="per-trial RSS cap; default is min(40%% RAM, RAM-4 GiB)")
    parser.add_argument("--deadline", default=None,
                        help="stop launching trials at local YYYY-MM-DDTHH:MM")
    parser.add_argument("--broad", type=int, default=3000)
    parser.add_argument("--per-region", type=int, default=256)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--n-regions", type=int, default=6)
    parser.add_argument("--n-finalists", type=int, default=30)
    parser.add_argument("--n-screen", type=int, default=50)
    parser.add_argument(
        "--stages", nargs="+",
        default=["broad", "refine", "screen", "final"],
        choices=["broad", "refine", "screen", "final"],
    )
    args = parser.parse_args(argv)
    if args.processes != 1:
        parser.error("the memory-safe oracle runner requires --processes 1")
    if args.memory_limit_gib is not None:
        os.environ["DISELL_MEMORY_LIMIT_GIB"] = str(args.memory_limit_gib)
    if args.deadline:
        deadline = dt.datetime.fromisoformat(args.deadline)
        os.environ["DISELL_DEADLINE_EPOCH"] = str(deadline.timestamp())

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    oc.limit_threads()

    lock_path = out_dir / "oracle_search.lock"
    lock_stream = lock_path.open("a+")
    try:
        import fcntl
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another oracle search holds {lock_path}")
    lock_stream.seek(0); lock_stream.truncate()
    lock_stream.write(f"pid={os.getpid()} started={dt.datetime.now().isoformat()}\n")
    lock_stream.flush()

    store = Store(out_dir / "evaluations.jsonl")
    workspace = oc.load_workspace(cache_dir)
    print(f"store: {len(store)} rows already present", flush=True)

    summary: dict[str, Any] = {}
    started = time.perf_counter()
    if "broad" in args.stages:
        summary["broad"] = stage_broad(
            store, workspace, cache_dir, args.processes, n_unique=int(args.broad)
        )
    if "refine" in args.stages:
        summary["refine"] = stage_refine(
            store, workspace, cache_dir, args.processes,
            per_region=int(args.per_region),
            max_rounds=int(args.max_rounds),
            n_regions=int(args.n_regions),
        )
    if "screen" in args.stages:
        summary["screen"] = stage_screening(
            store, cache_dir, args.processes, n_candidates=int(args.n_screen)
        )
    if "final" in args.stages:
        summary["final"] = stage_finalists(
            store, cache_dir, args.processes, n_finalists=int(args.n_finalists)
        )
    summary["total_runtime_seconds"] = time.perf_counter() - started
    summary["n_rows"] = len(store)

    path = out_dir / "search_summary.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(summary)
    path.write_text(json.dumps(_plain(existing), indent=2, sort_keys=True))

    aggregated = osel.aggregate(store.ok_rows())
    selected = osel.select_solutions(aggregated)
    print(f"\n{len(store)} evaluations, {len(aggregated)} unique configurations")
    for name, row in selected.get("solutions", {}).items():
        print(f"  {name:>10}: {osel.describe_solution(row)}")
    return 0


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 4


if __name__ == "__main__":
    raise SystemExit(main())

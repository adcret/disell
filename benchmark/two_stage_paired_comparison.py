#!/usr/bin/env python3
"""Paired random-order versus size-prioritised multi-seed experiment.

This is deliberately separate from the primary two-stage trial store.  It uses
the same fixed latent field, truth, mask, physical spacing, KAM field and one
watershed implementation for both arms.  A single coordinator owns the fsynced
JSONL writer; spawned workers return scalar data only and are recycled after
15 trials.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIMARY = HERE / "two_stage_oracle_results"
OUT = HERE / "continuation_results" / "algorithm_comparison"
sys.path.insert(0, str(HERE))

import object_orientation_metrics as oom
import oracle_core as oc
import oracle_metrics as om
import pipelines
from two_stage_store import ResultStore, normalize_result, utc_now

ALGORITHMS = (
    "random_order_multiseed_v1",
    "size_prioritised_multiseed_v1",
)
MAX_ITERATIONS = 700_000
STAGNATION_TOLERANCE = 2_000
WATERSHED_CONNECTIVITY = 1
WORK = None
IDENTITY = None
STOP = False


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True))
    temporary.replace(path)


def _memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return (
        values["MemAvailable"], values["MemTotal"],
        values["SwapTotal"] - values["SwapFree"],
    )


def _tree_rss():
    import psutil
    root = psutil.Process(os.getpid())
    rss = []
    for process in [root] + root.children(recursive=True):
        try:
            rss.append(int(process.memory_info().rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return sum(rss), max(rss, default=0)


def _init_worker():
    global WORK, IDENTITY
    WORK = oc.load_workspace(HERE / "oracle_results" / "cache")
    IDENTITY = json.loads((PRIMARY / "implementation_audit.json").read_text())


def _identity(algorithm, parameters, seed):
    payload = {
        "experiment": "paired_algorithm_ablation_v1",
        "algorithm": algorithm,
        "python_source_sha256": IDENTITY["python_source_sha256"],
        "compiled_extension_sha256": IDENTITY["compiled_extension_sha256"],
        "latent_sha256": IDENTITY["latent_sha256"],
        "truth_sha256": IDENTITY["truth_sha256"],
        "mask_sha256": IDENTITY["mask_sha256"],
        "matching_metrics_version": 2,
        "matching_source_sha256": IDENTITY["matching_source_sha256"],
        "parameters": parameters,
        "candidate_order_seed": int(seed),
        "max_iterations": MAX_ITERATIONS,
        "stagnation_tolerance": STAGNATION_TOLERANCE,
        "recycle_small_grains": False,
        "watershed_connectivity": WATERSHED_CONNECTIVITY,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest(), payload


def _marker_metrics(markers):
    pairs, identity = oom.match_cells(
        WORK.labels, markers, WORK.field, WORK.spacing_um_zyx,
        purity_threshold=0.6, completeness_threshold=0.6,
    )
    rows, cols, counts, _, pred_sizes = om.contingency(WORK.labels, markers)
    dominant = om.dominant_map(rows, cols, counts, int(pred_sizes.size), by="pred")
    interface = om.interface_precision(
        WORK.labels, markers, WORK.spacing_um_zyx, dominant,
    )
    return {
        **{f"marker_{key}": value for key, value in om.marker_errors(markers, WORK.labels).items()},
        **{f"marker_identity_{key}": value for key, value in identity.items()},
        **{f"marker_{key}": value for key, value in interface.items()},
        "marker_matched_pair_count": len(pairs),
    }


def _run(task):
    import resource
    parameters = task["parameters"]
    seed = int(task["seed"])
    algorithm = task["algorithm"]
    digest, identity = _identity(algorithm, parameters, seed)
    cfg = oc.Config(**parameters)
    footprint = WORK.footprint(cfg.footprint_radius_um)
    kam = WORK.kam(cfg.kam_radius_um)
    started = utc_now()
    clock = time.perf_counter()
    raw = {
        "config_hash": digest,
        "configuration_identity": identity,
        "algorithm": algorithm,
        "candidate_order_seed": seed,
        "pairing_key": hashlib.sha256(
            json.dumps({"parameters": parameters, "seed": seed}, sort_keys=True).encode()
        ).hexdigest(),
        **parameters,
    }
    try:
        if algorithm == "size_prioritised_multiseed_v1":
            labels, markers, initial, diagnostics, final_sizes = (
                pipelines.run_flood_fill_two_stage(
                    WORK.field, WORK.mask, kam, footprint,
                    local_threshold_deg=cfg.local_threshold_deg,
                    global_threshold_deg=cfg.global_threshold_deg,
                    footprint_tolerance=cfg.footprint_tolerance,
                    min_cell_size=cfg.min_cell_size,
                    max_seed_attempts=MAX_ITERATIONS,
                    stagnation_tolerance=STAGNATION_TOLERANCE,
                    random_seed=seed,
                    watershed_connectivity=WATERSHED_CONNECTIVITY,
                    recycle_small_grains=False,
                )
            )
            candidate = _plain(diagnostics.get("candidate_collection") or {})
            final = _plain(diagnostics.get("final_growth") or {})
            raw.update({
                "preliminary_candidates_detected": int(len(initial)),
                "preliminary_candidate_size_median": (
                    float(np.median(initial)) if len(initial) else None
                ),
                "preliminary_candidate_size_p90": (
                    float(np.percentile(initial, 90)) if len(initial) else None
                ),
                "candidate_seeds_supplied": int(final.get("user_seeds_supplied", 0)),
                "candidate_seeds_skipped_claimed": int(final.get("user_seeds_skipped_claimed", 0)),
                "candidate_seeds_processed": int(final.get("user_seeds_processed", 0)),
                "marker_size_median": float(np.median(final_sizes)) if len(final_sizes) else None,
                "candidate_pass_saturated": bool(candidate.get("max_iterations_reached", False)),
                "final_pass_saturated": bool(final.get("max_iterations_reached", False)),
            })
            if int(markers.max()) == 0:
                raise RuntimeError("No valid seeds found or no accepted markers")
        else:
            labels, markers = pipelines.run_flood_fill(
                WORK.field, WORK.mask, kam, footprint,
                local_threshold_deg=cfg.local_threshold_deg,
                global_threshold_deg=cfg.global_threshold_deg,
                footprint_tolerance=cfg.footprint_tolerance,
                min_cell_size=cfg.min_cell_size,
                max_seed_attempts=MAX_ITERATIONS,
                stagnation_tolerance=STAGNATION_TOLERANCE,
                random_seed=seed,
                watershed_connectivity=WATERSHED_CONNECTIVITY,
            )
            raw.update(candidate_pass_saturated=False, final_pass_saturated=False)
        _, object_summary = oom.match_cells(
            WORK.labels, labels, WORK.field, WORK.spacing_um_zyx,
            purity_threshold=0.6, completeness_threshold=0.6,
        )
        raw.update(om.evaluate_partition(
            WORK.labels, labels, WORK.spacing_um_zyx, with_boundary=True,
        ))
        raw.update(object_summary)
        raw.update(_marker_metrics(markers))
        raw.update({
            "accepted_marker_count": int(markers.max()),
            "final_watershed_count": int(labels.max()),
            "unlabelled_marker_stage_fraction": float(np.mean(markers == 0)),
            "status": "ok",
        })
    except Exception as error:
        raw.update({
            "status": "error", "error_type": type(error).__name__,
            "error_message": str(error),
        })
    finally:
        raw.update({
            "worker_pid": os.getpid(),
            "worker_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "started": started, "finished": utc_now(),
            "elapsed_seconds": time.perf_counter() - clock,
        })
    return normalize_result(raw, identity)


def _config_tuple(row):
    return tuple(row[name] for name in oc.PARAMETER_NAMES)


def build_plan(n_configurations=32, seeds=range(21)):
    rows = [json.loads(line) for line in (PRIMARY / "evaluations.jsonl").read_text().splitlines()]
    good = pd.DataFrame([
        row for row in rows
        if row.get("status_category") == "ok" and row.get("scientific_metrics_version") == 2
    ])
    invalid = [
        row for row in rows
        if row.get("status_category") == "expected_algorithmic_invalid"
    ]
    if good.empty:
        raise RuntimeError("paired plan requires completed schema-v2 primary trials")
    good = good.drop_duplicates(list(oc.PARAMETER_NAMES))
    chosen = []

    def add(row, stratum):
        parameters = {
            name: int(row[name]) if name == "min_cell_size" else float(row[name])
            for name in oc.PARAMETER_NAMES
        }
        key = tuple(parameters.values())
        if key not in {tuple(item["parameters"].values()) for item in chosen}:
            chosen.append({"parameters": parameters, "stratum": stratum})

    # Object-optimal, ARI-optimal, count-near, and deliberately marginal bands.
    ordered = good.sort_values(
        ["orientation_correct_cells_at_0p02deg", "one_to_one_recovered_cells",
         "unmatched_true_cells", "merged_predicted_cells", "split_true_cells", "ari"],
        ascending=[False, False, True, True, True, False],
    )
    for _, row in ordered.head(10).iterrows():
        add(row, "object_promising")
    for _, row in good.nlargest(5, "ari").iterrows():
        add(row, "high_ari")
    for _, row in good.loc[good.cell_count_error.abs().sort_values().index].head(5).iterrows():
        add(row, "near_count")
    quantiles = np.linspace(0.05, 0.95, 12)
    for quantile in quantiles:
        target = good.orientation_correct_cells_at_0p02deg.quantile(quantile)
        row = good.iloc[(good.orientation_correct_cells_at_0p02deg - target).abs().argmin()]
        add(row, "marginal_quantile")
    for row in invalid[:4]:
        add(row, "failed_regime")
    chosen = chosen[:n_configurations]
    if len(chosen) < n_configurations:
        for _, row in good.sample(frac=1, random_state=20260824).iterrows():
            add(row, "coverage_fill")
            if len(chosen) == n_configurations:
                break
    plan = [
        {**item, "algorithm": algorithm, "seed": int(seed)}
        for item in chosen for seed in seeds for algorithm in ALGORITHMS
    ]
    _atomic_json(OUT / "paired_plan.json", {
        "n_configurations": len(chosen), "seeds": list(map(int, seeds)),
        "algorithms": list(ALGORITHMS), "configurations": chosen,
        "candidate_discovery_limitation": (
            "The extension returns preliminary sizes and pass counters, not a "
            "preliminary-region label volume; candidate-stage spatial identity "
            "is therefore not inferred from unavailable regions."
        ),
    })
    return plan


def _finalise(store):
    frame = pd.DataFrame(store.rows)
    frame.to_csv(OUT / "paired_algorithm_trials.csv", index=False)
    usable = frame[frame.status_category == "ok"].copy()
    metrics = [
        "one_to_one_recovered_cells", "orientation_correct_cells_at_0p02deg",
        "unmatched_true_cells", "unmatched_predictions", "merged_predicted_cells",
        "split_true_cells", "cell_count_error",
        "median_matched_mean_orientation_error_deg", "vi_total_bits", "ari",
        "accepted_marker_count", "marker_cells_unseeded",
        "marker_markers_covering_multiple_cells", "marker_cells_with_multiple_markers",
    ]
    summaries = []
    differences = []
    for (algorithm, pairing), group in usable.groupby(["algorithm_id", "pairing_key"]):
        summaries.append({
            "algorithm_id": algorithm, "pairing_key": pairing,
            **{name: group[name].iloc[0] for name in oc.PARAMETER_NAMES},
            **{f"{metric}_value": group[metric].iloc[0] for metric in metrics if metric in group},
        })
    wide = usable.pivot(index="pairing_key", columns="algorithm_id", values=metrics)
    if all(name in wide.columns.get_level_values(1) for name in ALGORITHMS):
        for pairing in wide.index:
            row = {"pairing_key": pairing}
            for metric in metrics:
                if metric not in wide.columns.get_level_values(0):
                    continue
                row[f"delta_size_minus_random__{metric}"] = (
                    wide.loc[pairing, (metric, ALGORITHMS[1])]
                    - wide.loc[pairing, (metric, ALGORITHMS[0])]
                )
            differences.append(row)
    pd.DataFrame(summaries).to_csv(OUT / "paired_values.csv", index=False)
    pd.DataFrame(differences).to_csv(OUT / "paired_differences.csv", index=False)
    def p05(values):
        return values.quantile(0.05)
    p05.__name__ = "p05"
    seed_summary = usable.groupby(["algorithm_id"] + list(oc.PARAMETER_NAMES))[metrics].agg(
        ["mean", "std", p05, "min"]
    )
    seed_summary.columns = [f"{metric}_{statistic}" for metric, statistic in seed_summary.columns]
    seed_summary.reset_index().to_csv(OUT / "seed_order_stability.csv", index=False)
    root_copy = PRIMARY / "random_order_vs_size_prioritised_paired_summary.csv"
    pd.DataFrame(differences).to_csv(root_copy, index=False)
    _verify_preserved_random_store()


def _verify_preserved_random_store():
    """Recalculate, rather than quote, old random-order seed variability."""
    source = (
        HERE / "continuation_results" / "flood_fill_random_order"
        / "expanded_sensitivity_trials.jsonl"
    )
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame([row for row in rows if row.get("status") == "ok"])
    metrics = [
        name for name in (
            "one_to_one_recovered_cells", "orientation_correct_cells_at_0p02deg",
            "unrepresented_true_cells", "merged_predicted_cells", "split_true_cells",
            "cell_count_error", "median_matched_mean_orientation_error_deg",
            "vi_total_bits", "ari",
        ) if name in frame
    ]
    repeated = frame.groupby("config_key").filter(lambda group: group.random_seed.nunique() > 1)
    if repeated.empty:
        summary = pd.DataFrame(columns=["config_key", "n_seed_orders"])
    else:
        summary = repeated.groupby("config_key")[metrics].agg(["mean", "std", "min", "max"])
        summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
        summary.insert(0, "n_seed_orders", repeated.groupby("config_key").random_seed.nunique())
        summary = summary.reset_index()
    summary.to_csv(OUT / "preserved_random_order_seed_stability_verified.csv", index=False)
    _atomic_json(OUT / "preserved_random_order_integrity.json", {
        "source": str(source), "records": len(rows),
        "unique_config_seed_pairs": int(frame[["config_key", "random_seed"]].drop_duplicates().shape[0]),
        "duplicate_config_seed_pairs": int(
            len(frame) - frame[["config_key", "random_seed"]].drop_duplicates().shape[0]
        ),
        "repeated_configurations": int(summary.shape[0]),
    })


def run(workers=8, n_configurations=32, n_seeds=21):
    global IDENTITY, STOP
    OUT.mkdir(parents=True, exist_ok=True)
    IDENTITY = json.loads((PRIMARY / "implementation_audit.json").read_text())
    lock = (OUT / "paired_comparison.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    _atomic_json(OUT / "pid.json", {"pid": os.getpid(), "workers": workers, "started": utc_now()})
    store = ResultStore(OUT / "paired_trials.jsonl")
    plan = build_plan(n_configurations, range(n_seeds))
    pending = []
    for task in plan:
        digest, _ = _identity(task["algorithm"], task["parameters"], task["seed"])
        if digest not in store.hashes:
            pending.append(task)
    initial = len(store.rows)
    available, total, swap_baseline = _memory()
    started = time.monotonic()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_init_worker,
        max_tasks_per_child=15,
    ) as pool:
        futures = {}
        while pending or futures:
            available, total, swap = _memory()
            aggregate, maximum = _tree_rss()
            swap_growth = max(0, swap - swap_baseline)
            unsafe = (
                aggregate >= 10 * 1024**3
                or available < max(6 * 1024**3, 0.25 * total)
                or swap_growth > 256 * 1024**2
            )
            if aggregate >= 12 * 1024**3 or unsafe:
                STOP = True
            while pending and len(futures) < workers and not STOP:
                task = pending.pop(0)
                futures[pool.submit(_run, task)] = task
            if not futures:
                break
            ready, _ = wait(futures, timeout=2, return_when=FIRST_COMPLETED)
            for future in ready:
                task = futures.pop(future)
                row = future.result()
                digest, identity = _identity(task["algorithm"], task["parameters"], task["seed"])
                if row.get("config_hash") != digest:
                    row = normalize_result({
                        **row, "config_hash": digest, "status": "error",
                        "status_category": "coordinator/schema_error",
                        "error_type": "SchemaError",
                        "error_message": "paired configuration identity mismatch",
                    }, identity)
                store.append(row, identity)
            elapsed = max(time.monotonic() - started, 1e-9)
            completed = len(store.rows) - initial
            _atomic_json(OUT / "progress.json", {
                "completed": len(store.rows), "completed_this_run": completed,
                "pending": len(pending), "in_flight": len(futures),
                "status_counts": store.counts(), "worker_count": workers,
                "trials_per_second": completed / elapsed,
                "aggregate_rss_bytes": aggregate, "maximum_process_rss_bytes": maximum,
                "available_ram_bytes": available, "swap_baseline_bytes": swap_baseline,
                "swap_growth_bytes": swap_growth, "safety_stop": STOP,
            })
    _finalise(store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--configurations", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=21)
    arguments = parser.parse_args()
    run(arguments.workers, arguments.configurations, arguments.seeds)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resumable sequential continuation of the completed fixed-phantom oracle.

This driver never calls ``oracle_search.py`` and never writes its store.  It
derives the remaining primary diagnostics from that immutable store, then runs
small independent strain/difficulty phantoms.  Each ordinary trial is a fresh
spawned process and is fsynced to a per-phantom JSONL store immediately.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import oracle_core as oc
import oracle_metrics as om
import oracle_runner
import oracle_select as osel
from oracle_store import Store
from phantom import PhantomConfig, generate_phantom, intradomain_angular_spread_deg

PRIMARY = HERE / "oracle_results"
DEFAULT_OUT = HERE / "continuation_results"
STRAINS = {
    "2p4": (2.4, 5.02, 2.91, 1.60, 0.14),
    "3p5": (3.5, 4.87, 2.60, 1.59, 0.22),
    "4p6": (4.6, 4.59, 2.29, 1.72, 0.28),
    "6p2": (6.2, 4.20, 2.20, 1.70, 0.36),
}
DIFFICULTIES = {
    "clean": (0.0010, 0.002, 0.004),
    "intermediate": (0.0045, 0.014, 0.012),
    "difficult": (0.0090, 0.038, 0.020),
}
SHAPE = (12, 80, 80)
SPACING = (1.0, 0.4, 0.4)
PRIMARY_PARAMS = {
    "local_threshold_deg": 0.010055,
    "global_threshold_deg": 1.574,
    "footprint_tolerance": 0.1837,
    "footprint_radius_um": 1.391,
    "min_cell_size": 134,
    "kam_radius_um": 1.232,
}


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(payload), indent=2, sort_keys=True))
    temporary.replace(path)


def parse_deadline(text: str) -> float:
    value = dt.datetime.fromisoformat(text)
    if value.tzinfo is None:
        value = value.astimezone()
    return value.timestamp()


def primary_diagnostic(out: Path) -> dict:
    store = Store(PRIMARY / "evaluations.jsonl")
    frame = osel.aggregate(store.ok_rows())
    selected = json.loads((PRIMARY / "selected_solutions.json").read_text())
    rows = []
    for solution, values in selected.items():
        key = values["config_key"]
        trials = [r for r in store.ok_rows() if r["config_key"] == key]
        for row in trials:
            rows.append({
                "solution": solution,
                "seed": row["random_seed"],
                "marker_count": row.get("marker_marker_count"),
                "watershed_count": row.get("n_cells_pred"),
                "marker_unseeded_true_cells": row.get("marker_cells_unseeded"),
                "marker_split_true_cells": row.get("marker_cells_with_multiple_markers"),
                "marker_leaked_across_true_cells": row.get("marker_markers_covering_multiple_cells"),
                "marker_leaked_volume_fraction": row.get("marker_percolating_marker_volume_fraction"),
                "marker_false_internal_boundary_fraction": row.get("marker_false_internal_boundary_fraction"),
                "marker_interface_precision": row.get("marker_interface_precision"),
                "watershed_split_true_cells": row.get("true_cells_split"),
                "watershed_excess_fragments": row.get("excess_fragments_total"),
                "watershed_merging_predictions": row.get("pred_cells_merging"),
                "watershed_unrepresented_true_cells": row.get("true_cells_unrepresented"),
                "watershed_false_internal_boundary_fraction": row.get("false_internal_boundary_fraction"),
                "watershed_interface_precision": row.get("interface_precision"),
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "primary_marker_vs_watershed_trials.csv", index=False)
    summary = detail.groupby("solution").agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary.to_csv(out / "primary_marker_vs_watershed_summary.csv", index=False)
    analysis = json.loads((PRIMARY / "analysis_summary.json").read_text())
    importance = pd.read_csv(PRIMARY / "surrogate_importance.csv")
    accuracy = analysis["surrogate"]
    classifications = []
    for parameter in osel.PARAMETER_NAMES:
        values = importance[importance.parameter == parameter]
        mean_importance = float(values.importance.mean()) if len(values) else 0.0
        interval = analysis["sensitivity"]["intervals"].get(parameter, {})
        if parameter == "global_threshold_deg":
            label = "interacting"
        elif mean_importance < 0.005:
            label = "inactive"
        elif interval.get("n_unique", 0) > 100:
            label = "plateaued"
        else:
            label = "constrained"
        classifications.append({"parameter": parameter, "classification": label,
                                "mean_permutation_importance": mean_importance})
    pd.DataFrame(classifications).to_csv(
        out / "primary_parameter_classification.csv", index=False
    )
    payload = {
        "source_rows": len(store), "unique_configurations": len(frame),
        "solutions": list(selected), "surrogate_accuracy": accuracy,
        "note": "Marker fields are identification-stage; unprefixed fields are watershed output.",
    }
    atomic_json(out / "primary_optimisation_summary.json", payload)
    return payload


def phantom_spec(strain_key: str, realization: int, difficulty: str) -> tuple[str, PhantomConfig, dict]:
    strain, mean_d, sd_d, chi_k, chi_scale = STRAINS[strain_key]
    volume = float(np.prod(np.asarray(SHAPE) * np.asarray(SPACING)))
    sphere_volume = math.pi * mean_d**3 / 6.0
    n_cells = max(24, int(round(volume / sphere_volume)))
    # Diameter-to-volume conversion is an explicit approximation, not a claim
    # that the digitised 2-D diameter distribution is a 3-D volume distribution.
    sigma_log_d = math.sqrt(math.log1p((sd_d / mean_d) ** 2))
    log_volume_sigma = min(3.0 * sigma_log_d, 1.15)
    gradient, curvature, drift = DIFFICULTIES[difficulty]
    seed = 2026081300 + 100 * list(STRAINS).index(strain_key) + 10 * realization + list(DIFFICULTIES).index(difficulty)
    config = PhantomConfig(
        shape_zyx=SHAPE, spacing_um_zyx=SPACING, n_cells=n_cells,
        log_volume_sigma=log_volume_sigma, misorientation_k=chi_k,
        misorientation_sigma_deg=chi_scale,
        intracell_gradient_deg_per_um=gradient,
        intracell_curvature_deg=curvature, drift_deg=drift,
        noise_sigma_deg=0.001, seed=seed,
    )
    phantom_id = f"strain_{strain_key}_r{realization}_{difficulty}"
    meta = {"phantom_id": phantom_id, "strain_percent": strain,
            "target_mean_cell_diameter_um": mean_d,
            "target_cell_diameter_sd_um": sd_d, "chi_k": chi_k,
            "chi_scale_deg": chi_scale, "difficulty": difficulty,
            "realization": realization, "cell_count": n_cells,
            "volume_um3": volume, "diameter_to_volume_assumption": "equivalent-sphere approximation",
            "config": asdict(config)}
    return phantom_id, config, meta


def measurable_proxies(field: np.ndarray, spacing) -> dict:
    flat = field.reshape(-1, field.shape[-1])
    centre = np.median(flat, axis=0)
    total = np.linalg.norm(flat - centre, axis=1)
    differences = []
    for axis in range(3):
        differences.append(np.linalg.norm(np.diff(field, axis=axis), axis=-1).ravel())
    adjacent = np.concatenate(differences)
    q50, q75, q90, q95 = np.percentile(adjacent, [50, 75, 90, 95])
    voxel_volume = float(np.prod(spacing))
    physical_volume = float(np.prod(field.shape[:3]) * voxel_volume)
    # Separate the low-gradient interior mode from interfaces in log space.
    # The former implementation counted values above their own 90th percentile;
    # that selects (almost) exactly 10% of faces for every continuous field and
    # therefore reduced to a fixed field-of-view scale.  Otsu's threshold is
    # data adaptive and remains label-free.  Face areas are axis-specific.
    positive = adjacent[adjacent > 0]
    if positive.size:
        from skimage.filters import threshold_otsu
        log_adjacent = np.log10(np.maximum(positive, np.finfo(float).tiny))
        interface_threshold = float(10.0 ** threshold_otsu(log_adjacent))
    else:
        interface_threshold = float("inf")
    boundary_area_proxy = 0.0
    for axis in range(3):
        delta = np.linalg.norm(np.diff(field, axis=axis), axis=-1)
        face_area = voxel_volume / float(spacing[axis])
        boundary_area_proxy += float(np.count_nonzero(delta >= interface_threshold)) * face_area
    scale_proxy = 6.0 * physical_volume / max(boundary_area_proxy, 1e-9)
    return {"nn_diff_median_deg": float(q50), "nn_diff_iqr_deg": float(q75-q50),
            "nn_diff_p90_deg": float(q90), "nn_diff_p95_deg": float(q95),
            "total_spread_p95_deg": float(np.percentile(total, 95)),
            "cell_scale_proxy_um": float(scale_proxy),
            "cell_interface_threshold_deg": interface_threshold,
            "boundary_width_proxy_um": float(np.mean(spacing) * q90 / max(q95-q50, 1e-9)),
            "voxel_volume_um3": voxel_volume,
            "anisotropy": float(max(spacing) / min(spacing))}


def ensure_phantom(path: Path, config: PhantomConfig, meta: dict) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "manifest.json"
    expected_hash = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    phantom_path = path / "cache" / "phantom.npz"
    if manifest_path.exists() and phantom_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_hash") == expected_hash:
            return manifest
        raise RuntimeError(f"validated checkpoint mismatch at {path}")
    phantom = generate_phantom(config)
    phantom_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(phantom_path, labels=phantom.labels,
                        latent=phantom.latent_field, field=phantom.field,
                        wall_width_um=phantom.wall_width_um,
                        spacing=np.asarray(phantom.spacing_um_zyx))
    latent_means = np.stack([
        np.bincount(phantom.labels.ravel(), weights=phantom.latent_field[..., c].ravel(),
                    minlength=config.n_cells + 1)[1:] /
        np.maximum(np.bincount(phantom.labels.ravel(), minlength=config.n_cells + 1)[1:], 1)
        for c in range(2)
    ], axis=1)
    pairs = []
    for axis in range(3):
        one = np.take(phantom.labels, range(phantom.labels.shape[axis]-1), axis=axis)
        two = np.take(phantom.labels, range(1, phantom.labels.shape[axis]), axis=axis)
        mask = one != two
        pairs.extend(zip(one[mask].tolist(), two[mask].tolist()))
    pairs = np.asarray(sorted({tuple(sorted(p)) for p in pairs}), dtype=int)
    delta = np.linalg.norm(latent_means[pairs[:, 0]-1] - latent_means[pairs[:, 1]-1], axis=1)
    spread = intradomain_angular_spread_deg(
        phantom.labels, phantom.latent_field, config.n_cells
    )
    manifest = {**meta, "config_hash": expected_hash,
                "measurable_proxies": measurable_proxies(phantom.latent_field, SPACING),
                "realised_adjacent_latent_mean_delta_mean_deg": float(delta.mean()),
                "realised_adjacent_latent_mean_delta_median_deg": float(np.median(delta)),
                "intradomain_s_k_median_deg": float(np.median(spread)),
                "intradomain_s_k_p95_deg": float(np.percentile(spread, 95)),
                "experimental_s_k_comparison_deg": 0.184}
    atomic_json(manifest_path, manifest)
    del phantom, latent_means, pairs, delta, spread
    gc.collect()
    return manifest


def candidate_configs(meta: dict, workspace: oc.Workspace, n: int) -> list[dict]:
    from scipy.stats import qmc
    strain_scale = meta["chi_scale_deg"] / 0.36
    size_scale = meta["target_mean_cell_diameter_um"] / 4.2
    base = dict(PRIMARY_PARAMS)
    base["local_threshold_deg"] *= strain_scale
    base["global_threshold_deg"] *= strain_scale
    base["footprint_radius_um"] *= size_scale
    base["kam_radius_um"] *= size_scale
    base["min_cell_size"] = max(5, int(round(base["min_cell_size"] * size_scale**3)))
    design = qmc.LatinHypercube(6, seed=int(meta["config"]["seed"])).random(n)
    configs = []
    for point in design:
        factors = np.exp((point - 0.5) * np.log([6, 6, 3, 3, 8, 3]))
        raw = dict(base)
        raw["local_threshold_deg"] *= factors[0]
        raw["global_threshold_deg"] *= factors[1]
        raw["footprint_tolerance"] = float(np.clip(base["footprint_tolerance"] * factors[2], .02, .8))
        raw["footprint_radius_um"] = float(np.clip(base["footprint_radius_um"] * factors[3], .45, 2.6))
        raw["min_cell_size"] = max(5, min(400, int(round(base["min_cell_size"] * factors[4]))))
        raw["kam_radius_um"] = float(np.clip(base["kam_radius_um"] * factors[5], .42, 2.6))
        configs.append(oc.canonical(oc.Config(**raw), workspace).as_dict())
    configs.append(oc.canonical(oc.Config(**base), workspace).as_dict())
    unique = {oc.Config(**c).key(): c for c in configs}
    return list(unique.values())


def run_phantom(path: Path, meta: dict, broad_trials: int, deadline: float) -> None:
    workspace = oc.load_workspace(path / "cache")
    store = Store(path / "evaluations.jsonl")
    configs = candidate_configs(meta, workspace, broad_trials)
    pending = [(c, 0) for c in configs if not store.has(oc.Config(**c).key(), 0)]
    os.environ["DISELL_DEADLINE_EPOCH"] = str(deadline)
    for row in oracle_runner.evaluate_batch(
            pending, path / "cache", processes=1, progress_every=1,
            label=f"{meta['phantom_id']} "):
        row.update(stage="broad", phantom_id=meta["phantom_id"])
        store.append([row])
        atomic_json(path / "progress.json", {"stored_trials": len(store),
                    "requested_broad": len(configs), "updated": time.time(),
                    "last_peak_rss_bytes": row.get("peak_rss_bytes"),
                    "last_swap_used_bytes": row.get("swap_used_bytes_after")})
    aggregate = osel.aggregate(store.ok_rows())
    if aggregate.empty:
        return
    refinement_plan_path = path / "refinement_plan.json"
    if refinement_plan_path.exists():
        refinement_configs = json.loads(refinement_plan_path.read_text())["configs"]
    else:
        centre = osel.select_solutions(aggregate)["solutions"]["balanced"]
        from scipy.stats import qmc
        refinement_configs = []
        local = qmc.LatinHypercube(6, seed=int(meta["config"]["seed"]) + 991).random(12)
        for point in local:
            factors = np.exp((point - 0.5) * np.log([1.8, 1.8, 1.6, 1.6, 2.5, 1.6]))
            raw = {name: centre[name] for name in osel.PARAMETER_NAMES}
            raw["local_threshold_deg"] *= factors[0]
            raw["global_threshold_deg"] *= factors[1]
            raw["footprint_tolerance"] = float(np.clip(raw["footprint_tolerance"] * factors[2], .02, .8))
            raw["footprint_radius_um"] = float(np.clip(raw["footprint_radius_um"] * factors[3], .45, 2.6))
            raw["min_cell_size"] = max(5, min(400, int(round(raw["min_cell_size"] * factors[4]))))
            raw["kam_radius_um"] = float(np.clip(raw["kam_radius_um"] * factors[5], .42, 2.6))
            refinement_configs.append(
                oc.canonical(oc.Config(**raw), workspace).as_dict()
            )
        atomic_json(refinement_plan_path, {"configs": refinement_configs})
    refinement = [(params, 0) for params in refinement_configs
                  if not store.has(oc.Config(**params).key(), 0)]
    for row in oracle_runner.evaluate_batch(
            refinement, path / "cache", processes=1, progress_every=1,
            label=f"{meta['phantom_id']} refine "):
        row.update(stage="refine", phantom_id=meta["phantom_id"])
        store.append([row])
        atomic_json(path / "progress.json", {"stored_trials": len(store),
                    "updated": time.time(), "last_peak_rss_bytes": row.get("peak_rss_bytes"),
                    "last_swap_used_bytes": row.get("swap_used_bytes_after")})
    aggregate = osel.aggregate(store.ok_rows())
    # Repeat the best representatives for seed-order sensitivity. Existing
    # hashes are skipped, and ordinary labels are never returned by workers.
    repeat_plan_path = path / "repeat_plan.json"
    if repeat_plan_path.exists():
        chosen_configs = json.loads(repeat_plan_path.read_text())["configs"]
    else:
        chosen = pd.concat([
            aggregate.nlargest(2, "ari_mean"), aggregate.nsmallest(2, "vi_total_bits_mean"),
            aggregate.nsmallest(2, "abs_cell_count_error")
        ]).drop_duplicates("config_key")
        chosen_configs = [
            {name: int(choice[name]) if name == "min_cell_size" else float(choice[name])
             for name in osel.PARAMETER_NAMES}
            for _, choice in chosen.iterrows()
        ]
        atomic_json(repeat_plan_path, {"configs": chosen_configs, "seeds": [1, 2]})
    tasks = []
    for params in chosen_configs:
        for seed in (1, 2):
            if not store.has(oc.Config(**params).key(), seed):
                tasks.append((params, seed))
    for row in oracle_runner.evaluate_batch(
            tasks, path / "cache", processes=1, progress_every=1,
            label=f"{meta['phantom_id']} repeat "):
        row.update(stage="repeat", phantom_id=meta["phantom_id"])
        store.append([row])
        atomic_json(path / "progress.json", {"stored_trials": len(store),
                    "updated": time.time(), "last_peak_rss_bytes": row.get("peak_rss_bytes"),
                    "last_swap_used_bytes": row.get("swap_used_bytes_after")})


def finalise(out: Path) -> None:
    summaries = []
    trial_frames = []
    for manifest_path in sorted((out / "phantoms").glob("*/manifest.json")):
        path = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        store = Store(path / "evaluations.jsonl")
        frame = osel.aggregate(store.ok_rows())
        if frame.empty:
            continue
        selected = osel.select_solutions(frame)
        balanced = selected["solutions"]["balanced"]
        summaries.append({**{k: manifest[k] for k in (
            "phantom_id", "strain_percent", "difficulty", "realization", "cell_count")},
            **manifest["measurable_proxies"],
            "intradomain_s_k_median_deg": manifest["intradomain_s_k_median_deg"],
            **{f"oracle_{k}": balanced[k] for k in osel.PARAMETER_NAMES},
            "oracle_ari": balanced["ari_mean"], "oracle_vi_bits": balanced["vi_total_bits_mean"],
            "oracle_n_cells_pred": balanced["n_cells_pred_mean"],
            "oracle_boundary_f1": balanced.get("boundary_f1_at_0p4um_mean")})
        raw = pd.DataFrame(store.ok_rows())
        raw["phantom_id"] = manifest["phantom_id"]
        trial_frames.append(raw)
        # Full labels only for the selected balanced finalist.
        labels_path = path / "balanced_labels.npz"
        if not labels_path.exists():
            workspace = oc.load_workspace(path / "cache")
            params = {name: int(balanced[name]) if name == "min_cell_size" else float(balanced[name])
                      for name in osel.PARAMETER_NAMES}
            labels, markers = oc.segment(oc.Config(**params), workspace, 0)
            np.savez_compressed(labels_path, labels=labels, markers=markers)
            del labels, markers, workspace
            gc.collect()
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "multi_phantom_summary.csv", index=False)
    if trial_frames:
        pd.concat(trial_frames, ignore_index=True).to_csv(out / "scalar_trials.csv", index=False)
    if len(summary) >= 4:
        fit_rules(summary, out)
        make_plots(summary, out)
    status = {"updated": dt.datetime.now().astimezone().isoformat(),
              "phantoms_complete": len(summary), "phantoms_planned": 36,
              "primary_search_preserved": True,
              "full_tests_deferred_until_detached_run_finishes": True}
    atomic_json(out / "status.json", status)
    (out / "OVERNIGHT_STATUS.md").write_text(
        "# Overnight continuation status\n\n"
        f"Updated: {status['updated']}\n\n"
        f"Completed multi-phantom analyses: {len(summary)} / 36.\n\n"
        "The fixed-primary oracle search was reused and was not rerun. "
        "See `continue.log`, `status.json`, and each phantom `progress.json`.\n"
    )


def fit_rules(frame: pd.DataFrame, out: Path) -> None:
    from sklearn.linear_model import Ridge
    predictors = ["nn_diff_median_deg", "nn_diff_iqr_deg", "nn_diff_p90_deg",
                  "total_spread_p95_deg", "cell_scale_proxy_um",
                  "boundary_width_proxy_um", "voxel_volume_um3", "anisotropy"]
    targets = [f"oracle_{p}" for p in osel.PARAMETER_NAMES if p != "footprint_tolerance"]
    rows = []
    x = np.log(np.maximum(frame[predictors].to_numpy(float), 1e-9))
    for target in targets:
        y = np.log(np.maximum(frame[target].to_numpy(float), 1e-9))
        for scheme, groups in (("leave_one_strain_out", frame.strain_percent),
                               ("leave_one_phantom_out", frame.phantom_id)):
            for group in pd.unique(groups):
                test = np.asarray(groups == group)
                if test.sum() == 0 or (~test).sum() < 3:
                    continue
                model = Ridge(alpha=1.0).fit(x[~test], y[~test])
                prediction = np.exp(model.predict(x[test]))
                for index, predicted in zip(frame.index[test], prediction):
                    rows.append({"scheme": scheme, "held_out_group": group,
                                 "phantom_id": frame.loc[index, "phantom_id"],
                                 "parameter": target.removeprefix("oracle_"),
                                 "oracle_value": float(frame.loc[index, target]),
                                 "rule_value": float(predicted),
                                 "absolute_log_error": abs(float(np.log(predicted) - y[index]))})
    pd.DataFrame(rows).to_csv(out / "transferable_rule_cross_validation.csv", index=False)


def make_plots(frame: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), layout="constrained")
    for ax, parameter in zip(axes.ravel(), osel.PARAMETER_NAMES):
        for difficulty, subset in frame.groupby("difficulty"):
            ax.scatter(subset.strain_percent, subset[f"oracle_{parameter}"], s=12,
                       label=difficulty)
        ax.set(xlabel="strain (%)", ylabel=parameter)
    axes[0, 0].legend(fontsize=6)
    fig.savefig(out / "parameters_vs_strain.png", dpi=250)
    fig.savefig(out / "parameters_vs_strain.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--trials-per-phantom", type=int, default=40)
    parser.add_argument("--finalise-reserve-minutes", type=float, default=35.0)
    parser.add_argument("--max-phantoms", type=int, default=None,
                        help="Testing/debug limit; omitted for the full suite")
    args = parser.parse_args()
    deadline = parse_deadline(args.deadline)
    compute_deadline = deadline - args.finalise_reserve_minutes * 60.0
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_stream = (out / "overnight_continue.lock").open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another continuation driver holds the lock")
    process = {"pid": os.getpid(), "ppid": os.getppid(), "sid": os.getsid(0),
               "started": dt.datetime.now().astimezone().isoformat(),
               "deadline": args.deadline, "compute_deadline_epoch": compute_deadline,
               "stdout": str(Path(f"/proc/{os.getpid()}/fd/1").resolve()),
               "stderr": str(Path(f"/proc/{os.getpid()}/fd/2").resolve())}
    atomic_json(out / "overnight_continue.pid.json", process)
    print(json.dumps(process, indent=2), flush=True)
    primary_diagnostic(out)
    atomic_json(out / "plan.json", {"strains": STRAINS, "difficulties": DIFFICULTIES,
                "realisations": [0, 1, 2], "trials_per_phantom": args.trials_per_phantom,
                "strictly_sequential": True})
    completed_phantoms = 0
    for strain_key in STRAINS:
        for realization in range(3):
            for difficulty in DIFFICULTIES:
                if args.max_phantoms is not None and completed_phantoms >= args.max_phantoms:
                    finalise(out)
                    return 0
                if time.time() >= compute_deadline:
                    print("finalisation reserve reached", flush=True)
                    finalise(out)
                    return 0
                phantom_id, config, meta = phantom_spec(strain_key, realization, difficulty)
                path = out / "phantoms" / phantom_id
                manifest = ensure_phantom(path, config, meta)
                print(f"starting {phantom_id}", flush=True)
                run_phantom(path, manifest, args.trials_per_phantom, compute_deadline)
                finalise(out)
                completed_phantoms += 1
    finalise(out)
    print("continuation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

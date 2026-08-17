#!/usr/bin/env python3
"""Finalist export, auditable selection, sensitivity analysis, and figures.

This command never changes the phantom or the search store.  It can be rerun
after any search batch; outputs are replaced atomically or are deterministic
derived products.  Full volumes are saved only for the four reported solutions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figures
import oracle_core as oc
import oracle_metrics as om
import oracle_select as osel
from oracle_store import Store


def _plain(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_plain(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def surrogate(frame: pd.DataFrame, out: Path, seed: int = 20260812) -> dict:
    """Cross-validated random-forest importance and held-out predictions."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    work = frame.copy().replace([np.inf, -np.inf], np.nan)
    work["global_regime"] = np.where(work.global_threshold_deg > 0, "active", "off")
    features = list(osel.PARAMETER_NAMES) + ["global_regime"]
    categorical = ["global_regime"]
    numeric = [x for x in features if x not in categorical]
    pre = ColumnTransformer((("num", "passthrough", numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)))
    n_splits = 5
    minimum_rows = 50
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    reports = {}
    importance_rows = []
    for target in ("ari_mean", "vi_total_bits_mean", "abs_cell_count_error",
                   "boundary_f1_at_0p4um_mean"):
        if target not in work:
            continue
        selected = features + [target]
        retained_mask = work[selected].notna().all(axis=1)
        fit_work = work.loc[retained_mask]
        total = len(work)
        retained = len(fit_work)
        excluded = total - retained
        print(
            f"surrogate {target}: total={total}, retained={retained}, "
            f"excluded={excluded}"
        )
        reports[target] = {
            "total_rows": total,
            "retained_rows": retained,
            "excluded_rows": excluded,
        }
        if retained < n_splits:
            raise ValueError(
                f"surrogate target {target!r} retains {retained} rows, fewer "
                f"than the configured {n_splits}-fold cross-validation"
            )
        if retained < minimum_rows:
            reports[target]["status"] = "skipped_too_few_rows"
            continue
        model = make_pipeline(pre, RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=1, random_state=seed))
        predicted = cross_val_predict(
            model, fit_work[features], fit_work[target], cv=cv
        )
        reports[target].update(
            status="ok",
            r2=float(r2_score(fit_work[target], predicted)),
            mae=float(mean_absolute_error(fit_work[target], predicted)),
        )
        model.fit(fit_work[features], fit_work[target])
        # Permute original columns, so categorical regime remains interpretable.
        pi = permutation_importance(model, fit_work[features], fit_work[target],
                                    n_repeats=8, random_state=seed, n_jobs=1)
        for name, mean, std in zip(features, pi.importances_mean, pi.importances_std):
            importance_rows.append({"target": target, "parameter": name,
                                    "importance": mean, "importance_std": std,
                                    "held_out_r2": reports[target]["r2"]})
        prediction_column = f"{target}_cv_prediction"
        work[prediction_column] = np.nan
        work.loc[retained_mask, prediction_column] = predicted
    pd.DataFrame(importance_rows).to_csv(out / "surrogate_importance.csv", index=False)
    work[["config_key"] + [c for c in work if c.endswith("_cv_prediction")]].to_csv(
        out / "surrogate_predictions.csv", index=False)
    return reports


def sensitivity(frame: pd.DataFrame, balanced: dict, out: Path) -> dict:
    """Empirical OAT profiles, near-optimal intervals and pairwise surfaces."""
    scaled = frame.copy()
    profiles = []
    for parameter in osel.PARAMETER_NAMES:
        others = [p for p in osel.PARAMETER_NAMES if p != parameter]
        distance = np.zeros(len(scaled))
        for other in others:
            values = scaled[other].to_numpy(float)
            span = max(np.nanpercentile(values, 95) - np.nanpercentile(values, 5), 1e-12)
            distance += ((values - float(balanced[other])) / span) ** 2
        nearest = scaled.iloc[np.argsort(distance)[:min(250, len(scaled))]].copy()
        nearest["parameter"] = parameter
        nearest["value"] = nearest[parameter]
        profiles.append(nearest[["parameter", "value", "ari_mean", "vi_total_bits_mean",
                                 "n_cells_pred_mean", "boundary_f1_at_0p4um_mean"]])
    pd.concat(profiles).to_csv(out / "oat_profiles.csv", index=False)
    best = float(frame.ari_mean.max())
    near = frame[frame.ari_mean >= best - osel.ARI_BAND]
    intervals = {p: {"low": float(near[p].min()), "high": float(near[p].max()),
                     "n_unique": int(near[p].nunique())} for p in osel.PARAMETER_NAMES}
    # Surface tables use quantile bins and are auditable independently of plots.
    pair_rows = []
    pairs = (("local_threshold_deg", "min_cell_size"),
             ("footprint_tolerance", "min_cell_size"),
             ("footprint_radius_um", "kam_radius_um"),
             ("local_threshold_deg", "global_threshold_deg"))
    for a, b in pairs:
        work = frame[[a, b, "ari_mean", "vi_total_bits_mean", "n_cells_pred_mean"]].copy()
        work["a_bin"] = pd.qcut(work[a], min(12, work[a].nunique()), duplicates="drop")
        work["b_bin"] = pd.qcut(work[b], min(12, work[b].nunique()), duplicates="drop")
        grouped = work.groupby(["a_bin", "b_bin"], observed=True).mean(numeric_only=True).reset_index()
        for _, row in grouped.iterrows():
            pair_rows.append({"parameter_a": a, "parameter_b": b,
                              "a_low": row.a_bin.left, "a_high": row.a_bin.right,
                              "b_low": row.b_bin.left, "b_high": row.b_bin.right,
                              "ari_mean": row.ari_mean,
                              "vi_total_bits_mean": row.vi_total_bits_mean,
                              "n_cells_pred_mean": row.n_cells_pred_mean})
    pd.DataFrame(pair_rows).to_csv(out / "pairwise_surfaces.csv", index=False)
    return {"near_optimal_ari_band": osel.ARI_BAND, "intervals": intervals,
            "n_near_optimal": len(near)}


def facet_bins(truth, prediction, field, spacing, out_path: Path) -> None:
    table = om.facet_table(truth, prediction, spacing)
    means = np.zeros((int(truth.max()) + 1, field.shape[-1]))
    for label in range(1, means.shape[0]): means[label] = field[truth == label].mean(axis=0)
    angle = np.linalg.norm(means[table["cell_a"]] - means[table["cell_b"]], axis=1)
    recovered = table["cells_separated"]
    bins = np.array([0, .05, .1, .2, .4, np.inf])
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        selected = (angle >= lo) & (angle < hi)
        n, k = int(selected.sum()), int(recovered[selected].sum())
        low, high = om.wilson_interval(k, n)
        rows.append({"low_deg": lo, "high_deg": hi, "n_facets": n,
                     "recovered_facets": k, "recall": k / n if n else np.nan,
                     "wilson_low": low, "wilson_high": high})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plots(frame: pd.DataFrame, profiles: pd.DataFrame, surfaces: pd.DataFrame,
          out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    colour = np.abs(frame.n_cells_pred_mean - 360)
    axes[0].scatter(frame.ari_mean, frame.vi_total_bits_mean, c=colour, s=8, cmap="viridis")
    axes[0].set(xlabel="ARI", ylabel="total VI (bits)")
    axes[1].scatter(frame.n_cells_pred_mean, frame.ari_mean, c=frame.vi_total_bits_mean,
                    s=8, cmap="magma_r")
    axes[1].axvline(360, color="black", lw=.7)
    axes[1].set(xlabel="predicted cells", ylabel="ARI")
    fig.savefig(out / "pareto.png", dpi=250); fig.savefig(out / "pareto.pdf"); plt.close(fig)
    parameters = list(osel.PARAMETER_NAMES)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for ax, p in zip(axes.ravel(), parameters):
        q = profiles[profiles.parameter == p].sort_values("value")
        ax.scatter(q.value, q.ari_mean, s=6); ax.set(xlabel=p, ylabel="ARI")
        if p in ("local_threshold_deg", "global_threshold_deg", "footprint_radius_um", "kam_radius_um"):
            positive = q.value > 0
            if positive.all(): ax.set_xscale("log")
    fig.savefig(out / "sensitivity_profiles.png", dpi=250)
    fig.savefig(out / "sensitivity_profiles.pdf"); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "oracle_results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    store = Store(out / "evaluations.jsonl")
    frame = osel.aggregate(store.ok_rows())
    if frame.empty: raise SystemExit("no completed evaluations")
    selection = osel.select_solutions(frame)
    decision = selection["retained"].copy()
    decision.to_csv(out / "decision_table.csv", index=False)
    selected = selection["solutions"]
    (out / "selected_solutions.json").write_text(json.dumps(_plain(selected), indent=2))
    report = {"n_unique_trials": len(frame), "n_seed_evaluations": len(store),
              "solutions": selected, "surrogate": surrogate(frame, out),
              "sensitivity": sensitivity(frame, selected["balanced"], out)}
    workspace = oc.load_workspace(out / "cache")
    labels_dir = out / "finalist_labels"; labels_dir.mkdir(exist_ok=True)
    made = {}
    for name, row in selected.items():
        config = oc.Config(**{p: int(row[p]) if p == "min_cell_size" else float(row[p])
                              for p in osel.PARAMETER_NAMES})
        labels, markers = oc.segment(config, workspace, args.seed)
        np.savez_compressed(labels_dir / f"{name}.npz", labels=labels, markers=markers)
        made[name] = labels
        facet_bins(workspace.labels, labels, workspace.field, workspace.spacing_um_zyx,
                   out / f"facet_bins_{name}.csv")
    old = np.load(HERE / "results" / "phantom_and_labels.npz")
    figures.render_figure(out / "final_comparison", field=workspace.field,
        kam=workspace.kam(float(selected["balanced"]["kam_radius_um"])),
        truth_labels=workspace.labels, kam_labels=old["kam_threshold_labels"],
        flood_fill_labels=made["balanced"], spacing_um_zyx=workspace.spacing_um_zyx,
        colour_reference_deg=.75)
    profiles = pd.read_csv(out / "oat_profiles.csv")
    surfaces = pd.read_csv(out / "pairwise_surfaces.csv")
    plots(frame, profiles, surfaces, out)
    (out / "analysis_summary.json").write_text(json.dumps(_plain(report), indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Audit continuation outputs and regenerate final tables, figures, and report."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/disell-finalise-mpl")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "continuation_results"
sys.path.insert(0, str(HERE))
import oracle_select as osel
from oracle_store import Store
from overnight_continue import primary_diagnostic


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def interaction_scores(frame: pd.DataFrame):
    pairs = (("local_threshold_deg", "min_cell_size"),
             ("footprint_tolerance", "min_cell_size"),
             ("footprint_radius_um", "kam_radius_um"),
             ("local_threshold_deg", "global_threshold_deg"))
    scores = {p: [] for p in osel.PARAMETER_NAMES}; surfaces = []
    for a, b in pairs:
        work = frame[[a, b, "ari_mean"]].copy()
        work["a_bin"] = pd.qcut(work[a], 8, duplicates="drop")
        work["b_bin"] = pd.qcut(work[b], 8, duplicates="drop")
        group = work.groupby(["a_bin", "b_bin"], observed=True).ari_mean.agg(["mean", "count"]).reset_index()
        group = group[group["count"] >= 5].copy()
        grand = np.average(group["mean"], weights=group["count"])
        am = group.groupby("a_bin", observed=True).apply(
            lambda x: np.average(x["mean"], weights=x["count"]), include_groups=False)
        bm = group.groupby("b_bin", observed=True).apply(
            lambda x: np.average(x["mean"], weights=x["count"]), include_groups=False)
        additive = np.asarray([am[x] + bm[y] - grand for x, y in zip(group.a_bin, group.b_bin)])
        rms = np.sqrt(np.average((group["mean"] - additive) ** 2, weights=group["count"]))
        scale = max(np.percentile(group["mean"], 95) - np.percentile(group["mean"], 5), 1e-12)
        score = float(rms / scale)
        scores[a].append(score); scores[b].append(score)
        for _, row in group.iterrows():
            surfaces.append({"parameter_a": a, "parameter_b": b,
                "a_low": row.a_bin.left, "a_high": row.a_bin.right,
                "b_low": row.b_bin.left, "b_high": row.b_bin.right,
                "ari_mean": row["mean"], "sample_count": int(row["count"]),
                "interaction_strength": score})
    return {p: max(v) if v else None for p, v in scores.items()}, pd.DataFrame(surfaces)


def classify(frame: pd.DataFrame) -> pd.DataFrame:
    profiles = pd.read_csv(HERE / "oracle_results" / "oat_profiles.csv")
    importance = pd.read_csv(HERE / "oracle_results" / "surrogate_importance.csv")
    interactions, surfaces = interaction_scores(frame)
    surfaces.to_csv(OUT / "primary_pairwise_response_surfaces.csv", index=False)
    rows = []
    for p in osel.PARAMETER_NAMES:
        q = profiles[profiles.parameter == p].dropna(subset=["value", "ari_mean"])
        low, high = np.percentile(frame[p], [5, 95]); best = q.ari_mean.max()
        near = q[q.ari_mean >= best - osel.ARI_BAND]
        width = float((near.value.max() - near.value.min()) / max(high-low, 1e-12))
        imp = float(importance.loc[importance.parameter == p, "importance"].mean())
        interaction = interactions[p]
        if width <= 0.10:
            local = "constrained"
        elif width >= 0.25:
            local = "plateaued"
        else:
            local = "intermediate"
        if imp < 0.01 and local == "constrained":
            label = "locally constrained; globally low-importance"
        elif interaction is not None and interaction >= 0.20:
            label = "interaction-dominated"
        else:
            label = local
        repeated = frame[(frame[p] >= near.value.min()) & (frame[p] <= near.value.max()) &
                         (frame.n_seeds >= 5)]
        seed = float(repeated.ari_std.median()) if len(repeated) else None
        rows.append({"parameter": p, "classification": label,
            "global_predictive_importance_mean": imp,
            "local_near_optimal_constraint": local,
            "near_optimal_profile_low": float(near.value.min()),
            "near_optimal_profile_high": float(near.value.max()),
            "plateau_width_fraction": width,
            "interaction_strength_max": interaction,
            "seed_sensitivity_median_ari_std": seed,
            "near_optimal_profile_points": int(len(near))})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "primary_parameter_classification.csv", index=False)
    return result


def validate(multi, trials, cv, rule):
    expected = {f"strain_{s}_r{r}_{d}" for s in ("2p4", "3p5", "4p6", "6p2")
                for r in range(3) for d in ("clean", "intermediate", "difficult")}
    by_stage = trials.groupby(["phantom_id", "stage"]).size().unstack(fill_value=0)
    issues = []
    if set(multi.phantom_id) != expected: issues.append("phantom ID set mismatch")
    required = {
        "multi": list(multi.columns),
        "trials": ["phantom_id", "config_key", "random_seed", "stage", "ari",
                   "vi_total_bits", "n_cells_pred"],
        "cv": list(cv.columns),
        "rule": ["phantom_id", "scheme", "ari", "vi_total_bits", "n_cells_pred",
                 "boundary_f1_at_0p4um"],
    }
    for name, table in (("multi", multi), ("trials", trials), ("cv", cv), ("rule", rule)):
        selected = table[required[name]]
        if selected.isna().any().any(): issues.append(f"{name} contains missing required values")
        numeric = selected.select_dtypes(include=np.number)
        if not np.isfinite(numeric.to_numpy()).all(): issues.append(f"{name} contains non-finite values")
    if not ((by_stage.get("broad", 0) >= 30).all() and
            (by_stage.get("refine", 0) == 12).all() and
            (by_stage.get("repeat", 0) >= 4).all()):
        issues.append("incomplete per-phantom oracle/repeat stages")
    if len(cv) != 360 or len(rule) != 72: issues.append("incomplete cross-validation rows")
    return {"passed": not issues, "issues": issues, "expected_phantoms": 36,
            "observed_phantoms": int(multi.phantom_id.nunique()),
            "balance": {f"{strain:g}|{difficulty}": int(count) for (strain, difficulty), count
                        in multi.groupby(["strain_percent", "difficulty"]).size().items()},
            "trial_count_range": [int(by_stage.sum(axis=1).min()), int(by_stage.sum(axis=1).max())],
            "undefined_optional_values": {
                "scalar_trials.boundary_f1_at_0p4um": int(trials.boundary_f1_at_0p4um.isna().sum()),
                "reason": "mathematically undefined for ten failed broad candidates predicting one region and therefore no boundary; their ARI, VI, and count remain finite"
            },
            "predictors": ["nn_diff_median_deg", "nn_diff_iqr_deg", "nn_diff_p90_deg",
                "total_spread_p95_deg", "cell_scale_proxy_um", "boundary_width_proxy_um",
                "voxel_volume_um3", "anisotropy"],
            "predictor_leakage": "none: all predictors derive from the angular field or acquisition geometry; no truth labels, target strain, difficulty, or realisation are inputs",
            "grouping": {"leave_one_strain_out": "all nine phantoms at one strain held out together",
                         "leave_one_phantom_out": "one complete phantom held out"}}


def plots(frame, profiles, diagnostics, multi, rule):
    import matplotlib.pyplot as plt
    # Preserve the original primary Pareto rendering in the continuation bundle.
    for suffix in ("png", "pdf"):
        shutil.copyfile(HERE / "oracle_results" / f"pareto.{suffix}", OUT / f"primary_pareto_front.{suffix}")
        shutil.copyfile(HERE / "oracle_results" / f"sensitivity_profiles.{suffix}", OUT / f"primary_oat_profiles.{suffix}")
    surfaces = pd.read_csv(OUT / "primary_pairwise_response_surfaces.csv")
    pairs = surfaces[["parameter_a", "parameter_b"]].drop_duplicates().itertuples(index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for ax, (a, b) in zip(axes.ravel(), pairs):
        q = surfaces[(surfaces.parameter_a == a) & (surfaces.parameter_b == b)]
        sc = ax.scatter((q.a_low+q.a_high)/2, (q.b_low+q.b_high)/2,
                        c=q.ari_mean, s=np.clip(q.sample_count, 10, 220), cmap="viridis")
        ax.set(xlabel=a, ylabel=b, title=f"min n/bin={q.sample_count.min()}")
        fig.colorbar(sc, ax=ax, label="mean ARI")
    fig.savefig(OUT / "primary_pairwise_response_surfaces.png", dpi=220)
    fig.savefig(OUT / "primary_pairwise_response_surfaces.pdf"); plt.close(fig)
    means = diagnostics.groupby("solution").mean(numeric_only=True)
    fields = [("marker_false_internal_boundary_fraction", "watershed_false_internal_boundary_fraction", "false internal boundary"),
              ("marker_interface_precision", "watershed_interface_precision", "interface precision"),
              ("marker_split_true_cells", "watershed_split_true_cells", "split true cells"),
              ("marker_unseeded_true_cells", "watershed_unrepresented_true_cells", "unrepresented true cells")]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    x=np.arange(len(means)); w=.36
    for ax,(a,b,title) in zip(axes.ravel(),fields):
        ax.bar(x-w/2,means[a],w,label="markers"); ax.bar(x+w/2,means[b],w,label="watershed")
        ax.set(xticks=x,xticklabels=means.index, title=title); ax.tick_params(axis='x',rotation=25)
    axes[0,0].legend(); fig.savefig(OUT/'marker_vs_watershed.png',dpi=220); fig.savefig(OUT/'marker_vs_watershed.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for difficulty,q in multi.groupby('difficulty'):
        z=q.groupby('strain_percent').oracle_ari.agg(['mean','std'])
        axes[0].errorbar(z.index,z['mean'],yerr=z['std'],marker='o',label=difficulty)
    for strain,q in multi.groupby('strain_percent'):
        z=q.groupby('difficulty').oracle_ari.agg(['mean','std']).reindex(['clean','intermediate','difficult'])
        axes[1].errorbar(z.index,z['mean'],yerr=z['std'],marker='o',label=f'{strain}%')
    axes[0].set(xlabel='strain (%)',ylabel='oracle ARI'); axes[1].set(xlabel='difficulty',ylabel='oracle ARI'); axes[0].legend(); axes[1].legend(fontsize=7)
    fig.savefig(OUT/'performance_vs_strain_difficulty.png',dpi=220); fig.savefig(OUT/'performance_vs_strain_difficulty.pdf'); plt.close(fig)
    joined=rule.merge(multi[['phantom_id','oracle_ari']],on='phantom_id')
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,(scheme,q) in zip(axes,joined.groupby('scheme')):
        ax.scatter(q.oracle_ari,q.ari,c=q.cell_count_error.abs(),cmap='magma',s=22); lo=min(q.oracle_ari.min(),q.ari.min()); ax.plot([lo,1],[lo,1],'k--',lw=.7)
        ax.set(xlabel='small-search oracle ARI',ylabel='held-out rule ARI',title=scheme.replace('_',' '))
    fig.savefig(OUT/'oracle_vs_transferable_rule.png',dpi=220); fig.savefig(OUT/'oracle_vs_transferable_rule.pdf'); plt.close(fig)


def main() -> int:
    primary_diagnostic(OUT)
    store=Store(HERE/'oracle_results'/'evaluations.jsonl'); frame=osel.aggregate(store.ok_rows())
    classes=classify(frame)
    multi=pd.read_csv(OUT/'multi_phantom_summary.csv'); trials=pd.read_csv(OUT/'scalar_trials.csv')
    cv=pd.read_csv(OUT/'transferable_rule_cross_validation.csv'); rule=pd.read_csv(OUT/'transferable_rule_performance.csv')
    diagnostic=pd.read_csv(OUT/'primary_marker_vs_watershed_trials.csv')
    audit=validate(multi,trials,cv,rule)
    dump(OUT/'continuation_audit.json', audit)
    plots(frame,pd.read_csv(HERE/'oracle_results'/'oat_profiles.csv'),diagnostic,multi,rule)
    perf=multi.groupby(['strain_percent','difficulty'])[['oracle_ari','oracle_vi_bits','oracle_boundary_f1']].agg(['mean','std'])
    cvperf=[]
    for scheme,q in rule.merge(multi[['phantom_id','oracle_ari','oracle_vi_bits','oracle_boundary_f1']],on='phantom_id').groupby('scheme'):
        cvperf.append({'scheme':scheme,'rule_ari_mean':q.ari.mean(),'oracle_ari_mean':q.oracle_ari.mean(),
            'ari_degradation_mean':(q.oracle_ari-q.ari).mean(),'vi_change_mean':(q.vi_total_bits-q.oracle_vi_bits).mean(),
            'boundary_f1_change_mean':(q.boundary_f1_at_0p4um-q.oracle_boundary_f1).mean()})
    report={'created':dt.datetime.now().astimezone().isoformat(),'audit':audit,
            'classification_rule':{'near_optimal':'within 0.002 ARI of each OAT-profile maximum',
              'constrained':'near-optimal width <=10% of the parameter 5th-to-95th percentile search span',
              'plateaued':'width >=25%; otherwise intermediate',
              'globally_low_importance':'mean permutation importance <0.01',
              'interaction_dominated':'maximum sampled-pair additive-residual RMS / 5th-to-95th ARI range >=0.20',
              'surface_support':'8x8 quantile bins; bins with fewer than five configurations omitted',
              'seed_sensitivity':'median ARI standard deviation among >=5-seed configurations inside the OAT near-optimal interval'},
            'classifications':classes.replace({np.nan:None}).to_dict('records'),
            'rule_performance':cvperf}
    dump(OUT/'finalisation_report.json',report)
    print(json.dumps({'audit':audit,'rule_performance':cvperf},indent=2,default=str))
    return 0 if audit['passed'] else 1

if __name__=='__main__': raise SystemExit(main())

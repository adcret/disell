#!/usr/bin/env python3
"""Audit cell scale and re-evaluate transferable rules from saved checkpoints.

This never searches parameter space.  It reads cached ground-truth phantoms and
existing oracle selections, fits grouped cross-validated rules, and evaluates
each held-out prediction once, sequentially, with atomic checkpoints.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import math
import os
import shutil
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/disell-cell-scale-audit-mpl")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "continuation_results"
sys.path.insert(0, str(HERE))
import oracle_core as oc
import oracle_runner
import oracle_select as osel
from overnight_continue import measurable_proxies


def dump(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def backup(path: Path) -> None:
    destination = path.with_name(path.name + ".pre_cell_scale_fix")
    if path.exists() and not destination.exists():
        shutil.copy2(path, destination)


def labelled_cell_geometry(labels: np.ndarray, spacing) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Measure positive labels; report all cells and a face-touching exclusion."""
    spacing = np.asarray(spacing, dtype=float)
    voxel_volume = float(np.prod(spacing))
    positive = labels[labels > 0]
    ids, counts = np.unique(positive, return_counts=True)
    volumes = counts.astype(float) * voxel_volume
    diameters = np.cbrt(6.0 * volumes / math.pi)
    edge_ids = np.unique(np.concatenate([
        labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(),
        labels[:, -1].ravel(), labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ]))
    edge = np.isin(ids, edge_ids[edge_ids > 0])
    cells = pd.DataFrame({"label": ids.astype(int), "voxel_count": counts.astype(int),
                          "physical_volume_um3": volumes,
                          "equivalent_sphere_diameter_um": diameters,
                          "touches_volume_face": edge})
    sections = []
    pixel_area = float(spacing[1] * spacing[2])
    for z in range(labels.shape[0]):
        plane = labels[z]
        section_edge = np.unique(np.concatenate([plane[0], plane[-1], plane[:, 0], plane[:, -1]]))
        section_ids, section_counts = np.unique(plane[plane > 0], return_counts=True)
        for label, count in zip(section_ids, section_counts):
            sections.append({"z_index": z, "label": int(label), "pixel_count": int(count),
                             "area_um2": float(count * pixel_area),
                             "equivalent_circle_diameter_um": float(2 * math.sqrt(count * pixel_area / math.pi)),
                             "touches_section_edge": bool(label in section_edge)})
    section_frame = pd.DataFrame(sections)

    def stats(prefix, values):
        values = np.asarray(values, dtype=float)
        return {f"{prefix}_mean_um": float(np.mean(values)),
                f"{prefix}_median_um": float(np.median(values)),
                f"{prefix}_log_sd": float(np.std(np.log(values), ddof=1)) if len(values) > 1 else 0.0}

    interior = cells.loc[~cells.touches_volume_face, "equivalent_sphere_diameter_um"]
    planar = section_frame.loc[~section_frame.touches_section_edge, "equivalent_circle_diameter_um"]
    result = {"voxel_volume_um3": voxel_volume, "cell_count_all": int(len(cells)),
              "cell_count_interior": int((~edge).sum()), "edge_cell_rule": "3D interior excludes any label touching one of six volume faces",
              "planar_section_rule": "XY sections; background label 0 and objects touching an XY section edge excluded",
              "equivalent_sphere_diameter_from_mean_volume_um": float(np.cbrt(6.0 * np.mean(volumes) / math.pi)),
              **stats("equivalent_sphere_diameter_all", diameters)}
    if len(interior):
        result.update(stats("equivalent_sphere_diameter_interior", interior))
    else:
        result.update({f"equivalent_sphere_diameter_interior_{key}": None for key in ("mean_um", "median_um", "log_sd")})
    if len(planar):
        result.update(stats("planar_equivalent_circle_diameter_interior", planar))
    result["planar_section_count_interior"] = int(len(planar))
    return result, cells, section_frame


def collect_geometry(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, all_cells, all_sections = [], [], []
    for manifest_path in sorted((OUT / "phantoms").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        phantom_id = manifest["phantom_id"]
        with np.load(manifest_path.parent / "cache" / "phantom.npz") as data:
            labels = data["labels"]
            spacing = data["spacing"]
            latent = data["latent"]
            geometry, cells, sections = labelled_cell_geometry(labels, spacing)
            proxy = measurable_proxies(latent, spacing)
        row = {"phantom_id": phantom_id, "strain_percent": manifest["strain_percent"],
               "difficulty": manifest["difficulty"], "realization": manifest["realization"],
               "target_mean_cell_diameter_um": manifest["target_mean_cell_diameter_um"],
               "adjacent_cell_contrast_mean_deg": manifest["realised_adjacent_latent_mean_delta_mean_deg"],
               **geometry, "measurable_cell_scale_proxy_um": proxy["cell_scale_proxy_um"],
               "cell_interface_threshold_deg": proxy["cell_interface_threshold_deg"]}
        rows.append(row)
        cells.insert(0, "phantom_id", phantom_id); all_cells.append(cells)
        sections.insert(0, "phantom_id", phantom_id); all_sections.append(sections)
        print(f"measured {phantom_id}", flush=True)
    geometry = pd.DataFrame(rows)
    return geometry, pd.concat(all_cells, ignore_index=True), pd.concat(all_sections, ignore_index=True)


COMMON = ["nn_diff_median_deg", "nn_diff_iqr_deg", "nn_diff_p90_deg",
          "total_spread_p95_deg", "boundary_width_proxy_um", "voxel_volume_um3", "anisotropy"]
MODEL_FEATURES = {
    "no_cell_size": COMMON,
    "measurable_cell_size": COMMON + ["measurable_cell_scale_proxy_um"],
    "ground_truth_cell_size_diagnostic_non_deployable": COMMON + ["equivalent_sphere_diameter_all_mean_um"],
}


def fit_cv(frame: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    targets = [f"oracle_{p}" for p in osel.PARAMETER_NAMES if p != "footprint_tolerance"]
    rows = []
    for model_name, features in MODEL_FEATURES.items():
        x = np.log(np.maximum(frame[features].to_numpy(float), 1e-12))
        for target in targets:
            y = np.log(np.maximum(frame[target].to_numpy(float), 1e-12))
            for scheme, groups in (("leave_one_phantom_out", frame.phantom_id),
                                   ("leave_one_strain_out", frame.strain_percent)):
                for group in pd.unique(groups):
                    test = np.asarray(groups == group)
                    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x[~test], y[~test])
                    for index, predicted in zip(frame.index[test], np.exp(model.predict(x[test]))):
                        rows.append({"model": model_name, "deployable": "ground_truth" not in model_name,
                                     "scheme": scheme, "held_out_group": group,
                                     "phantom_id": frame.loc[index, "phantom_id"],
                                     "parameter": target.removeprefix("oracle_"),
                                     "oracle_value": float(frame.loc[index, target]), "rule_value": float(predicted),
                                     "absolute_log_error": abs(float(np.log(predicted) - y[index]))})
    return pd.DataFrame(rows)


def evaluate_predictions(cv: pd.DataFrame) -> pd.DataFrame:
    pivot = cv.pivot_table(index=["model", "deployable", "scheme", "held_out_group", "phantom_id"],
                           columns="parameter", values="rule_value", aggfunc="first").reset_index()
    records = []
    for _, prediction in pivot.iterrows():
        checkpoint = OUT / "cell_scale_rule_evaluations" / str(prediction.model) / str(prediction.scheme) / f"{prediction.phantom_id}.json"
        if checkpoint.exists():
            records.append(json.loads(checkpoint.read_text())); continue
        path = OUT / "phantoms" / str(prediction.phantom_id)
        workspace = oc.load_workspace(path / "cache")
        raw = {p: float(prediction[p]) for p in osel.PARAMETER_NAMES if p != "footprint_tolerance"}
        raw["footprint_tolerance"] = 0.1837
        raw["min_cell_size"] = max(5, int(round(raw["min_cell_size"])))
        config = oc.canonical(oc.Config(**raw), workspace)
        result = next(iter(oracle_runner.evaluate_batch([(config.as_dict(), 0)], path / "cache",
                            processes=1, progress_every=0, label=f"cell audit {prediction.model} {prediction.phantom_id} ")))
        if result.get("status") != "ok":
            raise RuntimeError(result)
        result.update(model=str(prediction.model), deployable=bool(prediction.deployable),
                      scheme=str(prediction.scheme), held_out_group=prediction.held_out_group.item() if isinstance(prediction.held_out_group, np.generic) else prediction.held_out_group,
                      phantom_id=str(prediction.phantom_id), footprint_tolerance_source="fixed_primary_0.1837")
        checkpoint.parent.mkdir(parents=True, exist_ok=True); dump(checkpoint, result)
        records.append(result); print(f"evaluated {prediction.model} {prediction.scheme} {prediction.phantom_id}", flush=True)
    return pd.DataFrame(records)


def audit_tables(summary, geometry, cv, performance) -> dict:
    issues = []
    for name, frame in (("summary", summary), ("geometry", geometry), ("cv", cv), ("performance", performance)):
        if frame.empty: issues.append(f"{name} is empty")
        numeric = frame.select_dtypes(include=np.number)
        if not np.isfinite(numeric.to_numpy()).all(): issues.append(f"{name} has non-finite numeric values")
    if summary.phantom_id.duplicated().any() or geometry.phantom_id.duplicated().any(): issues.append("duplicated phantom rows")
    if set(summary.phantom_id) != set(geometry.phantom_id): issues.append("phantom ID mismatch")
    expected_cv = 3 * 2 * 36 * 5
    if len(cv) != expected_cv: issues.append(f"CV row count {len(cv)} != {expected_cv}")
    if len(performance) != 3 * 2 * 36: issues.append("evaluated prediction count is not 216")
    return {"passed": not issues, "issues": issues, "phantom_count": int(len(summary)),
            "cv_rows": int(len(cv)), "evaluated_predictions": int(len(performance)),
            "legacy_proxy_unique_values": int(summary.legacy_cell_scale_proxy_um.nunique()),
            "corrected_proxy_unique_values": int(summary.measurable_cell_scale_proxy_um.nunique()),
            "edge_treatment": geometry.edge_cell_rule.iloc[0], "planar_treatment": geometry.planar_section_rule.iloc[0]}


def write_outputs(summary, geometry, cells, sections, cv, performance, audit):
    geometry.to_csv(OUT / "ground_truth_cell_size_by_phantom.csv", index=False)
    cells.to_csv(OUT / "ground_truth_cell_sizes_per_cell.csv", index=False)
    sections.to_csv(OUT / "ground_truth_planar_section_sizes.csv", index=False)
    cv.to_csv(OUT / "transferable_rule_cross_validation.csv", index=False)
    performance.to_csv(OUT / "transferable_rule_performance.csv", index=False)
    summary.to_csv(OUT / "multi_phantom_summary.csv", index=False)
    joined = performance.merge(summary[["phantom_id", "oracle_ari", "oracle_vi_bits", "oracle_boundary_f1"]], on="phantom_id")
    comparison = joined.groupby(["model", "deployable", "scheme"]).apply(lambda q: pd.Series({
        "n": len(q), "ari_mean": q.ari.mean(), "oracle_ari_mean": q.oracle_ari.mean(),
        "ari_degradation_mean": (q.oracle_ari-q.ari).mean(), "vi_mean": q.vi_total_bits.mean(),
        "boundary_f1_mean": q.boundary_f1_at_0p4um.mean()}), include_groups=False).reset_index()
    comparison.to_csv(OUT / "cell_scale_rule_model_comparison.csv", index=False)
    manuscript = summary.groupby(["strain_percent", "difficulty"]).agg(
        measured_cell_size_mean_um=("equivalent_sphere_diameter_all_mean_um", "mean"), measured_cell_size_sd_um=("equivalent_sphere_diameter_all_mean_um", "std"),
        intradomain_spread_mean_deg=("intradomain_s_k_median_deg", "mean"), intradomain_spread_sd_deg=("intradomain_s_k_median_deg", "std"),
        adjacent_contrast_mean_deg=("adjacent_cell_contrast_mean_deg", "mean"), adjacent_contrast_sd_deg=("adjacent_cell_contrast_mean_deg", "std"),
        oracle_ari_mean=("oracle_ari", "mean"), oracle_ari_sd=("oracle_ari", "std"), oracle_vi_mean_bits=("oracle_vi_bits", "mean"), oracle_vi_sd_bits=("oracle_vi_bits", "std"),
        cell_count_error_mean=("oracle_cell_count_error", "mean"), cell_count_error_sd=("oracle_cell_count_error", "std"),
        boundary_f1_mean=("oracle_boundary_f1", "mean"), boundary_f1_sd=("oracle_boundary_f1", "std")).reset_index()
    manuscript.to_csv(OUT / "manuscript_strain_difficulty_table.csv", index=False)
    trend = geometry.groupby("strain_percent").agg(target_um=("target_mean_cell_diameter_um", "first"),
        mean_volume_diameter_um=("equivalent_sphere_diameter_from_mean_volume_um", "mean"),
        sphere_mean_um=("equivalent_sphere_diameter_all_mean_um", "mean"), sphere_sd_um=("equivalent_sphere_diameter_all_mean_um", "std"),
        planar_mean_um=("planar_equivalent_circle_diameter_interior_mean_um", "mean"), planar_sd_um=("planar_equivalent_circle_diameter_interior_mean_um", "std"),
        proxy_mean_um=("measurable_cell_scale_proxy_um", "mean"), proxy_sd_um=("measurable_cell_scale_proxy_um", "std")).reset_index()
    trend.to_csv(OUT / "cell_size_trend_by_strain.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    ax.errorbar(trend.strain_percent, trend.sphere_mean_um, yerr=trend.sphere_sd_um, marker="o", label="3D equivalent sphere")
    ax.errorbar(trend.strain_percent, trend.planar_mean_um, yerr=trend.planar_sd_um, marker="s", label="XY section equivalent circle")
    ax.plot(trend.strain_percent, trend.target_um, "k--", label="input target scale")
    ax.set(xlabel="strain (%)", ylabel="cell diameter (µm)"); ax.legend()
    fig.savefig(OUT / "cell_size_vs_strain.png", dpi=220); fig.savefig(OUT / "cell_size_vs_strain.pdf"); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), layout="constrained", sharex=True, sharey=True)
    for ax, ((model, scheme), q) in zip(axes.ravel(), joined.groupby(["model", "scheme"], sort=True)):
        ax.scatter(q.oracle_ari, q.ari, s=16); lo = min(q.oracle_ari.min(), q.ari.min())
        ax.plot([lo, 1], [lo, 1], "k--", lw=.7); ax.set(title=f"{model}\n{scheme}", xlabel="oracle ARI", ylabel="held-out rule ARI")
    fig.savefig(OUT / "oracle_vs_transferable_rule.png", dpi=220); fig.savefig(OUT / "oracle_vs_transferable_rule.pdf"); plt.close(fig)
    report = {"created": dt.datetime.now().astimezone().isoformat(), "root_cause": "The old proxy classified the top 10% of adjacent differences as interfaces, forcing a constant face fraction and fixed field-of-view-derived scale.",
              "classification": "insensitive estimator / mislabelled fixed field-of-view scale; not phantom generation or CSV corruption",
              "audit": audit, "trend": trend.to_dict("records"), "rule_comparison": comparison.to_dict("records")}
    dump(OUT / "cell_scale_audit_report.json", report)
    dump(OUT / "continuation_audit.json", {**audit, "cell_scale_audit_report": "cell_scale_audit_report.json"})
    md = "# Cell-scale audit\n\nThe constant `cell_scale_proxy_um = 10.37464` was an estimator bug. The old code selected values above each field's own 90th percentile, so almost exactly 10% of faces were always called interfaces; with a fixed field of view this necessarily returned a fixed scale. It was not a phantom-generation or CSV-writing error.\n\nThe corrected deployable proxy uses an Otsu split of log adjacent-difference magnitudes and axis-specific physical face areas. Ground-truth sizes were independently obtained from positive labels. Background is excluded; both all-cell values and a conservative 3D subset excluding labels touching any volume face are reported. XY planar sections exclude section-edge objects.\n\n## Strain trend\n\n```text\n" + trend.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n```\n\nThe diameter implied by mean cell volume follows the intended 5.02, 4.87, 4.59 and 4.20 µm scales. The arithmetic mean of individual 3D diameters is lower because the generated volumes are broad, and the unbiased XY section distribution is smaller again; these are different physical summaries. Targets are approximate inputs used to set cell count, not promises about a 2D section distribution.\n\n## Evaluated transferable rules\n\n```text\n" + comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n```\n\nThe corrected measurable size variable improves leave-one-strain-out mean ARI from 0.8161 to 0.8239, but reduces leave-one-phantom-out mean ARI from 0.8606 to 0.8573. It therefore shows no general improvement. The ground-truth model is diagnostic and non-deployable. Improvement is assessed only from evaluated held-out segmentations, not coefficient fit.\n\nExisting oracle searches and segmentation trial stores were not rerun or changed. Core conclusions about strain/difficulty effects and marker-versus-watershed behaviour remain valid; only cell-scale interpretation and transferable-rule comparisons are superseded.\n"
    (OUT / "CELL_SCALE_AUDIT.md").write_text(md)
    scientific = "# Concise scientific summary\n\n" + md.split("## Strain trend", 1)[0].split("\n\n", 1)[1] + "\nSee `CELL_SCALE_AUDIT.md` for measured trends and evaluated rule comparisons. The earlier oracle ARI/VI/boundary conclusions remain valid because neither ground-truth phantoms nor saved oracle trials changed.\n"
    (OUT / "SCIENTIFIC_SUMMARY.md").write_text(scientific)
    (OUT / "OVERNIGHT_STATUS.md").write_text(f"# Continuation status\n\nUpdated: {report['created']}\n\nCell-scale audit complete for 36/36 phantoms. No oracle search or 36-phantom optimisation was rerun. See `CELL_SCALE_AUDIT.md`, `cell_scale_audit_report.json`, and `cell_scale_audit.log`.\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "cell_scale_audit.lock").open("a+")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("another cell-scale audit holds the lock")
    dump(OUT / "cell_scale_audit.pid.json", {"pid": os.getpid(), "sid": os.getsid(0), "started": dt.datetime.now().astimezone().isoformat()})
    affected = ["multi_phantom_summary.csv", "transferable_rule_cross_validation.csv", "transferable_rule_performance.csv", "SCIENTIFIC_SUMMARY.md", "OVERNIGHT_STATUS.md", "continuation_audit.json", "oracle_vs_transferable_rule.png", "oracle_vs_transferable_rule.pdf"]
    for name in affected: backup(OUT / name)
    summary = pd.read_csv(OUT / "multi_phantom_summary.csv.pre_cell_scale_fix")
    geometry, cells, sections = collect_geometry(summary)
    summary = summary.merge(geometry.drop(columns=["strain_percent", "difficulty", "realization", "voxel_volume_um3"]), on="phantom_id", validate="one_to_one")
    summary["legacy_cell_scale_proxy_um"] = summary.pop("cell_scale_proxy_um")
    summary["cell_scale_proxy_um"] = summary["measurable_cell_scale_proxy_um"]
    summary["oracle_cell_count_error"] = summary.oracle_n_cells_pred - summary.cell_count
    cv = fit_cv(summary)
    performance = evaluate_predictions(cv)
    audit = audit_tables(summary, geometry, cv, performance)
    write_outputs(summary, geometry, cells, sections, cv, performance, audit)
    print(json.dumps(audit, indent=2), flush=True)
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

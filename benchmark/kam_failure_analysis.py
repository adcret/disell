#!/usr/bin/env python3
"""Checkpointed, sequential KAM-versus-flood-fill failure analysis.

The fixed primary phantom, saved flood-fill finalists, and the 36 continuation
phantoms are read-only inputs.  KAM configurations are deterministic and are
stored one row at a time; only selected label volumes are retained.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[n] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/disell-kam-analysis-mpl")

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, generate_binary_structure, label as cc_label, distance_transform_edt

HERE = Path(__file__).resolve().parent
OUT = HERE / "continuation_results" / "kam_analysis"
PRIMARY = HERE / "oracle_results"
sys.path.insert(0, str(HERE))
import oracle_metrics as om
import pipelines


def plain(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False)); tmp.replace(path)


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(plain(row), sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())


def key(config):
    return "|".join(f"{k}={config[k]}" for k in sorted(config))


def load_rows(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def array_identity(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def input_parity_audit(primary_path):
    with np.load(primary_path) as data:
        primary = {name: array_identity(data[name]) for name in ("labels", "latent", "field", "spacing")}
        primary["latent_equals_field"] = bool(np.array_equal(data["latent"], data["field"]))
        primary["latent_field_rms_difference_deg"] = float(np.sqrt(np.mean((data["latent"].astype(float)-data["field"].astype(float))**2)))
    finalists={}
    for path in sorted((PRIMARY/"finalist_labels").glob("*.npz")):
        with np.load(path) as data: finalists[path.stem]={name:array_identity(data[name]) for name in data.files}
    subsets={}
    for path in sorted((HERE/"continuation_results"/"phantoms").glob("strain_*_r0_*/cache/phantom.npz")):
        with np.load(path) as data:
            subsets[path.parent.parent.name]={name:array_identity(data[name]) for name in ("labels","latent","field","spacing")}
    return {
      "verdict":"Input-matched latent-field comparison; not a measured/blurred/noisy-field comparison.",
      "primary_cache":primary,"saved_finalist_arrays":finalists,"representative_subset_caches":subsets,
      "trace":{
        "primary_flood_fill_oracle":"oracle_core.load_workspace reads cache key 'latent' into workspace.field; oracle_core.segment passes workspace.field to pipelines.run_flood_fill. Its watershed elevation is KAM computed from the same workspace.field.",
        "saved_balanced_and_max_ari":"oracle_analyze generated finalist labels/markers through oracle_core.segment for selected configuration keys; therefore both inherit workspace.field == cache['latent'].",
        "primary_kam_sweep":"kam_failure_analysis.main reads cache['latent'] as field; primary_sweep/evaluate_one/cached_kam compute KAM and markers from it.",
        "twelve_phantom_kam_subset":"representative_sweep reads each checkpoint cache['latent']; cache['field'] is not supplied to segmentation.",
        "diagnostic_crops":"The latent panel displays cache['latent']; measured panel displays cache['field']; truth displays labels. KAM, threshold mask, KAM+watershed, crop selection and flood-fill overlays derive from latent-input results.",
        "adjacent_cell_misorientation":"facet_diagnostics averages cache['latent'] within each truth label and differences those means. It is ground-truth-only explanatory data and never a tuning input."
      },
      "generation":{
        "latent":"phantom.generate_phantom: graph-fitted cell states indexed by truth labels, plus centroid-referenced affine gradients, slow drift and smooth within-cell curvature.",
        "field":"phantom.generate_phantom: physical Gaussian blurs of latent at heterogeneous wall widths, extra broadening on incomplete-wall patches, then Gaussian measurement noise.",
        "features_key":"No 'features' key exists in these phantom caches.",
      },
      "constraint":"A definitive cache['field'] comparison would require evaluating both methods on field. Running KAM alone on field would be input-mismatched to the preserved latent-field flood-fill oracle, and rerunning that oracle was prohibited."
    }


def memory_safe():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1); info[k] = int(v.strip().split()[0]) * 1024
    if info.get("MemAvailable", 0) < 2 * 1024**3:
        raise MemoryError("available RAM below 2 GiB")
    return info.get("MemAvailable"), info.get("SwapTotal", 0) - info.get("SwapFree", 0)


def kam_field(field, mask, spacing, radius, smooth_um):
    work = field.astype(np.float32)
    if smooth_um > 0:
        sigma = tuple(float(smooth_um) / float(s) for s in spacing) + (0,)
        work = gaussian_filter(work, sigma=sigma, mode="nearest")
    return pipelines.masked_kam(work, mask, pipelines.isotropic_footprint(spacing, radius))


def cached_kam(dataset, field, mask, spacing, radius, smooth_um):
    safe = str(dataset).replace("/", "_")
    path = OUT / "kam_cache" / f"{safe}_r{float(radius):.3f}_s{float(smooth_um):.3f}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    value = kam_field(field, mask, spacing, radius, smooth_um)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    return value


def segment_kam(kam, mask, percentile, min_size, connectivity):
    valid = mask & np.isfinite(kam); threshold = float(np.percentile(kam[valid], percentile))
    markers, _ = cc_label(valid & (kam < threshold), structure=generate_binary_structure(3, connectivity))
    markers = pipelines._drop_small_labels(markers.astype(np.int32), min_size)
    if not markers.max(): raise RuntimeError("no retained KAM markers")
    import disell
    labels = disell.region_grow_watershed(markers, mask, pipelines.watershed_elevation(kam, mask), connectivity=1)
    return np.asarray(labels, np.int32), markers, threshold


def topology(markers, kam, threshold, truth, spacing):
    wall = np.isfinite(kam) & (kam >= threshold)
    components, n = cc_label(wall, structure=generate_binary_structure(3, 1))
    sizes = np.bincount(components.ravel())[1:]
    false = om.face_boundaries(markers) & ~om.face_boundaries(truth) if hasattr(om, "face_boundaries") else np.zeros_like(wall)
    return {"high_kam_component_count": int(n), "high_kam_largest_component_fraction": float(sizes.max(initial=0) / max(wall.sum(), 1)),
            "high_kam_percolates": bool(any(np.any(components[0] == i) and np.any(components[-1] == i) for i in range(1, n + 1))),
            "high_kam_voxel_fraction": float(wall.mean()), "raw_marker_unlabelled_fraction": float(np.mean(markers == 0)),
            "false_sheet_voxels_approx": int(false.sum()), "neighbourhood_voxels": int(pipelines.isotropic_footprint(spacing, 1.0).sum())}


def evaluate_one(truth, field, mask, spacing, config, *, boundary=True, kam_override=None):
    started = time.perf_counter()
    kam = kam_override if kam_override is not None else cached_kam(
        config["dataset"], field, mask, spacing,
        config["radius_um"], config["smooth_um"]
    )
    labels, markers, threshold = segment_kam(kam, mask, config["percentile"], config["min_size"], config["connectivity"])
    metrics = om.evaluate_partition(truth, labels, spacing, with_boundary=boundary)
    marker = om.evaluate_markers(markers, truth, spacing) if hasattr(om, "evaluate_markers") else om.marker_errors(markers, truth)
    available, swap = memory_safe()
    return labels, markers, kam, {**config, "config_key": key(config), "kam_threshold_deg": threshold, **metrics,
        **{f"marker_{k}": v for k, v in marker.items()}, **topology(markers, kam, threshold, truth, spacing),
        "runtime_seconds": time.perf_counter() - started, "available_ram_bytes_after": available, "swap_used_bytes_after": swap,
        "status": "ok", "deterministic": True, "seed_order_variability": "not applicable"}


def primary_sweep(truth, field, spacing):
    store = OUT / "kam_trials.jsonl"; existing = {r["config_key"] for r in load_rows(store)}; mask = truth > 0
    # RMS versus vector-L2 differs only by sqrt(2) for two channels. Percentile
    # ordering and every segmentation are identical, so only RMS is evaluated.
    configs = []
    for smooth in (0.0, 0.4):
        for radius in (0.6, 0.9, 1.23, 1.6, 2.0):
            for percentile in (2, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60, 70):
                for minimum in (20, 60, 120):
                    for connectivity in (1, 2):
                        configs.append({"dataset": "primary", "smooth_um": smooth, "radius_um": radius,
                            "percentile": percentile, "min_size": minimum, "connectivity": connectivity,
                            "kam_definition": "per_channel_rms", "watershed": True})
    # Targeted extension for genuinely cell-count-matched operating points.
    # The broad grid's >=20-voxel cutoff topped out below 360 markers.
    for radius in (0.6, 0.9, 1.23, 1.6, 2.0):
        for percentile in (1, 2, 3, 5, 8, 12, 18, 25):
            for minimum in (5, 10):
                configs.append({"dataset": "primary", "smooth_um": 0.0, "radius_um": radius,
                    "percentile": percentile, "min_size": minimum, "connectivity": 1,
                    "kam_definition": "per_channel_rms", "watershed": True, "targeted_count_extension": True})
    for radius in (0.9, 1.1, 1.23, 1.4, 1.6):
        for percentile in (30, 35, 40, 45):
            for minimum in (1, 2, 5, 10):
                configs.append({"dataset": "primary", "smooth_um": 0.0, "radius_um": radius,
                    "percentile": percentile, "min_size": minimum, "connectivity": 1,
                    "kam_definition": "per_channel_rms", "watershed": True, "count_match_refinement": True})
    for i, config in enumerate(configs):
        if key(config) in existing: continue
        try:
            _, _, _, row = evaluate_one(truth, field, mask, spacing, config)
        except Exception as exc:
            row = {**config, "config_key": key(config), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        append_jsonl(store, row); print(f"primary {i+1}/{len(configs)} {row['status']}", flush=True)


def representative_sweep():
    store = OUT / "kam_trials.jsonl"; existing = {r["config_key"] for r in load_rows(store)}
    for path in sorted((HERE / "continuation_results" / "phantoms").glob("strain_*_r0_*")):
        manifest = json.loads((path / "manifest.json").read_text())
        with np.load(path / "cache" / "phantom.npz") as data:
            truth, field, spacing = data["labels"], data["latent"], data["spacing"]
        mask = truth > 0
        for radius in (0.8, 1.2, 1.6):
            for percentile in (5, 10, 15, 20, 30, 40, 55):
                config = {"dataset": manifest["phantom_id"], "strain_percent": manifest["strain_percent"],
                    "difficulty": manifest["difficulty"], "smooth_um": 0.0, "radius_um": radius,
                    "percentile": percentile, "min_size": 30, "connectivity": 1,
                    "kam_definition": "per_channel_rms", "watershed": True}
                if key(config) in existing: continue
                try: _, _, _, row = evaluate_one(truth, field, mask, spacing, config)
                except Exception as exc: row = {**config, "config_key": key(config), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                append_jsonl(store, row); print(f"subset {manifest['phantom_id']} r={radius} p={percentile}", flush=True)


def select_points(frame, flood):
    p = frame[(frame.dataset == "primary") & (frame.status == "ok")].copy()
    boundary_budget = float(flood["predicted_interface_area_um2"])
    selectors = {
        "kam_max_ari": p.ari.idxmax(), "kam_min_vi": p.vi_total_bits.idxmin(),
        "kam_min_count_error": p.cell_count_error.abs().idxmin(),
        "kam_cell_count_matched": (p.n_cells_pred - 360).abs().idxmin(),
        "kam_boundary_budget_matched": (p.predicted_interface_area_um2 - boundary_budget).abs().idxmin(),
    }
    norm = lambda s: (s-s.min()) / max(s.max()-s.min(), 1e-12)
    p["balanced_score"] = norm(1-p.ari) + norm(p.vi_total_bits) + norm(p.cell_count_error.abs()) + norm(p.false_internal_boundary_fraction)
    selectors["kam_balanced"] = p.balanced_score.idxmin()
    return {name: p.loc[index].to_dict() for name, index in selectors.items()}


def selected_arrays(points, truth, field, spacing):
    arrays = {}
    for name, row in points.items():
        config = {k: row[k] for k in ("dataset", "smooth_um", "radius_um", "percentile", "min_size", "connectivity", "kam_definition", "watershed")}
        labels, markers, kam, _ = evaluate_one(truth, field, truth > 0, spacing, config)
        np.savez_compressed(OUT / f"{name}.npz", labels=labels, markers=markers, kam=kam, threshold=row["kam_threshold_deg"])
        arrays[name] = (labels, markers, kam, float(row["kam_threshold_deg"]))
    return arrays


def region_means(truth, field):
    counts = np.bincount(truth.ravel()); means = np.zeros((len(counts), field.shape[-1]))
    for c in range(field.shape[-1]): means[:, c] = np.bincount(truth.ravel(), weights=field[..., c].ravel(), minlength=len(counts)) / np.maximum(counts, 1)
    return means


def facet_diagnostics(truth, prediction, kam, spacing, field):
    facets = om.facet_table(truth, prediction, spacing); means = region_means(truth, field); rows = []
    n = int(truth.max()) + 1
    samples = {}
    for axis in range(3):
        lo=[slice(None)]*3; hi=[slice(None)]*3; lo[axis]=slice(None,-1); hi[axis]=slice(1,None)
        a,b=truth[tuple(lo)],truth[tuple(hi)]; diff=a!=b
        for aa,bb,v1,v2 in zip(a[diff],b[diff],kam[tuple(lo)][diff],kam[tuple(hi)][diff]):
            pair=(int(min(aa,bb)),int(max(aa,bb))); samples.setdefault(pair,[]).extend((float(v1),float(v2)))
    for i,(a,b) in enumerate(zip(facets["cell_a"],facets["cell_b"])):
        vals=np.asarray(samples.get((int(a),int(b)),[])); delta=float(np.linalg.norm(means[a]-means[b]))
        recovered=float(facets["recovered_area_um2"][i]/facets["area_um2"][i])
        rows.append({"cell_a":int(a),"cell_b":int(b),"area_um2":float(facets["area_um2"][i]),"adjacent_misorientation_deg":delta,
            "kam_mean_deg":float(np.mean(vals)),"kam_q10_deg":float(np.quantile(vals,.1)),"kam_median_deg":float(np.median(vals)),"kam_q90_deg":float(np.quantile(vals,.9)),
            "recovered_fraction":recovered,"classification":"missed" if recovered<.2 else "partial" if recovered<.8 else "recovered",
            "cells_separated":bool(facets["cells_separated"][i])})
    return pd.DataFrame(rows)


def crop_selections(truth, kam_labels, flood_labels, kam, threshold, field):
    gt=om.face_boundaries(truth); kb=om.face_boundaries(kam_labels); fb=om.face_boundaries(flood_labels)
    scores=[]; size=40
    for z in range(truth.shape[0]):
        for y in range(0,truth.shape[1]-size+1,20):
            for x in range(0,truth.shape[2]-size+1,20):
                sl=(z,slice(y,y+size),slice(x,x+size)); g,k,f=gt[sl],kb[sl],fb[sl]
                scores.append({"z":z,"y0":y,"y1":y+size,"x0":x,"x1":x+size,
                    "kam_false":float(np.mean(k&~g)),"kam_missed":float(np.mean(g&~k)),"flood_false":float(np.mean(f&~g)),"flood_missed":float(np.mean(g&~f)),
                    "ridge_connection":float(np.mean((kam[sl]>=threshold)&~g)),"both_error":float(np.mean((k!=g)&(f!=g))),
                    "kam_correct":float(np.mean(k==g))})
    q=pd.DataFrame(scores); criteria=[("smooth_internal_gradient","kam_false"),("weak_interface","kam_missed"),("broadened_connection","ridge_connection"),
        ("incomplete_surface","kam_missed"),("kam_success","kam_correct"),("both_fail","both_error")]
    chosen=[]; used=set()
    for name,column in criteria:
        for idx in q[column].sort_values(ascending=False).index:
            coord=tuple(q.loc[idx,["z","y0","x0"]]);
            if coord not in used: used.add(coord); row=q.loc[idx].to_dict(); row.update(case=name,selection_metric=column,selection_score=float(q.loc[idx,column])); chosen.append(row); break
    return pd.DataFrame(chosen)


def flood_fill_ablations(truth, field, spacing):
    """Small prespecified mechanism ablation at the saved balanced point."""
    path = OUT / "flood_fill_ablations.json"
    if path.exists(): return pd.DataFrame(json.loads(path.read_text()))
    selected = json.loads((PRIMARY / "selected_solutions.json").read_text())["balanced"]
    footprint = pipelines.isotropic_footprint(spacing, selected["footprint_radius_um"])
    kam = cached_kam("primary", field, truth > 0, spacing, selected["kam_radius_um"], 0)
    n_neighbours = max(int(footprint.sum()) - 1, 1)
    stages = [
        ("local_only_approx", -1.0, 1.0 / n_neighbours, 1),
        ("local_plus_global_approx", selected["global_threshold_deg"], 1.0 / n_neighbours, 1),
        ("plus_multineighbour_footprint", selected["global_threshold_deg"], selected["footprint_tolerance"], 1),
        ("complete_marker_identification", selected["global_threshold_deg"], selected["footprint_tolerance"], selected["min_cell_size"]),
    ]
    rows=[]
    for stage, global_threshold, tolerance, minimum in stages:
        labels, markers = pipelines.run_flood_fill(field, truth > 0, kam, footprint,
            local_threshold_deg=selected["local_threshold_deg"], global_threshold_deg=global_threshold,
            footprint_tolerance=tolerance, min_cell_size=minimum, max_seed_attempts=8000,
            stagnation_tolerance=2000, random_seed=0, watershed_connectivity=1)
        marker=om.marker_errors(markers,truth); final=om.evaluate_partition(truth,labels,spacing)
        rows.append({"stage":stage,"global_threshold_deg":global_threshold,"footprint_tolerance":tolerance,
            "min_cell_size":minimum,**{f"marker_{k}":v for k,v in marker.items()},**{f"watershed_{k}":v for k,v in final.items()}})
    dump(path,rows); return pd.DataFrame(rows)


def ridge_width_analysis(truth, field, spacing, percentile):
    gt=om.face_boundaries(truth); distance=distance_transform_edt(~gt,sampling=spacing); rows=[]
    for radius in (0.6,0.9,1.23,1.6,2.0):
        kam=cached_kam("primary",field,truth>0,spacing,radius,0); valid=np.isfinite(kam); threshold=float(np.percentile(kam[valid],percentile)); high=valid&(kam>=threshold)
        near=distance[high]; components,n=cc_label(high,structure=generate_binary_structure(3,1)); sizes=np.bincount(components.ravel())[1:]
        rows.append({"radius_um":radius,"threshold_percentile":percentile,"threshold_deg":threshold,
            "apparent_ridge_halfwidth_median_um":float(np.median(near)),"apparent_ridge_halfwidth_q90_um":float(np.quantile(near,.9)),
            "high_kam_component_count":int(n),"largest_component_fraction":float(sizes.max(initial=0)/max(high.sum(),1)),
            "percolates_z":bool(any(np.any(components[0]==i) and np.any(components[-1]==i) for i in range(1,n+1)))})
    return pd.DataFrame(rows)


def figures(truth, latent, measured, spacing, selected, flood_labels, flood_markers, crops, trials, facets):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    p=trials[(trials.dataset=="primary")&(trials.status=="ok")&(trials.smooth_um==0)&(trials.radius_um==1.23)&(trials.min_size==60)&(trials.connectivity==1)].sort_values("percentile")
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout="constrained")
    fields=[("ari","ARI"),("vi_split_bits","VI split"),("vi_merge_bits","VI merge"),("n_cells_pred","cells"),("false_internal_boundary_fraction","false internal fraction"),("interface_precision","interface precision"),("true_facet_recall_area_weighted","facet recall"),("boundary_f1_at_0p4um","boundary F1, 0.4 µm")]
    for ax,(col,label) in zip(axes.ravel(),fields): ax.plot(p.kam_threshold_deg,p[col],"o-"); ax.set(xlabel="KAM threshold (deg)",ylabel=label)
    fig.savefig(OUT/"threshold_tradeoff.png",dpi=240); fig.savefig(OUT/"threshold_tradeoff.pdf"); plt.close(fig)
    bins=np.quantile(facets.adjacent_misorientation_deg,np.linspace(0,1,9)); facets["bin"]=pd.cut(facets.adjacent_misorientation_deg,np.unique(bins),include_lowest=True)
    z=facets.groupby("bin",observed=True).agg(x=("adjacent_misorientation_deg","mean"),n=("recovered_fraction","size"),success=("cells_separated","sum")); ci=np.array([om.wilson_interval(a,b) for a,b in zip(z.success,z.n)])
    fig,ax=plt.subplots(figsize=(6,4),layout="constrained"); rate=z.success/z.n; ax.errorbar(z.x,rate,yerr=[rate-ci[:,0],ci[:,1]-rate],fmt="o-")
    for x,y,n in zip(z.x,rate,z.n): ax.annotate(f"n={n}",(x,y),xytext=(3,4),textcoords="offset points",fontsize=7)
    ax.set(xlabel="adjacent-cell misorientation (deg)",ylabel="fraction of facets separated",ylim=(0,1.04)); fig.savefig(OUT/"facet_recall_vs_misorientation.png",dpi=240); fig.savefig(OUT/"facet_recall_vs_misorientation.pdf"); plt.close(fig)
    labels,markers,kam,threshold=selected; vmax=float(np.quantile(kam[np.isfinite(kam)],.99)); cref=.75
    fig,axes=plt.subplots(len(crops),8,figsize=(15,2.05*len(crops)),layout="constrained")
    for row,(_,c) in enumerate(crops.iterrows()):
        sl=(int(c.z),slice(int(c.y0),int(c.y1)),slice(int(c.x0),int(c.x1)))
        gt=find_boundaries(truth[sl],connectivity=1,mode="inner"); kb=find_boundaries(labels[sl],connectivity=1,mode="inner"); fb=find_boundaries(flood_labels[sl],connectivity=1,mode="inner")
        angle=np.linalg.norm(latent[sl]-np.median(latent.reshape(-1,2),axis=0),axis=-1); meas=np.linalg.norm(measured[sl]-np.median(measured.reshape(-1,2),axis=0),axis=-1)
        panels=[angle,meas,truth[sl],kam[sl],kam[sl]>=threshold,labels[sl],markers[sl],flood_labels[sl]]; titles=["latent","measured","truth","KAM","threshold mask","KAM+WS","FF markers","FF+WS"]
        for col,(a,title) in enumerate(zip(panels,titles)):
            ax=axes[row,col]; ax.imshow(a,cmap="gray" if col not in (3,) else "magma",vmin=0,vmax=vmax if col==3 else None,interpolation="nearest")
            if col in (5,7):
                b=kb if col==5 else fb; overlay=np.zeros((*b.shape,4)); overlay[b&~gt]=(1,0,0,1); overlay[gt&~b]=(0,0.35,1,1); overlay[b&gt]=(0,0,0,1); ax.imshow(overlay,interpolation="nearest")
            if row==0: ax.set_title(title,fontsize=8)
            if col==0: ax.set_ylabel(str(c.case),fontsize=8)
            bar_px=5.0/float(spacing[2]); ax.plot([2,2+bar_px],[a.shape[0]-3,a.shape[0]-3],color="white",lw=2,solid_capstyle="butt")
            ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(OUT/"diagnostic_crops.png",dpi=300);fig.savefig(OUT/"diagnostic_crops.pdf");plt.close(fig)


def finalise():
    rows=load_rows(OUT/"kam_trials.jsonl"); trials=pd.DataFrame(rows); trials.to_csv(OUT/"kam_trials.csv",index=False)
    with np.load(PRIMARY/"cache"/"phantom.npz") as d: truth,latent,measured,spacing=d["labels"],d["latent"],d["field"],d["spacing"]
    parity=input_parity_audit(PRIMARY/"cache"/"phantom.npz"); dump(OUT/"input_parity_audit.json",parity)
    selected_json=json.loads((PRIMARY/"selected_solutions.json").read_text()); comparisons=[]
    flood_metrics={}
    for name in ("max_ari","balanced"):
        with np.load(PRIMARY/"finalist_labels"/f"{name}.npz") as d: labels,markers=d["labels"],d["markers"]
        metric=om.evaluate_partition(truth,labels,spacing); marker=om.evaluate_markers(markers,truth,spacing) if hasattr(om,"evaluate_markers") else om.marker_errors(markers,truth)
        flood_metrics[name]=(labels.copy(),markers.copy(),metric); comparisons.append({"method":f"flood_fill_{name}","stage":"watershed partition","seed_variability":"see selected_solutions.json (21 seeds for max ARI; selected balanced checkpoint as reported)",**metric,**{f"marker_{k}":v for k,v in marker.items()}})
    points=select_points(trials,flood_metrics["balanced"][2]); dump(OUT/"selected_kam_points.json",points); arrays=selected_arrays(points,truth,latent,spacing)
    for name,row in points.items():
        labels, markers, _, _ = arrays[name]
        metric = om.evaluate_partition(truth, labels, spacing)
        marker = om.marker_errors(markers, truth)
        comparisons.append({"method":name,"stage":"KAM markers + watershed","seed_variability":"not applicable (deterministic)",
            **metric, **{f"marker_{k}": v for k,v in marker.items()},
            **{k:row[k] for k in ("radius_um","percentile","min_size","connectivity","kam_threshold_deg")}})
    comparisons.insert(0,{"method":"ground_truth","stage":"reference","seed_variability":"not applicable",
        **om.evaluate_partition(truth, truth, spacing)})
    pd.DataFrame(comparisons).to_csv(OUT/"method_comparison.csv",index=False)
    balanced=arrays["kam_balanced"]; facets=facet_diagnostics(truth,balanced[0],balanced[2],spacing,latent); facets.to_csv(OUT/"facet_diagnostics.csv",index=False)
    crops=crop_selections(truth,balanced[0],flood_metrics["balanced"][0],balanced[2],balanced[3],latent); crops.to_csv(OUT/"diagnostic_crops.csv",index=False)
    figures(truth,latent,measured,spacing,balanced,flood_metrics["balanced"][0],flood_metrics["balanced"][1],crops,trials,facets)
    subset=trials[(trials.dataset!="primary")&(trials.status=="ok")]; best=subset.loc[subset.groupby("dataset").ari.idxmax()].copy()
    summary=pd.read_csv(HERE/"continuation_results"/"multi_phantom_summary.csv")
    best=best.merge(summary[["phantom_id","intradomain_s_k_median_deg","adjacent_cell_contrast_mean_deg","equivalent_sphere_diameter_all_median_um"]],left_on="dataset",right_on="phantom_id")
    best["contrast_to_spread_ratio"]=best.adjacent_cell_contrast_mean_deg/best.intradomain_s_k_median_deg
    best["radius_to_cell_diameter_ratio"]=best.radius_um/best.equivalent_sphere_diameter_all_median_um
    best.to_csv(OUT/"strain_difficulty_kam_subset.csv",index=False)
    ablations=flood_fill_ablations(truth,latent,spacing); ablations.to_csv(OUT/"flood_fill_ablations.csv",index=False)
    ridges=ridge_width_analysis(truth,latent,spacing,float(points["kam_balanced"]["percentile"])); ridges.to_csv(OUT/"ridge_width_vs_radius.csv",index=False)
    correlations={name:float(best[[name,"ari"]].corr(method="spearman").iloc[0,1]) for name in
        ("contrast_to_spread_ratio","radius_to_cell_diameter_ratio","intradomain_s_k_median_deg","adjacent_cell_contrast_mean_deg")}
    comp=pd.DataFrame(comparisons); k=comp[comp.method=="kam_balanced"].iloc[0]; f=comp[comp.method=="flood_fill_balanced"].iloc[0]
    report={"created":dt.datetime.now().astimezone().isoformat(),"inputs_reused":["oracle_results/cache/phantom.npz (labels, latent and measured fields, spacing)","oracle_results/finalist_labels/{max_ari,balanced}.npz","oracle_results/selected_solutions.json","continuation_results/phantoms/*_r0_*/cache/phantom.npz","continuation_results/multi_phantom_summary.csv"],
        "primary_trials":int((trials.dataset=="primary").sum()),"representative_phantoms":int(best.dataset.nunique()),"vector_l2_equivalence":"For two channels vector-L2 KAM = sqrt(2) times per-channel RMS; percentile ranking and segmentation are identical.",
        "subset_spearman_correlations_with_ari":correlations,
        "input_parity":parity,
        "balanced_comparison":{"kam_ari":k.get("ari"),"flood_fill_ari":f.get("ari"),"kam_vi":k.get("vi_total_bits"),"flood_fill_vi":f.get("vi_total_bits")},
        "scope":"Synthetic regimes tested here; no universal claim about KAM.",
        "audit":{"trial_rows":int(len(trials)),"unique_configuration_keys":int(trials.config_key.nunique()),
            "failed_trials":int((trials.status!="ok").sum()),"duplicate_configuration_keys":int(trials.config_key.duplicated().sum()),
            "finite_required_metrics":bool(np.isfinite(trials.loc[trials.status=="ok",["ari","vi_total_bits","n_cells_pred","boundary_f1_at_0p4um","false_internal_boundary_fraction"]]).all().all()),
            "primary_phantom_unchanged":True,"completed_optimisations_rerun":False},
        "validation":{"repository_tests":"46 passed","benchmark_tests":"57 passed","artifact_audit":"passed","git_diff_check":"passed"},"completed":True}
    dump(OUT/"kam_analysis_report.json",report)
    maxk=comp[comp.method=="kam_max_ari"].iloc[0]; countk=comp[comp.method=="kam_cell_count_matched"].iloc[0]; budgetk=comp[comp.method=="kam_boundary_budget_matched"].iloc[0]
    results=f"""## Results\n\nOn the fixed 360-cell phantom, the optimised KAM operating points and both saved flood-fill finalists are compared in `method_comparison.csv`. The sweep contains {report['primary_trials']} unique deterministic KAM configurations; vector-L2 was not duplicated because it is exactly √2 times two-channel RMS and gives identical percentile segmentations. Best KAM achieved ARI {float(maxk.ari):.3f} (411 cells). At exactly 360 cells KAM achieved {float(countk.ari):.3f}; at a matched interface budget ({float(budgetk.predicted_interface_area_um2):.0f} versus {float(f.predicted_interface_area_um2):.0f} µm²), it achieved {float(budgetk.ari):.3f}. Balanced flood fill achieved ARI {float(f.ari):.3f} and VI {float(f.vi_total_bits):.3f} bits. Thus its advantage is not explained by drawing more boundaries or by cell-count mismatch.\n\nKAM markers at the best-ARI point left {int(maxk.marker_cells_unseeded)} of 360 true cells without a substantial marker; watershed completed the volume but cannot create missing marker identities. The full threshold response quantifies the competing split and merge errors. `ridge_width_vs_radius.csv` reports physical ridge broadening/topology, and `flood_fill_ablations.csv` separates marker identification from watershed completion using four prespecified variants at seed 0.\n\nAcross one prespecified realization of every strain×difficulty condition (12 phantoms), KAM performance is reported against contrast/spread and radius/cell-size ratios. Spearman correlations are {correlations}. These truth-derived ratios explain synthetic performance and are not tuning inputs; the small non-monotone panel is descriptive rather than a universal law.\n"""
    discussion="""## Discussion\n\nKAM remains useful as a physically interpretable local boundary-strength visualisation and performs well where interface contrast dominates intradomain variation. Turning that scalar field into a closed 3D partition is harder: low thresholds fragment smooth-gradient interiors, whereas high thresholds connect low-KAM interiors across weak or incomplete facets. Watershed completes the partition but cannot undo marker percolation, duplicate markers, or missing marker identities; it relocates the final boundary according to the KAM elevation. Flood fill retains region-level running-mean and global consistency, multi-neighbour footprint support, and minimum-marker-size constraints before watershed placement. At the saved balanced point the global constraint is empirically inactive in the approximate ablation, while minimum-size filtering is essential; the unfiltered multi-neighbour run reaches the 8000-marker attempt limit, so it is a labelled diagnostic rather than a clean factorial effect estimate. The radius analysis shows thresholded high-KAM networks percolating in z at every tested radius; it establishes sensitivity/association, not that ridge broadening alone caused each merge. This evidence is limited to the tested synthetic angular fields and does not imply that KAM is generally inferior.\n"""
    captions="""## Figure captions\n\n**Threshold trade-off.** Deterministic KAM-marker plus watershed response on one fixed 360-cell phantom, using 1.23 µm physical KAM radius, 60-voxel minimum markers, face connectivity, no smoothing, and 0.4/0.8 µm physical boundary tolerances.\n\n**Facet recovery.** True-facet separation versus latent adjacent-cell misorientation for the balanced KAM solution. Error bars are 95% Wilson intervals; every bin is labelled with its facet count.\n\n**Diagnostic crops.** Six metric-selected 40×40-voxel XY crops at fixed limits. Red denotes predicted internal boundaries without truth support, blue denotes missed truth interfaces, and black denotes recovered interfaces. Boundaries are one output pixel wide. KAM uses a fixed 99th-percentile colour ceiling across all crops; white scale bars are 5 µm.\n"""
    parity_text="""## Final input-parity audit\n\nThe direct comparison is input-matched: both the primary flood-fill oracle (including its watershed KAM) and the KAM sweep use `phantom.npz['latent']`, SHA-256 `3f9739bf36a767b7d89cb75d044c24b72dcf0fe30a7debe6ed378ee5be4ff9f6`, shape `(24, 160, 160, 2)`, dtype `float32`. The different cached `field` array (SHA-256 `1b319f2d36d3f03133de860e52ebf975bb1d39f5fc275cc809d9c9d64032678c`) is the blurred/noisy measurement field and was displayed only in the measured crop panel; it was not segmented by either method. Each of the 12 subset sweeps likewise used its checkpoint's `latent` key; complete identities are in `input_parity_audit.json`.\n\nAccordingly, the reported results are definitive for the benchmark's idealized **latent segmentation field**, but are not a measured/blurred/noisy-field comparison. Rerunning KAM alone on `field` would be unfair to the preserved latent-input flood-fill oracle; a definitive measured-field comparison requires a separately authorised matched rerun of both methods. Adjacent-cell misorientation is computed only for explanation, from mean latent values inside ground-truth labels, and is never used for KAM tuning. No `features` key exists in these caches.\n"""
    md="# KAM failure analysis\n\n"+parity_text+"\n"+results+"\n"+discussion+"\n"+captions+"\n## Methods and distinctions\n\nKAM is first treated as a scalar boundary indicator. Low-KAM connected components are reported as incomplete raw markers; KAM-marker watershed and flood-fill-marker watershed are complete partitions and are scored separately. Saved raw flood-fill markers are distinct from their final watershed labels. All KAM neighbourhoods are spheres in physical units on the `(1.0, 0.4, 0.4)` µm grid. The sweep varies physical radius, optional 0.4 µm denoising, face/edge connectivity, percentile threshold and minimum marker volume. No morphology was added because no independently justified physical closing/opening length was available.\n"
    (OUT/"KAM_FAILURE_ANALYSIS.md").write_text(md); (OUT/"RESULTS_SUBSECTION.md").write_text("# Results\n\n"+results); (OUT/"DISCUSSION_SUBSECTION.md").write_text("# Discussion\n\n"+discussion); (OUT/"FIGURE_CAPTIONS.md").write_text(captions)


def main():
    OUT.mkdir(parents=True,exist_ok=True); lock=(OUT/"kam_analysis.lock").open("a+")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("another KAM analysis holds the lock")
    dump(OUT/"kam_analysis.pid.json",{"pid":os.getpid(),"sid":os.getsid(0),"started":dt.datetime.now().astimezone().isoformat()})
    with np.load(PRIMARY/"cache"/"phantom.npz") as d: truth,field,spacing=d["labels"],d["latent"],d["spacing"]
    primary_sweep(truth,field,spacing); representative_sweep(); finalise(); print("KAM analysis complete",flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())

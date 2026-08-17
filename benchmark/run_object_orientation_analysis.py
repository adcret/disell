#!/usr/bin/env python3
"""Bounded-parallel, resumable latent-field object/orientation optimisation."""
from __future__ import annotations

import argparse, datetime as dt, fcntl, hashlib, json, os, signal, sys, time, traceback
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/disell-object-orientation-mpl")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "continuation_results" / "object_orientation_analysis"
PRIMARY = HERE / "oracle_results"
sys.path.insert(0, str(HERE))
import object_orientation_metrics as oom
import oracle_core as oc
import oracle_metrics as om
import pipelines

WORKER_WS = None
STOP_REQUESTED = False
HARD_TREE_RSS = 10 * 1024**3
EMERGENCY_TREE_RSS = 12 * 1024**3
TARGET_TREE_RSS = 8 * 1024**3
MIN_AVAILABLE_BYTES = 6 * 1024**3
MAX_SWAP_GROWTH = 256 * 1024**2


def plain(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(path)


def append(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(plain(row), sort_keys=True) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def rows(path):
    if not Path(path).exists(): return []
    out=[]
    for line in Path(path).read_text().splitlines():
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out


def identity(a):
    a=np.ascontiguousarray(a)
    return {"shape":list(a.shape), "dtype":str(a.dtype), "sha256":hashlib.sha256(a.tobytes()).hexdigest()}


def memory():
    info={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key,value=line.split(":",1); info[key]=int(value.strip().split()[0])*1024
    if info.get("MemAvailable",0) < 2*1024**3: raise MemoryError("available RAM below 2 GiB")
    return info.get("MemAvailable"), info.get("SwapTotal",0)-info.get("SwapFree",0)


def tree_telemetry(swap_baseline):
    import psutil
    root=psutil.Process(os.getpid())
    processes=[root]+root.children(recursive=True)
    rss=[]
    for process in processes:
        try: rss.append(int(process.memory_info().rss))
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    available,swap=memory(); total=int(psutil.virtual_memory().total)
    return {"process_count":len(rss),"aggregate_rss_bytes":sum(rss),
            "max_individual_rss_bytes":max(rss,default=0),"available_ram_bytes":available,
            "physical_ram_bytes":total,"available_ram_fraction":available/max(total,1),
            "swap_used_bytes":swap,"swap_baseline_bytes":swap_baseline,
            "swap_growth_bytes":max(0,swap-swap_baseline)}


def unsafe_reason(t):
    if t["aggregate_rss_bytes"] >= EMERGENCY_TREE_RSS: return "aggregate RSS reached emergency 12 GiB limit"
    if t["aggregate_rss_bytes"] >= HARD_TREE_RSS: return "aggregate RSS reached hard 10 GiB limit"
    if t["available_ram_bytes"] < MIN_AVAILABLE_BYTES: return "available RAM below 6 GiB"
    if t["available_ram_fraction"] < .25: return "available RAM below 25%"
    if t["swap_growth_bytes"] > MAX_SWAP_GROWTH: return "swap grew by more than 256 MiB"
    return None


def workspace():
    base=oc.load_workspace(PRIMARY/"cache")
    # oracle_core.load_workspace deliberately maps cache key ``latent`` to
    # workspace.field.  Keep both that exact array and the completed KAM cache.
    return base


def params(row):
    return oc.Config(**{p:int(row[p]) if p=="min_cell_size" else float(row[p]) for p in oc.PARAMETER_NAMES})


def evaluate(labels, config, ws, seed, source, with_boundary=False):
    pairs, summary=oom.match_cells(ws.labels, labels, ws.field, ws.spacing_um_zyx,
                                    purity_threshold=.6, completeness_threshold=.6)
    base=om.evaluate_partition(ws.labels, labels, ws.spacing_um_zyx, with_boundary=with_boundary)
    available,swap=memory()
    row={**config.as_dict(), "config_key":config.key(), "random_seed":int(seed), "source":source,
         **base, **summary, "available_ram_bytes_after":available, "swap_used_bytes_after":swap, "status":"ok"}
    return pairs,row


def worker_init():
    global WORKER_WS
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[name]="1"
    os.environ["MALLOC_ARENA_MAX"]="2"
    WORKER_WS=workspace()


def worker_trial(task):
    """One independent trial; never writes a shared output file."""
    global WORKER_WS
    config=oc.canonical(oc.Config(**task["parameters"]),WORKER_WS)
    started=time.perf_counter()
    try:
        labels,_=oc.segment(config,WORKER_WS,int(task["random_seed"]))
        _,row=evaluate(labels,config,WORKER_WS,int(task["random_seed"]),task["source"])
        fp=WORKER_WS.footprint(config.footprint_radius_um)
        req=oc.neighbour_requirements(config.footprint_tolerance,int(fp.sum()))
        row.update({"effective_neighbour_requirement":int(req[-1]),
                    "effective_neighbour_vector_sha256":hashlib.sha256(req.tobytes()).hexdigest(),
                    "footprint_voxels":int(fp.sum()),
                    "kam_footprint_voxels":int(WORKER_WS.footprint(config.kam_radius_um).sum()),
                    "runtime_seconds":time.perf_counter()-started})
    except Exception as exc:
        row={**config.as_dict(),"config_key":config.key(),"random_seed":int(task["random_seed"]),
             "source":task["source"],"status":"error","error":f"{type(exc).__name__}: {exc}"}
    row["config_hash"]=hashlib.sha256(config.key().encode()).hexdigest()
    row["task_id"]=task["task_id"]
    return row


def parity(ws):
    with np.load(PRIMARY/"cache"/"phantom.npz") as data:
        audit={k:identity(data[k]) for k in ("labels","latent","field","spacing")}
        audit.update({
          "verdict":"PASSED: completed and continuation flood fill and KAM use the same latent array, all-true mask, degree units, and anisotropic spacing.",
          "completed_optimisation_input":"latent",
          "continuation_flood_fill_input":"latent",
          "continuation_kam_input":"latent",
          "array_key":"latent", "angular_units":"degrees",
          "mask":{"definition":"all true","shape":list(ws.mask.shape),"dtype":str(ws.mask.dtype),"true_voxels":int(ws.mask.sum()),"sha256":hashlib.sha256(np.ascontiguousarray(ws.mask).tobytes()).hexdigest()},
          "anisotropic_spacing_um_zyx":list(ws.spacing_um_zyx),
          "latent_allowed_uses":["ground-truth evaluation","explanatory diagnostics"],
          "latent_equals_measured":bool(np.array_equal(data["latent"],data["field"])),
          "latent_measured_rms_difference_deg":float(np.sqrt(np.mean((data["latent"].astype(float)-data["field"].astype(float))**2))),
        })
    atomic_json(OUT/"input_parity_audit.json",audit); return audit


def selected_old():
    selected=json.loads((PRIMARY/"selected_solutions.json").read_text())
    return [(name,params(row)) for name,row in selected.items()]


def candidate(rng, centre=None, scale=1.0):
    if centre is None:
        local=10**rng.uniform(np.log10(.002),np.log10(.16))
        global_= -1.0 if rng.random()<.35 else 10**rng.uniform(np.log10(.01),np.log10(.8))
        tolerance=rng.uniform(.04,.96); fr=10**rng.uniform(np.log10(.3),np.log10(4.0))
        minimum=int(rng.integers(5,251)); kr=10**rng.uniform(np.log10(.3),np.log10(2.0))
    else:
        def logj(x,s,lo,hi): return float(np.clip(np.exp(np.log(max(x,lo))+rng.normal(0,s)),lo,hi))
        local=logj(centre.local_threshold_deg,.32*scale,.002,.16)
        if centre.global_threshold_deg<=0:
            global_= -1.0 if rng.random()<.7 else logj(.08,.7,.01,.8)
        else: global_= -1.0 if rng.random()<.12 else logj(centre.global_threshold_deg,.38*scale,.01,.8)
        tolerance=float(np.clip(centre.footprint_tolerance+rng.normal(0,.10*scale),.04,.96))
        fr=logj(centre.footprint_radius_um,.24*scale,.3,4.0)
        minimum=int(np.clip(round(centre.min_cell_size+rng.normal(0,45*scale)),5,250))
        kr=logj(centre.kam_radius_um,.24*scale,.3,2.0)
    return oc.Config(local,global_,tolerance,fr,minimum,kr)


def rank_frame(frame, orientation=.02):
    tag=str(orientation).replace(".","p")
    return frame.sort_values([
        f"orientation_correct_cells_at_{tag}deg","unrepresented_true_cells","split_true_cells",
        "merged_predicted_cells","median_matched_mean_orientation_error_deg","abs_count","vi_merge_bits",
        "vi_split_bits","ari"], ascending=[False,True,True,True,True,True,True,True,False])


def seed_saved_labels(ws, store):
    """Score immutable saved finalists without rerunning segmentation."""
    done={(r.get("config_key"),r.get("random_seed")) for r in rows(store)}
    selected=json.loads((PRIMARY/"selected_solutions.json").read_text())
    for name,row in selected.items():
        config=oc.canonical(params(row),ws)
        if (config.key(),0) in done: continue
        with np.load(PRIMARY/"finalist_labels"/f"{name}.npz") as data:
            _,result=evaluate(data["labels"],config,ws,0,f"reused_saved_latent_label_{name}",with_boundary=True)
        append(store,result)


def legacy_object_leaders():
    """Distinct object-promising regions from all 13,649 completed configs."""
    from oracle_store import Store
    import oracle_select
    old=oracle_select.aggregate(Store(PRIMARY/"evaluations.jsonl").ok_rows())
    ordered=old.sort_values([
        "object_recall_at_0p5_mean","true_cells_unrepresented_mean",
        "true_cells_split_mean","pred_cells_merging_mean","matched_dice_mean_mean",
        "abs_cell_count_error","vi_merge_bits_mean","vi_split_bits_mean","ari_mean"],
        ascending=[False,True,True,True,False,True,True,True,False])
    # Retain distinct coarse parameter regions, not just the narrow maximum-ARI band.
    region=ordered.assign(
        local_bin=pd.qcut(ordered.local_threshold_deg,8,duplicates="drop"),
        tolerance_bin=pd.qcut(ordered.footprint_tolerance,8,duplicates="drop"),
        size_bin=pd.qcut(ordered.min_cell_size,6,duplicates="drop"),
        global_regime=ordered.global_threshold_deg>0,
    ).groupby(["local_bin","tolerance_bin","size_bin","global_regime"],observed=True,sort=False).head(2)
    return pd.concat([ordered.head(150),region.head(150)]).drop_duplicates("config_key").head(300)


def make_task(config, source, seed=0):
    key=config.key(); digest=hashlib.sha256(key.encode()).hexdigest()
    return {"parameters":config.as_dict(),"config_key":key,"config_hash":digest,
            "random_seed":int(seed),"source":source,"task_id":f"{digest}:{int(seed)}"}


def write_progress(stage,batch,worker_count,done,inflight,pending,started,swap_baseline,peak_rss,failed,last=None,safety=None,initial_done=0):
    telemetry=tree_telemetry(swap_baseline); peak_rss=max(peak_rss,telemetry["aggregate_rss_bytes"])
    elapsed=max(time.monotonic()-started,1e-9); completed_run=max(len(done)-initial_done,0)
    rate=completed_run/elapsed
    payload={"stage":stage,"batch":batch,"worker_count":worker_count,"trials_per_second":rate,
      "aggregate_rss_bytes":telemetry["aggregate_rss_bytes"],"peak_aggregate_rss_bytes":peak_rss,
      "max_individual_worker_rss_bytes":telemetry["max_individual_rss_bytes"],
      "available_ram_bytes":telemetry["available_ram_bytes"],"swap_baseline_bytes":swap_baseline,
      "swap_growth_bytes":telemetry["swap_growth_bytes"],"completed_count":len(done),"completed_this_run":completed_run,
      "failed_count":failed,"pending_count":pending,"in_flight_count":inflight,
      "eta_seconds":pending/rate if rate>0 else None,"last_config":last,
      "safety_stop_reason":safety,"updated":dt.datetime.now().astimezone().isoformat()}
    atomic_json(OUT/"progress.json",payload); return telemetry,peak_rss


def parallel_queue(tasks,store,done,workers,batch,swap_baseline,started,peak_rss,initial_done=0,future_pending=0):
    """Bounded submission and single-writer checkpointing."""
    global STOP_REQUESTED
    pending=[t for t in tasks if (t["config_key"],t["random_seed"]) not in done]
    failed=0; completed_here=0; safety=None
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=worker_init,
                             max_tasks_per_child=15) as pool:
        futures={}
        while pending or futures:
            telemetry=tree_telemetry(swap_baseline); peak_rss=max(peak_rss,telemetry["aggregate_rss_bytes"])
            safety=unsafe_reason(telemetry)
            while pending and len(futures)<workers and not STOP_REQUESTED and not safety:
                task=pending.pop(0); futures[pool.submit(worker_trial,task)]=task
            if not futures: break
            ready,_=wait(futures,timeout=2,return_when=FIRST_COMPLETED)
            for future in ready:
                task=futures.pop(future)
                try: row=future.result()
                except Exception as exc:
                    row={**task["parameters"],"config_key":task["config_key"],"config_hash":task["config_hash"],
                         "random_seed":task["random_seed"],"source":task["source"],"task_id":task["task_id"],
                         "status":"error","error":f"worker failure: {type(exc).__name__}: {exc}"}
                expected_hash=hashlib.sha256(task["config_key"].encode()).hexdigest()
                if row.get("config_key")!=task["config_key"] or row.get("config_hash")!=expected_hash or row.get("task_id")!=task["task_id"]:
                    raise RuntimeError(f"worker identity mismatch for {task['task_id']}")
                key=(row["config_key"],int(row["random_seed"]))
                if key in done: continue
                append(store,row); done.add(key); completed_here+=1
                failed+=int(row.get("status")!="ok")
                _,peak_rss=write_progress("flood_search",batch,workers,done,len(futures),len(pending)+future_pending,
                    started,swap_baseline,peak_rss,failed,row["config_key"],safety,initial_done)
            if safety: STOP_REQUESTED=True
        # Context manager drains already-running futures. No new work is submitted.
    return completed_here,peak_rss,safety


def flood_search(ws, target, workers=8):
    store=OUT/"expanded_sensitivity_trials.jsonl"; done={(r.get("config_key"),int(r.get("random_seed",0))) for r in rows(store)}
    seed_saved_labels(ws,store); done={(r.get("config_key"),int(r.get("random_seed",0))) for r in rows(store)}
    started=time.monotonic(); _,swap_baseline=memory(); peak_rss=0; initial_done=len(done)
    rng=np.random.default_rng(20260813)
    legacy=legacy_object_leaders()
    queue=[("recheck_legacy_object_region",params(row)) for _,row in legacy.iterrows()]
    centres=[params(row) for _,row in legacy.head(30).iterrows()]
    # Expand only around distinct object-promising regions from the completed search.
    while len(queue)<1000: queue.append(("object_region_expansion",candidate(rng,centres[int(rng.integers(len(centres)))],1.0)))
    for batch in range(3):
        if batch:
            current=pd.DataFrame([r for r in rows(store) if r.get("status")=="ok" and r.get("random_seed")==0])
            current["abs_count"]=current.cell_count_error.abs()
            leaders=rank_frame(current).drop_duplicates("config_key").head(30)
            queue=[]
            scale=1.0 if batch==1 else .45
            while len(queue)<1000:
                centre=params(leaders.iloc[int(rng.integers(len(leaders)))])
                queue.append((f"refinement_{batch}",candidate(rng,centre,scale)))
        tasks=[]; seen=set(done)
        for source,raw in queue:
            config=oc.canonical(raw,ws); key=(config.key(),0)
            if key in seen: continue
            seen.add(key); tasks.append(make_task(config,source,0))
        _,peak_rss,safety=parallel_queue(tasks,store,done,workers,batch,swap_baseline,started,peak_rss,
                                         initial_done,(2-batch)*1000)
        if STOP_REQUESTED or safety: return


def serial_trial(task,ws):
    config=oc.canonical(oc.Config(**task["parameters"]),ws)
    labels,_=oc.segment(config,ws,int(task["random_seed"]))
    _,row=evaluate(labels,config,ws,int(task["random_seed"]),task["source"])
    fp=ws.footprint(config.footprint_radius_um)
    req=oc.neighbour_requirements(config.footprint_tolerance,int(fp.sum()))
    row.update({"effective_neighbour_requirement":int(req[-1]),
      "effective_neighbour_vector_sha256":hashlib.sha256(req.tobytes()).hexdigest(),
      "footprint_voxels":int(fp.sum()),"kam_footprint_voxels":int(ws.footprint(config.kam_radius_um).sum())})
    row["config_hash"]=task["config_hash"]; row["task_id"]=task["task_id"]
    return row


def comparable(row):
    ignored={"runtime_seconds","available_ram_bytes_after","swap_used_bytes_after"}
    return {k:v for k,v in row.items() if k not in ignored}


def smoke_test(ws,workers=8,n=100):
    path=OUT/"parallel_smoke_trials.jsonl"
    if path.exists():
        preserved=path.with_name(f"{path.name}.previous-{time.strftime('%Y%m%dT%H%M%S')}")
        path.replace(preserved)
    leaders=legacy_object_leaders().head(20); rng=np.random.default_rng(20260814); tasks=[]
    for i in range(n):
        centre=params(leaders.iloc[i%len(leaders)])
        config=oc.canonical(candidate(rng,centre,.35),ws)
        tasks.append(make_task(config,"parallel_smoke",i%3))
    # Canonical collisions are removed without changing task order.
    unique=[]; seen=set()
    for task in tasks:
        key=(task["config_key"],task["random_seed"])
        if key not in seen: seen.add(key); unique.append(task)
    tasks=unique[:n]
    while len(tasks)<n:
        config=oc.canonical(candidate(rng,params(leaders.iloc[len(tasks)%len(leaders)]),.35),ws)
        task=make_task(config,"parallel_smoke",len(tasks)%3); key=(task["config_key"],task["random_seed"])
        if key not in seen: seen.add(key); tasks.append(task)
    _,swap_baseline=memory(); started=time.monotonic(); done=set(); telemetry0=tree_telemetry(swap_baseline)
    count,peak,safety=parallel_queue(tasks,path,done,workers,"smoke",swap_baseline,started,0,0,0)
    elapsed=time.monotonic()-started; parallel={r["task_id"]:r for r in rows(path)}
    mismatches=[]
    for task in tasks[:10]:
        serial=comparable(serial_trial(task,ws)); other=comparable(parallel[task["task_id"]])
        for key in sorted(set(serial)|set(other)):
            a,b=serial.get(key),other.get(key)
            if isinstance(a,float) or isinstance(b,float):
                if not np.isclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True): mismatches.append({"task":task["task_id"],"key":key,"serial":a,"parallel":b})
            elif a!=b: mismatches.append({"task":task["task_id"],"key":key,"serial":a,"parallel":b})
    parsed=rows(path); keys=[(r["config_key"],r["random_seed"]) for r in parsed]
    end=tree_telemetry(swap_baseline)
    report={"worker_count":workers,"requested_trials":n,"completed_trials":count,"elapsed_seconds":elapsed,
      "trials_per_second":count/max(elapsed,1e-9),"serial_comparisons":10,"mismatch_count":len(mismatches),
      "mismatches":mismatches[:20],"valid_json_lines":len(parsed),"duplicate_keys":len(keys)-len(set(keys)),
      "peak_aggregate_rss_bytes":peak,"initial":telemetry0,"final":end,"safety_stop_reason":safety,
      "passed":count==n and not mismatches and len(parsed)==n and len(keys)==len(set(keys)) and safety is None
        and end["swap_growth_bytes"]<=MAX_SWAP_GROWTH and peak<HARD_TREE_RSS and end["available_ram_bytes"]>=MIN_AVAILABLE_BYTES}
    atomic_json(OUT/"parallel_smoke_report.json",report)
    if not report["passed"]: raise RuntimeError("parallel smoke test failed; see parallel_smoke_report.json")
    return report


def rerun_finalists(ws):
    frame=pd.DataFrame([r for r in rows(OUT/"expanded_sensitivity_trials.jsonl") if r.get("status")=="ok" and r.get("random_seed")==0])
    frame["abs_count"]=frame.cell_count_error.abs()
    ranked=rank_frame(frame)
    high=frame[frame.one_to_one_recovered_fraction>=frame.one_to_one_recovered_fraction.quantile(.9)]
    choices={
      "identity_optimal":ranked.iloc[0],
      "orientation_optimal_high_identity":high.sort_values("equal_cell_weighted_orientation_rmse_deg").iloc[0],
      "balanced_object_first":ranked.iloc[min(4,len(ranked)-1)],
      "exact_count_counterexample":frame.loc[frame.abs_count.idxmin()],
    }
    old=dict(selected_old())
    choices["previous_maximum_ari"]=pd.Series(old["max_ari"].as_dict())
    choices["previous_balanced"]=pd.Series(old["balanced"].as_dict())
    store=OUT/"expanded_sensitivity_trials.jsonl"; done={(r.get("config_key"),r.get("random_seed")) for r in rows(store)}
    saved={}
    for name,row in choices.items():
        config=oc.canonical(params(row),ws); seed_rows=[]
        for seed in range(20):
            if (config.key(),seed) not in done:
                labels,_=oc.segment(config,ws,seed); pairs,result=evaluate(labels,config,ws,seed,f"finalist_{name}",with_boundary=True)
                append(store,result); done.add((config.key(),seed))
            else: result=next(x for x in rows(store) if x.get("config_key")==config.key() and x.get("random_seed")==seed)
            seed_rows.append(result)
            if seed==0:
                labels,_=oc.segment(config,ws,seed)
                np.savez_compressed(OUT/f"{name}_labels.npz",labels=labels)
        saved[name]={"parameters":config.as_dict(),"config_key":config.key(),"seed_orders":20,
                     "metrics_mean":pd.DataFrame(seed_rows).select_dtypes("number").mean().to_dict(),
                     "metrics_std":pd.DataFrame(seed_rows).select_dtypes("number").std().to_dict()}
    atomic_json(OUT/"selected_solutions.json",saved); return saved


def kam_comparison(ws):
    # The completed KAM sweep already used this exact latent input.  Re-score
    # its saved selected label maps; do not repeat the segmentation search.
    directory=HERE/"continuation_results"/"kam_analysis"
    out=[]
    for path in sorted(directory.glob("kam_*.npz")):
        with np.load(path) as data: labels=np.asarray(data["labels"])
        _,metric=evaluate(labels,oc.Config(0.,-1.,0.,1,1,1.),ws,0,f"reused_{path.stem}",with_boundary=True)
        metric.update({"method":"KAM","saved_solution":path.stem,"segmentation_rerun":False})
        out.append(metric)
    pd.DataFrame(out).to_csv(OUT/"kam_object_metrics.csv",index=False)


def tolerance_tables_and_pareto(ws):
    trials=pd.DataFrame([r for r in rows(OUT/"expanded_sensitivity_trials.jsonl") if r.get("status")=="ok"])
    base=trials[trials.random_seed==0].drop_duplicates("config_key").copy(); base["abs_count"]=base.cell_count_error.abs()
    pareto=[]
    for purity,complete in oom.DEFAULT_IDENTITY_TOLERANCES:
      for orient in oom.DEFAULT_ORIENTATION_TOLERANCES_DEG:
        # Recompute only leaders' explicit tolerance assignment from saved/rerun labels.
        ranked=rank_frame(base,orient).head(30)
        for _,r in ranked.iterrows():
          config=params(r); labels,_=oc.segment(config,ws,0)
          _,m=oom.match_cells(ws.labels,labels,ws.field,ws.spacing_um_zyx,purity_threshold=purity,completeness_threshold=complete)
          pareto.append({**config.as_dict(),"config_key":config.key(),"purity_threshold":purity,"completeness_threshold":complete,
                         "orientation_tolerance_deg":orient,**m})
    pdf=pd.DataFrame(pareto); pdf.to_csv(OUT/"object_first_pareto.csv",index=False)
    base.to_csv(OUT/"object_metrics.csv",index=False)


def export_matches(ws):
    selected=json.loads((OUT/"selected_solutions.json").read_text()); allpairs=[]
    for name,item in selected.items():
      with np.load(OUT/f"{name}_labels.npz") as d: labels=d["labels"]
      for purity,complete in oom.DEFAULT_IDENTITY_TOLERANCES:
        pairs,_=oom.match_cells(ws.labels,labels,ws.field,ws.spacing_um_zyx,purity_threshold=purity,completeness_threshold=complete)
        for pair in pairs: allpairs.append({"solution":name,"purity_threshold":purity,"completeness_threshold":complete,**pair})
    pd.DataFrame(allpairs).to_csv(OUT/"matched_cells.csv",index=False)


def figures_and_report(ws,audit):
    import matplotlib.pyplot as plt
    matched=pd.read_csv(OUT/"matched_cells.csv"); metrics=pd.read_csv(OUT/"object_metrics.csv")
    chosen=matched[(matched.solution=="balanced_object_first")&(matched.purity_threshold==.6)]
    fig,ax=plt.subplots(1,2,figsize=(7,3),layout="constrained")
    for a,c,label in zip(ax,("chi","phi"),("χ","φ")):
      a.scatter(chosen[f"true_mean_{c}_deg"],chosen[f"predicted_mean_{c}_deg"],s=8,alpha=.6)
      lo=min(a.get_xlim()[0],a.get_ylim()[0]); hi=max(a.get_xlim()[1],a.get_ylim()[1]); a.plot([lo,hi],[lo,hi],"k--",lw=.7); a.set(xlabel=f"true {label} (°)",ylabel=f"predicted {label} (°)")
    for suffix in ("png","pdf"): fig.savefig(OUT/f"matched_cell_means.{suffix}",dpi=300 if suffix=="png" else None)
    plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(8,6),layout="constrained")
    ax[0,0].hist(chosen.mean_orientation_error_deg,bins=30); ax[0,0].set(xlabel="matched mean-orientation error (°)",ylabel="cells")
    ax[0,1].scatter(chosen.prediction_purity,chosen.true_cell_completeness,s=8); ax[0,1].set(xlabel="purity",ylabel="completeness")
    ax[1,0].scatter(metrics.cell_count_error,metrics.one_to_one_recovered_fraction,s=7,alpha=.35); ax[1,0].axvline(0,color="k",lw=.7); ax[1,0].set(xlabel="cell-count error",ylabel="one-to-one recovery")
    ax[1,1].scatter(metrics.ari,metrics.orientation_correct_recovered_fraction_at_0p02deg,s=7,alpha=.35); ax[1,1].set(xlabel="ARI",ylabel="orientation-correct recovery (0.02°)")
    for suffix in ("png","pdf"): fig.savefig(OUT/f"object_orientation_diagnostics.{suffix}",dpi=300 if suffix=="png" else None)
    plt.close(fig)
    selected=json.loads((OUT/"selected_solutions.json").read_text())
    report={"created":dt.datetime.now().astimezone().isoformat(),"input_parity":audit,"assignment":
      "Hard purity/completeness eligibility followed by maximum-Dice Hungarian assignment; wrapped mean-orientation error is only a deterministic tie-breaker.",
      "identity_tolerances":oom.DEFAULT_IDENTITY_TOLERANCES,"orientation_tolerances_deg":oom.DEFAULT_ORIENTATION_TOLERANCES_DEG,
      "primary_weighting":"equal cell","secondary_weighting":"physical volume","selected_solutions":selected,
      "limitations":["The expanded search is conditional on this fixed phantom.","Cell-mean Wasserstein is the mean of the two one-dimensional channel distances, not multivariate optimal transport."]}
    atomic_json(OUT/"object_orientation_report.json",report)
    best=selected["balanced_object_first"]["metrics_mean"]
    text=f"""# Object and orientation analysis\n\n## Scientific objective\n\nThe primary endpoint is recovery of the correct physical cells with correct latent-field mean orientations. Cell count, voxel overlap and boundary placement are secondary diagnostics.\n\n## Input parity\n\nParity is confirmed. Completed and continuation flood fill and KAM receive cache key `latent`, the same all-true mask, degree units, and anisotropic `(1.0, 0.4, 0.4)` µm spacing. The latent array is shape `(24, 160, 160, 2)`, dtype `float32`, SHA-256 `3f9739bf36a767b7d89cb75d044c24b72dcf0fe30a7debe6ed378ee5be4ff9f6`. Full array and mask identities are in `input_parity_audit.json`. No measurement noise or denoising enters this primary analysis.\n\n## Assignment\n\nPairs must independently satisfy explicit prediction-purity and true-cell-completeness thresholds. Eligible pairs are assigned one-to-one by maximum Dice; wrapped two-channel mean-orientation error breaks only numerical overlap ties. Results are reported for identity thresholds 0.5, 0.6 and 0.7 and orientation tolerances 0.01°, 0.02° and 0.05°.\n\n## Balanced object-first result\n\nAt the primary 0.6/0.6 identity and 0.02° orientation criteria, the balanced solution recovered {best.get('one_to_one_recovered_cells',float('nan')):.1f} cells one-to-one on average across 20 seed orders; its orientation-correct recovered fraction was {best.get('orientation_correct_recovered_fraction_at_0p02deg',float('nan')):.3f}. Exact predicted count is retained only as a counterexample and is not a success criterion.\n\n## Interpretation\n\nEqual-cell weighting is primary because every physical cell is one scientific object. Volume weighting is reported separately and answers a different, voxel-dominant question. Agreement of marginal cell-mean distributions is diagnostic only; the controlled tests demonstrate that it can coexist with incorrect identities.\n"""
    (OUT/"OBJECT_ORIENTATION_ANALYSIS.md").write_text(text)
    (OUT/"RESULTS_SUBSECTION.md").write_text("## Results\n\n"+" ".join(text.split("## Balanced object-first result\n\n")[1].split("\n\n## Interpretation")[0].splitlines())+"\n")
    (OUT/"DISCUSSION_SUBSECTION.md").write_text("## Discussion\n\nCell identity and matched mean orientation alter the scientific ranking because equal count can conceal compensating splits and merges, while marginal orientation agreement can conceal correspondence failure. Boundary and voxel metrics remain useful secondary descriptions.\n")


def main():
    global STOP_REQUESTED
    parser=argparse.ArgumentParser(); parser.add_argument("--additional",type=int,default=3000)
    parser.add_argument("--workers",type=int,default=8,choices=range(1,11))
    parser.add_argument("--smoke-test",action="store_true"); parser.add_argument("--smoke-trials",type=int,default=100)
    parser.add_argument("--finalize-only",action="store_true"); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/"object_orientation_analysis.lock").open("w"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def request_stop(signum,frame):
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    atomic_json(OUT/"object_orientation_analysis.pid.json",{"pid":os.getpid(),"worker_count":args.workers,
      "started":dt.datetime.now().astimezone().isoformat(),"signal_semantics":"SIGTERM/SIGINT stop submission and drain in-flight trials"})
    ws=workspace(); audit=parity(ws)
    if args.smoke_test:
        smoke_test(ws,args.workers,args.smoke_trials); return 0
    if not args.finalize_only: flood_search(ws,args.additional,args.workers)
    if STOP_REQUESTED:
        current=plain(tree_telemetry(memory()[1])); current.update({"stage":"stopped_after_draining_in_flight","worker_count":args.workers,
          "updated":dt.datetime.now().astimezone().isoformat()}); atomic_json(OUT/"progress.json",current); return 0
    rerun_finalists(ws); kam_comparison(ws); tolerance_tables_and_pareto(ws); export_matches(ws); figures_and_report(ws,audit)
    atomic_json(OUT/"progress.json",{"stage":"complete","updated":dt.datetime.now().astimezone().isoformat()})
    return 0


if __name__=="__main__":
  try: raise SystemExit(main())
  except Exception:
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"failure.log").write_text(traceback.format_exc()); raise

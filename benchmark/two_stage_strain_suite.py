#!/usr/bin/env python3
"""Resumable two-stage object-first optimisation on 36 preserved phantoms.

The source phantom archives are read-only inputs.  KAM/radius caches, scalar
trials, and selected label volumes are written only below
``continuation_results/two_stage_strain_suite``.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,multiprocessing as mp,os,signal,sys,time
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[name]='1'
os.environ['MALLOC_ARENA_MAX']='2'
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; SOURCE=HERE/'continuation_results'/'phantoms'; OUT=HERE/'continuation_results'/'two_stage_strain_suite'; PRIMARY=HERE/'two_stage_oracle_results'; sys.path.insert(0,str(HERE))
import object_orientation_metrics as oom,oracle_core as oc,oracle_metrics as om,pipelines
from two_stage_store import ResultStore,normalize_result,utc_now
ALGORITHM='size_prioritised_multiseed_v1'; MAX_ITERATIONS=700000; WORK=None; ID=None; STOP=False

def plain(x):
 if isinstance(x,np.generic): return x.item()
 if isinstance(x,float) and not np.isfinite(x): return None
 if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [plain(v) for v in x]
 return x
def atomic(path,value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(plain(value),indent=2,sort_keys=True)); tmp.replace(path)
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def workspace(source_cache,output_cache):
 with np.load(Path(source_cache)/'phantom.npz') as z: labels=np.ascontiguousarray(z['labels']); field=np.ascontiguousarray(z['latent']); spacing=tuple(float(v) for v in z['spacing'])
 output_cache=Path(output_cache); output_cache.mkdir(parents=True,exist_ok=True)
 return oc.Workspace(labels,field,np.ones(labels.shape,bool),spacing,output_cache,oc.RadiusClasses(spacing,output_cache),{})
def init_worker(source_cache,output_cache,identity):
 global WORK,ID; WORK=workspace(source_cache,output_cache); ID=identity
def key(parameters,seed,stage):
 payload={'algorithm':ALGORITHM,'python_source_sha256':ID['python_source_sha256'],'compiled_extension_sha256':ID['compiled_extension_sha256'],'latent_sha256':ID['latent_sha256'],'truth_sha256':ID['truth_sha256'],'mask_sha256':ID['mask_sha256'],'matching_metrics_version':2,'matching_source_sha256':ID['matching_source_sha256'],'parameters':parameters,'candidate_order_seed':int(seed),'max_iterations':MAX_ITERATIONS,'stagnation_tolerance':2000,'recycle_small_grains':False,'watershed_connectivity':1}
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(),payload
def evaluate(task):
 import resource
 p=task['parameters']; seed=int(task['seed']); digest,identity=key(p,seed,task['stage']); started=utc_now(); clock=time.perf_counter(); raw={'config_hash':digest,'configuration_identity':identity,'algorithm':ALGORITHM,'candidate_order_seed':seed,'stage':task['stage'],**p}
 try:
  cfg=oc.Config(**p); fp=WORK.footprint(cfg.footprint_radius_um); labels,markers,initial,diag,final_sizes=pipelines.run_flood_fill_two_stage(WORK.field,WORK.mask,WORK.kam(cfg.kam_radius_um),fp,local_threshold_deg=cfg.local_threshold_deg,global_threshold_deg=cfg.global_threshold_deg,footprint_tolerance=cfg.footprint_tolerance,min_cell_size=cfg.min_cell_size,max_seed_attempts=MAX_ITERATIONS,stagnation_tolerance=2000,random_seed=seed,watershed_connectivity=1,recycle_small_grains=False); candidate=plain(diag.get('candidate_collection') or {}); final=plain(diag.get('final_growth') or {}); raw.update({'candidate_pass_saturated':bool(candidate.get('max_iterations_reached',False)),'final_pass_saturated':bool(final.get('max_iterations_reached',False)),'preliminary_candidates_detected':int(len(initial)),'accepted_marker_count':int(markers.max()),'candidate_seeds_skipped_claimed':int(final.get('user_seeds_skipped_claimed',0)),'unlabelled_marker_stage_fraction':float(np.mean(markers==0))})
  if int(markers.max())==0: raise RuntimeError('No valid seeds found or no accepted markers')
  _,obj=oom.match_cells(WORK.labels,labels,WORK.field,WORK.spacing_um_zyx,purity_threshold=.6,completeness_threshold=.6); raw.update(obj); raw.update(om.evaluate_partition(WORK.labels,labels,WORK.spacing_um_zyx,with_boundary=task.get('boundary',False))); raw.update({f'marker_{k}':v for k,v in om.marker_errors(markers,WORK.labels).items()}); raw['status']='ok'
 except Exception as exc: raw.update(status='error',error_type=type(exc).__name__,error_message=str(exc))
 finally: raw.update(worker_pid=os.getpid(),worker_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),started=started,finished=utc_now(),elapsed_seconds=time.perf_counter()-clock,scientific_metrics_version=2)
 return normalize_result(raw,identity)
def memory():
 d={}
 for line in Path('/proc/meminfo').read_text().splitlines(): k,v=line.split(':',1); d[k]=int(v.strip().split()[0])*1024
 return d['MemAvailable'],d['MemTotal'],d['SwapTotal']-d['SwapFree']
def tree_rss():
 import psutil
 root=psutil.Process(os.getpid()); values=[]
 for process in [root]+root.children(recursive=True):
  try: values.append(process.memory_info().rss)
  except (psutil.NoSuchProcess,psutil.AccessDenied): pass
 return sum(values),max(values,default=0)
def canonical(raw,w): return oc.canonical(oc.Config(**raw),w).as_dict()
def broad(w,n,seed):
 from scipy.stats import qmc
 points=qmc.Sobol(6,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n*2)))); out=[]; seen=set()
 for index,v in enumerate(points):
  raw={'local_threshold_deg':float(10**(-2.8+2.2*v[0])),'global_threshold_deg':float(-1 if v[1]<.3 else 10**(-2.3+2.4*v[1])),'footprint_tolerance':float(.03+.93*v[2]),'footprint_radius_um':float(10**(np.log10(.3)+v[3]*np.log10(4/.3))),'min_cell_size':int(5+395*v[4]),'kam_radius_um':float(10**(np.log10(.3)+v[5]*np.log10(2.6/.3)))}; p=canonical(raw,w); fp=w.footprint(p['footprint_radius_um']); signature=(p['local_threshold_deg'],p['global_threshold_deg'],hashlib.sha256(oc.neighbour_requirements(p['footprint_tolerance'],int(fp.sum())).tobytes()).hexdigest(),p['footprint_radius_um'],p['min_cell_size'],p['kam_radius_um'])
  if signature in seen: continue
  seen.add(signature); out.append(p)
  if len(out)==n: break
 return out
def rank(rows):
 good=[r for r in rows if r.get('status_category')=='ok' and r.get('scientific_metrics_version')==2]
 return sorted(good,key=lambda r:(-r['orientation_correct_cells_at_0p02deg'],-r['one_to_one_recovered_cells'],r['unmatched_true_cells'],r['unmatched_predictions'],r['merged_predicted_cells'],r['split_true_cells'],abs(r['cell_count_error']),r['median_matched_mean_orientation_error_deg'],r['vi_total_bits'],-r['ari']))
def adaptive(rows,w,n,seed):
 leaders=rank(rows)[:12]; rng=np.random.default_rng(seed); out=[]; seen=set()
 while len(out)<n:
  centre=leaders[int(rng.integers(len(leaders)))]; p={name:centre[name] for name in oc.PARAMETER_NAMES}
  for name,low,high in [('local_threshold_deg',.001,.3),('footprint_radius_um',.3,4.),('kam_radius_um',.3,2.6)]: p[name]=float(np.clip(np.exp(np.log(p[name])+rng.normal(0,.3)),low,high))
  p['global_threshold_deg']=-1. if p['global_threshold_deg']<=0 and rng.random()<.7 else float(np.clip(np.exp(np.log(max(p['global_threshold_deg'],.005))+rng.normal(0,.3)),.005,2.)); p['footprint_tolerance']=float(np.clip(p['footprint_tolerance']+rng.normal(0,.08),.03,.96)); p['min_cell_size']=int(np.clip(round(p['min_cell_size']+rng.normal(0,55)),5,400)); p=canonical(p,w); signature=oc.Config(**p).key()
  if signature not in seen: seen.add(signature); out.append(p)
 return out
def precompute_kam(w,parameters):
 for radius in sorted({p['kam_radius_um'] for p in parameters}): w.kam(radius)
 w.clear_kam_cache()
def run_batch(parameters,seeds,stage,source_cache,output_cache,identity,store,workers,boundary=False):
 global STOP
 w=workspace(source_cache,output_cache); precompute_kam(w,parameters); tasks=[]
 for p in parameters:
  for seed in seeds:
   digest,_=key_for(identity,p,seed,stage)
   if digest not in store.hashes: tasks.append({'parameters':p,'seed':seed,'stage':stage,'boundary':boundary})
 available,total,swap0=memory(); initial=len(store.rows); started=time.monotonic(); ctx=mp.get_context('spawn')
 with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_worker,initargs=(str(source_cache),str(output_cache),identity),max_tasks_per_child=15) as pool:
  futures={}
  while tasks or futures:
   available,total,swap=memory(); aggregate,maximum=tree_rss(); growth=max(0,swap-swap0); unsafe=aggregate>=10*1024**3 or available<max(6*1024**3,.2*total) or growth>512*1024**2
   if unsafe or aggregate>=12*1024**3: STOP=True
   while tasks and len(futures)<workers and not STOP:
    task=tasks.pop(0); futures[pool.submit(evaluate,task)]=task
   if not futures: break
   ready,_=wait(futures,timeout=2,return_when=FIRST_COMPLETED)
   for future in ready:
    task=futures.pop(future); digest,payload=key_for(identity,task['parameters'],task['seed'],task['stage']); row=future.result()
    if row.get('config_hash')!=digest: row=normalize_result({**row,'config_hash':digest,'status':'error','status_category':'coordinator/schema_error','error_type':'SchemaError','error_message':'strain-suite identity mismatch'},payload)
    store.append(row,payload)
   elapsed=max(time.monotonic()-started,1e-9); atomic(Path(output_cache).parent/'progress.json',{'stage':stage,'completed':len(store.rows),'completed_this_batch':len(store.rows)-initial,'pending':len(tasks),'in_flight':len(futures),'counts':store.counts(),'trials_per_second':(len(store.rows)-initial)/elapsed,'aggregate_rss_bytes':aggregate,'maximum_process_rss_bytes':maximum,'available_ram_bytes':available,'swap_growth_bytes':growth,'safety_stop':STOP})
 return list(store.rows)
def key_for(identity,p,seed,stage):
 global ID; previous=ID; ID=identity
 try: return key(p,seed,stage)
 finally: ID=previous
def phantom_identity(source_cache):
 audit=json.loads((PRIMARY/'implementation_audit.json').read_text())
 with np.load(Path(source_cache)/'phantom.npz') as z: latent=np.ascontiguousarray(z['latent']); truth=np.ascontiguousarray(z['labels']); mask=np.ones(truth.shape,np.uint8)
 return {**audit,'latent_sha256':sha(latent),'truth_sha256':sha(truth),'mask_sha256':sha(mask)}
def select_summary(rows):
 frame=pd.DataFrame([r for r in rows if r.get('status_category')=='ok' and r.get('scientific_metrics_version')==2]); metrics=['one_to_one_recovered_cells','orientation_correct_cells_at_0p02deg','unmatched_true_cells','unmatched_predictions','merged_predicted_cells','split_true_cells','cell_count_error','median_matched_mean_orientation_error_deg','vi_total_bits','ari']; grouped=frame.groupby(list(oc.PARAMETER_NAMES),dropna=False)[metrics].agg(['mean','std','min','max']).reset_index(); grouped.columns=['_'.join(c).rstrip('_') for c in grouped.columns]; ordered=grouped.sort_values(['orientation_correct_cells_at_0p02deg_mean','one_to_one_recovered_cells_mean','unmatched_true_cells_mean','unmatched_predictions_mean','merged_predicted_cells_mean','split_true_cells_mean','cell_count_error_mean','median_matched_mean_orientation_error_deg_mean','vi_total_bits_mean','ari_mean'],ascending=[False,False,True,True,True,True,True,True,True,False]); best=ordered.iloc[0]; out={name:(int(best[name]) if name=='min_cell_size' else float(best[name])) for name in oc.PARAMETER_NAMES}; out['n_seed_orders']=int(len(frame[(frame[list(oc.PARAMETER_NAMES)]==best[list(oc.PARAMETER_NAMES)].values).all(axis=1)])); out.update({f'{metric}_{stat}':float(best[f'{metric}_{stat}']) for metric in metrics for stat in ('mean','std','min','max')}); return out
def run_phantom(source_path,workers,broad_n,adaptive_n):
 name=source_path.name; out=OUT/'phantoms'/name; out.mkdir(parents=True,exist_ok=True); source_cache=source_path/'cache'; output_cache=out/'cache'; identity=phantom_identity(source_cache); global ID; ID=identity; w=workspace(source_cache,output_cache); seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16); broad_parameters=broad(w,broad_n,seed); store=ResultStore(out/'evaluations.jsonl'); rows=run_batch(broad_parameters,[0],'broad',source_cache,output_cache,identity,store,workers); refine=adaptive(rows,w,adaptive_n,seed+1); rows=run_batch(refine,[0],'adaptive',source_cache,output_cache,identity,store,workers); finalists=[]
 for row in rank(rows):
  p={k:row[k] for k in oc.PARAMETER_NAMES}
  if oc.Config(**p).key() not in {oc.Config(**x).key() for x in finalists}: finalists.append(p)
  if len(finalists)==12: break
 rows=run_batch(finalists,range(5),'five_seed_finalists',source_cache,output_cache,identity,store,workers,boundary=True); selected=select_summary(rows); atomic(out/'selected_solution.json',selected); cfg={k:selected[k] for k in oc.PARAMETER_NAMES}; init_worker(str(source_cache),str(output_cache),identity); labels,markers,initial,diag,sizes=pipelines.run_flood_fill_two_stage(WORK.field,WORK.mask,WORK.kam(cfg['kam_radius_um']),WORK.footprint(cfg['footprint_radius_um']),local_threshold_deg=cfg['local_threshold_deg'],global_threshold_deg=cfg['global_threshold_deg'],footprint_tolerance=cfg['footprint_tolerance'],min_cell_size=int(cfg['min_cell_size']),max_seed_attempts=MAX_ITERATIONS,stagnation_tolerance=2000,random_seed=0,watershed_connectivity=1,recycle_small_grains=False); np.savez_compressed(out/'selected_labels.npz',labels=labels,markers=markers,preliminary_sizes=initial,marker_sizes=sizes); return selected
def finalise(workers=8):
 records=[]
 for selected_path in sorted((OUT/'phantoms').glob('*/selected_solution.json')):
  source=SOURCE/selected_path.parent.name; manifest=json.loads((source/'manifest.json').read_text()); selected=json.loads(selected_path.read_text()); records.append({'phantom_id':selected_path.parent.name,'strain_percent':manifest['strain_percent'],'difficulty':manifest['difficulty'],'realization':manifest['realization'],'diagnostic_intradomain_s_k_median_deg':manifest.get('intradomain_s_k_median_deg'),**manifest['measurable_proxies'],**selected})
 frame=pd.DataFrame(records); frame.to_csv(OUT/'two_stage_multi_phantom_summary.csv',index=False)
 if len(frame)<4:
  atomic(OUT/'finalisation_note.json',{'completed_phantoms':len(frame),'rule_validation':'requires at least four completed phantoms'})
  return
 from sklearn.linear_model import Ridge
 predictors=['nn_diff_median_deg','nn_diff_iqr_deg','nn_diff_p90_deg','total_spread_p95_deg','cell_scale_proxy_um','boundary_width_proxy_um','voxel_volume_um3','anisotropy']; predictions=[]
 x=np.log(np.maximum(frame[predictors].to_numpy(float),1e-12))
 for scheme,groups in [('leave_one_phantom_out',frame.phantom_id),('leave_one_strain_out',frame.strain_percent)]:
  for group in pd.unique(groups):
   test=np.asarray(groups==group); train=~test
   for parameter in oc.PARAMETER_NAMES:
    target=frame[parameter].to_numpy(float); transformed=np.log(np.maximum(target,1e-12)) if parameter!='global_threshold_deg' else target
    model=Ridge(alpha=1.).fit(x[train],transformed[train]); predicted=model.predict(x[test]); predicted=np.exp(predicted) if parameter!='global_threshold_deg' else predicted
    for index,value in zip(frame.index[test],predicted): predictions.append({'scheme':scheme,'held_out_group':group,'phantom_id':frame.loc[index,'phantom_id'],'parameter':parameter,'predicted_value':float(value),'oracle_value':float(target[index])})
 prediction_frame=pd.DataFrame(predictions); prediction_frame.to_csv(OUT/'transferable_rule_cross_validation.csv',index=False)
 # Validate rules on held-out segmentations, not parameter-fit error alone.
 validation=[]
 for scheme_index,scheme in enumerate(('leave_one_phantom_out','leave_one_strain_out')):
  subset=prediction_frame[prediction_frame.scheme==scheme]
  for phantom_id,g in subset.groupby('phantom_id'):
   values={row.parameter:row.predicted_value for row in g.itertuples()}; values['local_threshold_deg']=float(np.clip(values['local_threshold_deg'],.001,.3)); values['global_threshold_deg']=float(values['global_threshold_deg']) if values['global_threshold_deg']>0 else -1.; values['footprint_tolerance']=float(np.clip(values['footprint_tolerance'],.03,.96)); values['footprint_radius_um']=float(np.clip(values['footprint_radius_um'],.3,4.)); values['min_cell_size']=int(np.clip(round(values['min_cell_size']),5,400)); values['kam_radius_um']=float(np.clip(values['kam_radius_um'],.3,2.6)); source_cache=SOURCE/phantom_id/'cache'; output_cache=OUT/'phantoms'/phantom_id/'cache'; identity=phantom_identity(source_cache); global ID; ID=identity; w=workspace(source_cache,output_cache); values=canonical(values,w); store=ResultStore(OUT/'phantoms'/phantom_id/'evaluations.jsonl'); rows=run_batch([values],[991+scheme_index],f'transferable_{scheme}',source_cache,output_cache,identity,store,workers,boundary=True); digest,_=key_for(identity,values,991+scheme_index,f'transferable_{scheme}'); row=next((r for r in rows if r.get('config_hash')==digest),None)
   validation.append({'scheme':scheme,'phantom_id':phantom_id,**values,'status_category':row.get('status_category') if row else 'missing',**({k:row.get(k) for k in ('orientation_correct_cells_at_0p02deg','one_to_one_recovered_cells','unmatched_true_cells','merged_predicted_cells','split_true_cells','cell_count_error','median_matched_mean_orientation_error_deg','ari','vi_total_bits','boundary_f1_at_0p4um')} if row else {})})
 pd.DataFrame(validation).to_csv(OUT/'transferable_rule_held_out_performance.csv',index=False)
 performance_columns=[f'{name}_mean' for name in ('orientation_correct_cells_at_0p02deg','one_to_one_recovered_cells','unmatched_true_cells','merged_predicted_cells','split_true_cells','ari','vi_total_bits')]; performance=frame.groupby(['strain_percent','difficulty'])[performance_columns].agg(['mean','std']).reset_index(); performance.columns=['_'.join(map(str,c)).rstrip('_') for c in performance.columns]; performance.to_csv(OUT/'performance_vs_strain_difficulty.csv',index=False)
 diagnostic=frame.groupby(['strain_percent','difficulty'])[['diagnostic_intradomain_s_k_median_deg']+list(oc.PARAMETER_NAMES)].mean().reset_index(); diagnostic.to_csv(OUT/'ground_truth_diagnostic_parameter_interactions.csv',index=False)
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(9,5.5),layout='constrained')
 for axis,parameter in zip(axes.ravel(),oc.PARAMETER_NAMES):
  for difficulty,g in frame.groupby('difficulty'): axis.scatter(g.strain_percent,g[parameter],s=12,label=difficulty)
  axis.set(xlabel='strain (%)',ylabel=parameter)
 axes[0,0].legend(fontsize=6)
 for ext in ('png','pdf'): fig.savefig(OUT/f'two_stage_parameters_vs_strain.{ext}',dpi=300 if ext=='png' else None)
 plt.close(fig)
 (OUT/'SCIENTIFIC_SUMMARY.md').write_text('# Two-stage strain/difficulty suite\n\nAll 36 preserved phantom geometries and latent fields are read unchanged. Each phantom receives a fresh size-prioritised broad search, object-first refinement, and five seed-order finalist evaluation. Deployable rule inputs are restricted to the label-free measurable proxies listed in `transferable_rule_cross_validation.csv`; intradomain spread is retained only in the explicitly ground-truth-derived diagnostic table.\n')
def main():
 global STOP
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--workers',type=int,default=8); parser.add_argument('--broad',type=int,default=256); parser.add_argument('--adaptive',type=int,default=128); parser.add_argument('--max-phantoms',type=int); a=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True); lock=(OUT/'strain_suite.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True)); signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('STOP',True)); atomic(OUT/'pid.json',{'pid':os.getpid(),'workers':a.workers,'started':utc_now()}); paths=sorted(path.parent.parent for path in SOURCE.glob('*/cache/phantom.npz'))
 for index,path in enumerate(paths):
  if STOP or (a.max_phantoms is not None and index>=a.max_phantoms): break
  run_phantom(path,a.workers,a.broad,a.adaptive); atomic(OUT/'status.json',{'completed_phantoms':index+1,'planned_phantoms':len(paths),'last_phantom':path.name,'stopped':STOP})
 finalise(a.workers)
if __name__=='__main__': main()

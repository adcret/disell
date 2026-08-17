#!/usr/bin/env python3
"""Post-queue completion of selectable schema-v2 broad coverage.

This module is invoked by the downstream analyser only after the active primary
queue has drained.  It never runs concurrently with that queue.  Independent,
deterministically scrambled Sobol replacements are added until 8,000 unique
successful schema-v2 broad trials exist; v1 rows never contribute.
"""
from __future__ import annotations
import fcntl,hashlib,json,math,multiprocessing as mp,os,sys,time
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
from pathlib import Path
import numpy as np
from scipy.stats import qmc
HERE=Path(__file__).resolve().parent; OUT=HERE/'two_stage_oracle_results'; sys.path.insert(0,str(HERE))
import oracle_core as oc,two_stage_oracle as tso
from two_stage_accounting import BROAD_SELECTABLE_TARGET,accounting,is_broad_stage,is_selectable_v2,trial_key,unique_rows
from two_stage_store import ResultStore,normalize_result,utc_now

SCRAMBLE_SEED=20260901
def atomic(path,value):
 path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(json.dumps(value,indent=2,sort_keys=True)); temporary.replace(path)
def load_rows(): return [json.loads(line) for line in (OUT/'evaluations.jsonl').read_text().splitlines() if line.strip()]
def drained():
 progress=json.loads((OUT/'progress.json').read_text())
 return int(progress.get('pending_count',0))==0 and int(progress.get('in_flight_count',0))==0
def effective_signature(parameters,w): return tso.parameter_signature(parameters,w)
def transform(value,w):
 raw={'local_threshold_deg':float(10**(-2.7+1.9*value[0])),'global_threshold_deg':float(-1 if value[1]<.3 else 10**(-2.1+2*value[1])),'footprint_tolerance':float(.04+.92*value[2]),'footprint_radius_um':float(10**(np.log10(.3)+value[3]*np.log10(4/.3))),'min_cell_size':int(5+245*value[4]),'kam_radius_um':float(10**(np.log10(.3)+value[5]*np.log10(2/.3)))}
 return tso.canonical(raw,w)
def replacement_batch(w,rows,count,cursor,batch):
 seen={effective_signature({name:row[name] for name in oc.PARAMETER_NAMES},w) for row in rows if all(name in row for name in oc.PARAMETER_NAMES)}; sampler=qmc.Sobol(6,scramble=True,seed=SCRAMBLE_SEED)
 if cursor: sampler.fast_forward(cursor)
 output=[]; inspected=0
 while len(output)<count:
  points=sampler.random(max(256,2*(count-len(output))))
  for point in points:
   sample_index=cursor+inspected; inspected+=1; parameters=transform(point,w); signature=effective_signature(parameters,w)
   if signature in seen: continue
   seen.add(signature); footprint=w.footprint(parameters['footprint_radius_um']); requirement=oc.neighbour_requirements(parameters['footprint_tolerance'],int(footprint.sum())); parameters['_sampling']={'broad_batch_id':f'broad_v2_replacement_{batch}','sampler_type':'independently scrambled Sobol','scramble_seed':SCRAMBLE_SEED,'original_sample_index':sample_index,'deduplication_reason':None,'effective_integer_neighbour_requirement':int(requirement[-1]),'effective_neighbour_vector_sha256':hashlib.sha256(requirement.tobytes()).hexdigest()}; output.append(parameters)
   if len(output)==count: break
 return output,cursor+inspected
def broad_selectable(rows): return unique_rows([row for row in rows if is_selectable_v2(row) and is_broad_stage(row)],is_selectable_v2)
def limit_identity(parameters,seed,limit):
 payload={'algorithm':tso.ALGORITHM,'python_source_sha256':tso.IDENTITY['python_source_sha256'],'compiled_extension_sha256':tso.IDENTITY['compiled_extension_sha256'],'latent_sha256':tso.IDENTITY['latent_sha256'],'truth_sha256':tso.IDENTITY['truth_sha256'],'mask_sha256':tso.IDENTITY['mask_sha256'],'matching_metrics_version':2,'matching_source_sha256':tso.IDENTITY['matching_source_sha256'],'parameters':parameters,'candidate_order_seed':int(seed),'max_iterations':int(limit),'stagnation_tolerance':tso.STAGNATION,'recycle_small_grains':False,'watershed_connectivity':tso.WATERSHED_CONNECTIVITY}
 digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(); return digest,payload
def init_limit_worker(limit):
 tso.MAX_ITERATIONS=int(limit); tso.init_worker()
def evaluate_at_limit(task):
 tso.MAX_ITERATIONS=int(task['max_iterations'])
 return tso.evaluate_task({'parameters':task['parameters'],'seed':task['seed'],'stage':'candidate_iteration_ceiling_validation','boundary':True})
def validate_higher_limit(flagged,workers,limit):
 """Re-evaluate only competitive near-ceiling trials under a new full identity."""
 tasks=[]; store=ResultStore(OUT/'evaluations.jsonl')
 for row in unique_rows(flagged,is_selectable_v2):
  parameters={name:row[name] for name in oc.PARAMETER_NAMES}; seed=int(row.get('candidate_order_seed',row.get('seed',0))); digest,payload=limit_identity(parameters,seed,limit)
  if digest not in store.hashes: tasks.append({'parameters':parameters,'seed':seed,'max_iterations':int(limit),'config_hash':digest,'identity':payload,'base_config_hash':row['config_hash']})
 if not tasks: return list(store.rows)
 pending=list(tasks); context=mp.get_context('spawn'); started=time.monotonic(); available,total,swap0=tso.memory()
 with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=init_limit_worker,initargs=(limit,),max_tasks_per_child=15) as pool:
  futures={}
  while pending or futures:
   available,total,swap=tso.memory(); aggregate,maximum=tso.tree_rss(); growth=max(0,swap-swap0); unsafe=aggregate>=10*1024**3 or available<max(6*1024**3,.25*total) or growth>256*1024**2
   while pending and len(futures)<workers and not unsafe: task=pending.pop(0); futures[pool.submit(evaluate_at_limit,task)]=task
   if not futures:
    if unsafe: raise RuntimeError('resource safety stop during candidate-limit validation')
    break
   ready,_=wait(futures,timeout=2,return_when=FIRST_COMPLETED)
   for future in ready:
    task=futures.pop(future); digest,payload=task['config_hash'],task['identity']
    try: raw=future.result()
    except Exception as exc: raw={'config_hash':digest,'configuration_identity':payload,'algorithm':tso.ALGORITHM,'candidate_order_seed':task['seed'],'status':'error','error_type':type(exc).__name__,'error_message':str(exc),'worker_pid':None,'worker_peak_rss_bytes':None,'started':None,'finished':utc_now(),'elapsed_seconds':None,'candidate_pass_saturated':False,'final_pass_saturated':False}
    raw={**raw,'config_hash':digest,'configuration_identity':payload,'base_700000_config_hash':task['base_config_hash'],'max_iterations':int(limit),'stage':'candidate_iteration_ceiling_validation'}; row=normalize_result(raw,payload)
    if row['config_hash']!=digest: row=normalize_result({**raw,'config_hash':digest,'status':'error','status_category':'coordinator/schema_error','error_type':'SchemaError','error_message':'higher-limit configuration identity mismatch'},payload)
    store.append(row,payload)
   elapsed=max(time.monotonic()-started,1e-9); atomic(OUT/'candidate_iteration_high_limit_progress.json',{'validated_limit':int(limit),'durable_rows':len(store.rows),'completed_this_run':len(tasks)-len(pending)-len(futures),'pending':len(pending),'in_flight':len(futures),'trials_per_second':(len(tasks)-len(pending)-len(futures))/elapsed,'aggregate_rss_bytes':aggregate,'maximum_process_rss_bytes':maximum,'available_ram_bytes':available,'swap_baseline_bytes':swap0,'swap_growth_bytes':growth,'updated':utc_now()})
 rows=list(store.rows); expected={task['config_hash'] for task in tasks}; finished=[row for row in rows if row.get('config_hash') in expected]
 bad=[row for row in finished if not is_selectable_v2(row) or row.get('candidate_pass_saturated') or row.get('final_pass_saturated')]
 atomic(OUT/'candidate_iteration_high_limit_validation.json',{'validated_limit':int(limit),'requested':len(tasks),'completed':len(finished),'all_scientifically_selectable_and_unsaturated':not bad,'failed_hashes':[row.get('config_hash') for row in bad]})
 if bad or len(finished)!=len(tasks): raise RuntimeError('higher candidate-iteration limit validation did not complete cleanly')
 return rows
def saturation_audit(rows):
 broad=[row for row in rows if row.get('scientific_metrics_version')==2 and is_broad_stage(row) and row.get('status_category') not in ('worker_exception','coordinator/schema_error')]; iterations=np.asarray([row.get('candidate_pass_iterations') for row in broad if row.get('candidate_pass_iterations') is not None and row.get('candidate_pass_iterations')>=0],float); selectable=broad_selectable(rows); threshold=np.percentile([row['orientation_correct_cells_at_0p02deg'] for row in selectable],90) if selectable else np.inf; competitive=[row for row in selectable if row.get('orientation_correct_cells_at_0p02deg',-1)>=threshold]; approach=int(.9*tso.MAX_ITERATIONS); flagged=[row for row in competitive if row.get('candidate_pass_saturated') or row.get('candidate_pass_iterations',0)>=approach]
 report={'max_iterations':tso.MAX_ITERATIONS,'mask_voxels':int(np.prod(tso.WORK.labels.shape)) if tso.WORK is not None else 614400,'observed_count':int(iterations.size),'minimum':float(iterations.min()) if iterations.size else None,'median':float(np.median(iterations)) if iterations.size else None,'p90':float(np.percentile(iterations,90)) if iterations.size else None,'p95':float(np.percentile(iterations,95)) if iterations.size else None,'p99':float(np.percentile(iterations,99)) if iterations.size else None,'maximum':float(iterations.max()) if iterations.size else None,'observed_maximum_fraction_of_limit':float(iterations.max()/tso.MAX_ITERATIONS) if iterations.size else None,'approach_definition_fraction':.9,'approach_threshold_iterations':approach,'competitive_definition':'top decile of orientation-correct recovery at 0.02 degrees','competitive_count':len(competitive),'competitive_approaching_or_saturated':len(flagged),'flagged_base_config_hashes':[row['config_hash'] for row in flagged],'validated_higher_limit':1000000,'limit_rationale':'The fixed mask has 614400 voxels and every candidate iteration removes at least one remaining voxel; both 700000 and 1000000 exceed that finite exhaustion bound.'}; atomic(OUT/'candidate_iteration_distribution_audit.json',report); return flagged,report
def post_completion_seed_screen(rows,w,workers):
 candidates=[]; seen=set()
 for row in tso.rank([row for row in unique_rows(rows,is_selectable_v2) if is_selectable_v2(row)]):
  parameters={name:row[name] for name in oc.PARAMETER_NAMES}; key=oc.Config(**parameters).key()
  if key not in seen: seen.add(key); candidates.append(parameters)
  if len(candidates)==50: break
 screened=tso.run_tasks(candidates,'post_v2_five_seed_screen',range(5),workers,OUT/'evaluations.jsonl'); candidate_keys={tuple(p[name] for name in oc.PARAMETER_NAMES) for p in candidates}; relevant=[row for row in screened if is_selectable_v2(row) and tuple(row.get(name) for name in oc.PARAMETER_NAMES) in candidate_keys]; finals=[{name:row[name] for name in oc.PARAMETER_NAMES} for row in tso.rank_configurations(relevant)[:25]]; tso.run_tasks(finals,'post_v2_twenty_seed_final',range(20),workers,OUT/'evaluations.jsonl')
def complete(workers=8):
 lock=(OUT/'v2_broad_completion.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not drained(): raise RuntimeError('active primary queue has not drained; v2 completion must not run concurrently')
 tso.IDENTITY=tso.identity(); w=oc.load_workspace(tso.CACHE); tso.WORK=w; state_path=OUT/'v2_broad_completion_state.json'; state=json.loads(state_path.read_text()) if state_path.exists() else {'cursor':0,'batch':0}; rows=load_rows()
 while len(broad_selectable(rows))<BROAD_SELECTABLE_TARGET:
  shortfall=BROAD_SELECTABLE_TARGET-len(broad_selectable(rows)); requested=max(256,int(math.ceil(shortfall*1.15))); state['batch']+=1; parameters,state['cursor']=replacement_batch(w,rows,requested,state['cursor'],state['batch']); before=len(broad_selectable(rows)); rows=tso.run_tasks(parameters,f"broad_v2_replacement_{state['batch']}",[0],workers,OUT/'evaluations.jsonl'); after=len(broad_selectable(rows)); atomic(state_path,{**state,'selectable_before':before,'selectable_after':after,'requested_candidates':requested,'target':BROAD_SELECTABLE_TARGET});
  if after<=before: raise RuntimeError('deterministic v2 broad replacement batch made no selectable progress')
 flagged,audit=saturation_audit(rows)
 if flagged:
  atomic(OUT/'candidate_iteration_high_limit_required.json',{'status':'required_before_selection','configurations':[{'parameters':{name:row[name] for name in oc.PARAMETER_NAMES},'seed':row['candidate_order_seed'],'base_config_hash':row['config_hash']} for row in flagged],'validated_limit':audit['validated_higher_limit']})
  rows=validate_higher_limit(flagged,workers,audit['validated_higher_limit'])
 post_completion_seed_screen(rows,w,workers); rows=load_rows(); atomic(OUT/'v2_broad_completion_report.json',{**accounting(rows),'replacement_batches':state['batch'],'replacement_sampler':'independently scrambled Sobol','scramble_seed':SCRAMBLE_SEED,'completed':len(broad_selectable(rows))>=BROAD_SELECTABLE_TARGET,'schema_v1_rows_used_for_scientific_count':0})
 return rows
if __name__=='__main__': complete()

#!/usr/bin/env python3
"""Independent balanced Saltelli design and Sobol analysis (schema-v4).

The design has its own identities, JSONL store, lock, progress file and writer.
It never reads or writes the mixed primary ``evaluations.jsonl`` store.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,multiprocessing as mp,os,signal,sys,time
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[name]='1'
os.environ['MALLOC_ARENA_MAX']='2'
import numpy as np
import pandas as pd
from scipy.stats import qmc
HERE=Path(__file__).resolve().parent
# This is deliberately separate from every optimisation namespace.  No broad
# or adaptive rows are read by this script.
OUT=HERE/'two_stage_saltelli_v4_results'; sys.path.insert(0,str(HERE))
import oracle_core as oc,oracle_metrics as om,two_stage_oracle as tso
from two_stage_store import ResultStore,normalize_result,utc_now
STOP=False; IDENTITY=None
SALTELLI_SCHEMA_VERSION=4
SCIENTIFIC_METRICS_VERSION=3
SELECTION_POLICY_ID='object_orientation_count_geomean_v1'

def atomic(path,value):
 path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)); temporary.replace(path)
def transform(u,w):
 p={'local_threshold_deg':float(10**(-2.7+1.9*u[0])),'global_threshold_deg':float(-1 if u[1]<.25 else 10**(-2.1+2*u[1])),'footprint_tolerance':float(.04+.92*u[2]),'footprint_radius_um':float(10**(np.log10(.3)+u[3]*np.log10(4/.3))),'min_cell_size':int(5+245*u[4]),'kam_radius_um':float(10**(np.log10(.3)+u[5]*np.log10(2/.3)))}
 return tso.canonical(p,w)
def design_identity(record,parameters):
 payload={'experiment':'independent_balanced_saltelli_v2','result_schema_version':SALTELLI_SCHEMA_VERSION,'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,'selection_policy_id':SELECTION_POLICY_ID,'algorithm_id':tso.ALGORITHM,'python_source_sha256':IDENTITY['python_source_sha256'],'compiled_extension_sha256':IDENTITY['compiled_extension_sha256'],'latent_sha256':IDENTITY['latent_sha256'],'truth_sha256':IDENTITY['truth_sha256'],'mask_sha256':IDENTITY['mask_sha256'],'matching_metrics_version':SCIENTIFIC_METRICS_VERSION,'matching_source_sha256':IDENTITY['matching_source_sha256'],'saltelli_role':record['role'],'saltelli_index':int(record['index']),'saltelli_axis':record['axis'],'parameters':parameters,'candidate_order_seed':0,'max_iterations':tso.MAX_ITERATIONS,'stagnation_tolerance':tso.STAGNATION,'recycle_small_grains':False,'watershed_connectivity':tso.WATERSHED_CONNECTIVITY}
 digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(); return digest,payload
def init_worker():
 global IDENTITY; tso.init_worker(); IDENTITY=tso.identity()
def evaluate(task):
 row=tso.evaluate_task({'parameters':task['parameters'],'seed':0,'stage':'saltelli_independent','boundary':True}); base_hash=row['config_hash']; digest,payload=design_identity(task,task['parameters']); raw={**row,'base_configuration_hash':base_hash,'config_hash':digest,'configuration_identity':payload,'saltelli_role':task['role'],'saltelli_index':int(task['index']),'saltelli_axis':task['axis'],'stage':'saltelli_independent'}; return normalize_result(raw,payload,schema_version=SALTELLI_SCHEMA_VERSION)
def build_design(n,w):
 d=len(oc.PARAMETER_NAMES); m=int(np.log2(n)); base=qmc.Sobol(2*d,scramble=True,seed=20260821).random_base2(m); A,B=base[:,:d],base[:,d:]; records=[]
 def add(role,index,axis,u):
  p=transform(u,w); record={'role':role,'index':int(index),'axis':axis,'parameters':p}; digest,_=design_identity(record,p); record['design_hash']=digest; records.append(record)
 for j in range(n): add('A',j,None,A[j]); add('B',j,None,B[j])
 for axis in range(d):
  for j in range(n): x=A[j].copy(); x[axis]=B[j,axis]; add('AB',j,axis,x)
 return records
def run_design(design,workers):
 store=ResultStore(OUT/'saltelli_evaluations.jsonl',schema_version=SALTELLI_SCHEMA_VERSION); pending=[record for record in design if record['design_hash'] not in store.hashes]; initial=len(store.rows); started=time.monotonic(); available,total,swap0=tso.memory(); context=mp.get_context('spawn')
 with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=init_worker,max_tasks_per_child=15) as pool:
  futures={}
  while pending or futures:
   available,total,swap=tso.memory(); aggregate,maximum=tso.tree_rss(); growth=max(0,swap-swap0); unsafe=aggregate>=10*1024**3 or available<max(6*1024**3,.25*total) or growth>256*1024**2
   if unsafe or aggregate>=12*1024**3: globals()['STOP']=True
   while pending and len(futures)<workers and not STOP:
    task=pending.pop(0); futures[pool.submit(evaluate,task)]=task
   if not futures: break
   ready,_=wait(futures,timeout=2,return_when=FIRST_COMPLETED)
   for future in ready:
    task=futures.pop(future)
    try: row=future.result()
    except Exception as exc:
     digest,payload=design_identity(task,task['parameters']); row=normalize_result({'config_hash':digest,'configuration_identity':payload,'algorithm':tso.ALGORITHM,'candidate_order_seed':0,'status':'error','error_type':type(exc).__name__,'error_message':str(exc),'worker_pid':None,'worker_peak_rss_bytes':None,'started':None,'finished':utc_now(),'elapsed_seconds':None,'candidate_pass_saturated':False,'final_pass_saturated':False},payload,schema_version=SALTELLI_SCHEMA_VERSION)
    store.append(row,row['configuration_identity'])
    elapsed=max(time.monotonic()-started,1e-9); completed=len(store.rows)-initial; atomic(OUT/'saltelli_progress.json',{'result_schema_version':SALTELLI_SCHEMA_VERSION,'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,'selection_policy_id':SELECTION_POLICY_ID,'durable_rows':len(store.rows),'completed_this_run':completed,'planned_matrix_rows':len(design),'pending':len(pending),'in_flight':len(futures),'status_counts':store.counts(),'trials_per_second':completed/elapsed,'aggregate_rss_bytes':aggregate,'maximum_process_rss_bytes':maximum,'available_ram_bytes':available,'swap_baseline_bytes':swap0,'swap_growth_bytes':growth,'safety_stop':STOP,'updated':utc_now()})
 return list(store.rows)
def analyse(design,rows,w,n):
 lookup={row['config_hash']:row for row in rows}; truth_entropy=om.variation_of_information(*om.contingency(w.labels,np.ones_like(w.labels)))[0]; n_true=int(w.labels.max()); fallback={'orientation_correct_recovered_fraction_at_0p02deg':0.,'unmatched_true_cells':n_true,'split_true_cells':0.,'merged_predicted_cells':0.,'split_merge_error_total':0.,'absolute_cell_count_error':n_true,'median_matched_mean_orientation_error_deg':float(np.sqrt(2)*180.),'vi_total_bits':truth_entropy,'ari':0.,'boundary_f1_at_0p4um':0.}
 # Expected algorithmic invalids are observations with a deterministic
 # scientific fallback.  Infrastructure/schema failures remain excluded.
 # In particular, None is never allowed to reach a NumPy estimator.
 fallback.update(split_true_cells=float(n_true), merged_predicted_cells=float(n_true),
                 split_merge_error_total=float(2*n_true))
 def finite_observed(value):
  try:
   value=float(value)
  except (TypeError,ValueError):
   return np.nan
  return value if np.isfinite(value) else np.nan
 def value(record,response):
  row=lookup.get(record['design_hash'])
  if row is None or row.get('status_category') in ('worker_exception','coordinator/schema_error','candidate_saturated','final_saturated'): return np.nan
  if row.get('status_category')=='expected_algorithmic_invalid': return fallback[response]
  if response=='split_merge_error_total': return finite_observed(row.get('split_true_cells')) + finite_observed(row.get('merged_predicted_cells'))
  return finite_observed(row.get(response))
 output=[]
 for response in fallback:
  YA=np.asarray([value(row,response) for row in design if row['role']=='A']); YB=np.asarray([value(row,response) for row in design if row['role']=='B']); variance=np.nanvar(np.r_[YA,YB],ddof=1)
  for axis,parameter in enumerate(oc.PARAMETER_NAMES):
   YAB=np.asarray([value(row,response) for row in design if row['role']=='AB' and row['axis']==axis]); valid=np.isfinite(YA)&np.isfinite(YB)&np.isfinite(YAB); s1=np.mean(YB[valid]*(YAB[valid]-YA[valid]))/variance if variance and valid.any() else np.nan; st=.5*np.mean((YA[valid]-YAB[valid])**2)/variance if variance and valid.any() else np.nan; rng=np.random.default_rng(20260830+axis); boots=[]
   if variance and valid.sum()>=32:
    av,bv,cv=YA[valid],YB[valid],YAB[valid]
    for _ in range(200): take=rng.integers(0,len(av),len(av)); boots.append((np.mean(bv[take]*(cv[take]-av[take]))/variance,.5*np.mean((av[take]-cv[take])**2)/variance))
   boot=np.asarray(boots) if boots else np.empty((0,2)); output.append({'response':response,'parameter':parameter,'S1':s1,'S1_ci95_low':float(np.percentile(boot[:,0],2.5)) if len(boot) else np.nan,'S1_ci95_high':float(np.percentile(boot[:,0],97.5)) if len(boot) else np.nan,'ST':st,'ST_ci95_low':float(np.percentile(boot[:,1],2.5)) if len(boot) else np.nan,'ST_ci95_high':float(np.percentile(boot[:,1],97.5)) if len(boot) else np.nan,'valid_N':int(valid.sum()),'adequate':bool(valid.sum()>=.9*n)})
 safe_output=[]
 for row in output:
  safe_output.append({key:(None if isinstance(value,(float,np.floating)) and not np.isfinite(float(value)) else value) for key,value in row.items()})
 pd.DataFrame(safe_output).to_csv(OUT/'sobol_indices.csv',index=False,na_rep=''); atomic(OUT/'sobol_indices.json',{'result_schema_version':SALTELLI_SCHEMA_VERSION,'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,'selection_policy_id':SELECTION_POLICY_ID,'rows':safe_output,'source_store':'saltelli_evaluations.jsonl only','mixed_primary_store_consumed':False}); atomic(OUT/'saltelli_response_policy.json',{'result_schema_version':SALTELLI_SCHEMA_VERSION,'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,'selection_policy_id':SELECTION_POLICY_ID,'source_store':'saltelli_evaluations.jsonl only','mixed_primary_store_consumed':False,'responses':list(fallback),'expected_algorithmic_invalid_policy':fallback,'infrastructure_failure_policy':'retained in the matrix and reported separately; excluded from estimator masks, never imputed','bootstrap_replicates':200})
def main():
 global IDENTITY
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--n',type=int,default=512); parser.add_argument('--workers',type=int,default=8); args=parser.parse_args()
 if args.n<512 or args.n&(args.n-1): raise SystemExit('N must be a power of two and >=512')
 lock=(OUT/'saltelli.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True)); signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('STOP',True)); IDENTITY=tso.identity(); tso.IDENTITY=IDENTITY; w=oc.load_workspace(tso.CACHE); design=build_design(args.n,w); atomic(OUT/'saltelli_manifest.json',{'manifest_version':1,'result_schema_version':SALTELLI_SCHEMA_VERSION,'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,'selection_policy_id':SELECTION_POLICY_ID,'algorithm_id':tso.ALGORITHM,'python_source_sha256':IDENTITY['python_source_sha256'],'compiled_extension_sha256':IDENTITY['compiled_extension_sha256'],'latent_sha256':IDENTITY['latent_sha256'],'truth_sha256':IDENTITY['truth_sha256'],'mask_sha256':IDENTITY['mask_sha256'],'matching_source_sha256':IDENTITY['matching_source_sha256'],'max_iterations':tso.MAX_ITERATIONS,'parameters':list(oc.PARAMETER_NAMES),'bounds':{'local_threshold_deg':[10**-2.7,10**-0.8],'global_threshold_deg':['disabled',10**-2.1,10**-0.1],'footprint_tolerance':[.04,.96],'footprint_radius_um':[.3,4.0],'min_cell_size':[5,250],'kam_radius_um':[.3,2.0]},'N':args.n,'D':len(oc.PARAMETER_NAMES),'second_order':False,'matrix_rows':len(design),'evaluation_formula':'N*(2+D)','sampler':'scrambled Sobol over 2D base coordinates; A, B and A_Bi hybrids','scramble_seed':20260821,'source_store':'saltelli_evaluations.jsonl only','mixed_primary_store_consumed':False}); atomic(OUT/'saltelli_design.json',{'N':args.n,'D':len(oc.PARAMETER_NAMES),'parameters':list(oc.PARAMETER_NAMES),'sampler':'independent scrambled 2D-dimensional Sobol base matrix split into A/B, with balanced A_B^i Saltelli hybrids','scramble_seed':20260821,'matrix_rows':len(design),'source_store':'saltelli_evaluations.jsonl','mixed_primary_store_consumed':False,'rows':design}); rows=run_design(design,args.workers); analyse(design,rows,w,args.n)
if __name__=='__main__': main()

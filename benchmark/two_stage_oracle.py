#!/usr/bin/env python3
"""Definitive size-prioritised two-stage oracle search."""
from __future__ import annotations
import argparse,collections,datetime as dt,fcntl,hashlib,json,math,multiprocessing as mp,os,re,signal,subprocess,sys,time,traceback
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
from pathlib import Path

for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[n]="1"
os.environ["MALLOC_ARENA_MAX"]="2"; os.environ.setdefault("MPLCONFIGDIR","/tmp/disell-two-stage-mpl")
import numpy as np
HERE=Path(__file__).resolve().parent
LEGACY_OUT=HERE/"two_stage_oracle_results"
FROZEN_BROAD_OUT=HERE/"two_stage_oracle_f1_results"
POLICY_OUT=HERE/"two_stage_oracle_count_results"
OUT=Path(os.environ.get("DISELL_TWO_STAGE_OUT",str(POLICY_OUT)))
CACHE=HERE/"oracle_results"/"cache"
sys.path.insert(0,str(HERE))
import object_orientation_metrics as oom,oracle_core as oc,oracle_metrics as om,pipelines
from two_stage_store import (REQUIRED_ENVELOPE, VALID_CATEGORIES,
                             ResultStore, minimal_schema_error,
                             normalize_result, utc_now)

ALGORITHM="size_prioritised_multiseed_v1"
SELECTION_POLICY_ID="object_orientation_count_geomean_v1"
SELECTION_POLICY_VERSION=1
SCIENTIFIC_METRICS_VERSION=3
SOURCE_RESULT_SCHEMA_VERSION=3
RESULT_SCHEMA_VERSION=4
SOURCE_SELECTION_POLICY_ID="object_orientation_f1_v1"
ORIENTATION_TAGS=('0p005','0p01','0p02','0p05')
EXPECTED_BROAD_RECORDS=8000
EXPECTED_BROAD_SELECTABLE=6930
EXPECTED_BROAD_INVALID=1070
EXPECTED_LEADER_HASH="75d29964871ba6769939e6b5b5e13dfa4ce4e8503b70787758b229ced828c77c"
EXPECTED_DIVERSE_COUNTS=(340,351,355,352,366,367,371,320,360)
ADAPTIVE_BATCH_SIZE=2000
MAX_ITERATIONS=700000; STAGNATION=2000; WATERSHED_CONNECTIVITY=1
WORK=None; IDENTITY=None; STOP=False

def plain(x):
 if isinstance(x,np.generic): return x.item()
 if isinstance(x,float) and not np.isfinite(x): return None
 if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [plain(v) for v in x]
 return x
def atomic(path,value):
 path=Path(path); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(plain(value),indent=2,sort_keys=True,allow_nan=False)); tmp.replace(path)
def sha256_bytes(raw): return hashlib.sha256(raw).hexdigest()
def sha256_json(value):
 text=json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False)
 return hashlib.sha256(text.encode()).hexdigest()
def nonfinite_paths(value,path='result'):
 if isinstance(value,dict):
  for key,child in value.items(): yield from nonfinite_paths(child,f'{path}.{key}')
 elif isinstance(value,(list,tuple)):
  for index,child in enumerate(value): yield from nonfinite_paths(child,f'{path}[{index}]')
 elif isinstance(value,float) and not math.isfinite(value): yield path
def strict_jsonl(path):
 """Read strict, newline-terminated JSONL without normalising or rewriting it."""
 path=Path(path); raw=path.read_bytes(); rows=[]
 if raw and not raw.endswith(b'\n'): raise RuntimeError(f'partial final JSONL line in {path}')
 for line_number,line in enumerate(raw.splitlines(keepends=True),1):
  if not line.endswith(b'\n'): raise RuntimeError(f'partial JSONL line {line_number} in {path}')
  try: row=json.loads(line,parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'non-standard JSON constant {value}')))
  except Exception as exc: raise RuntimeError(f'malformed JSONL line {line_number} in {path}: {exc}') from exc
  bad=list(nonfinite_paths(row))
  if bad: raise RuntimeError(f'non-finite values at line {line_number} in {path}: {bad[:8]}')
  rows.append(row)
 return raw,rows
def _source_selectable(row):
 if row.get('stage')!='broad' or row.get('status_category')!='ok': return False
 if row.get('schema_version')!=SOURCE_RESULT_SCHEMA_VERSION: return False
 if row.get('scientific_metrics_version')!=SCIENTIFIC_METRICS_VERSION: return False
 if row.get('selection_policy_id')!=SOURCE_SELECTION_POLICY_ID: return False
 required=('n_cells_true','n_cells_pred','identity_f1','panoptic_quality_at_0p5','matched_iou_mean','vi_total_bits','ari')
 required+=tuple(f'orientation_correct_f1_at_{tag}deg' for tag in ORIENTATION_TAGS)
 try: values=[float(row[key]) for key in required]
 except (KeyError,TypeError,ValueError): return False
 return all(math.isfinite(value) for value in values)
def policy_view(row,source_sha256):
 """Derive a new-policy in-memory view while retaining original provenance."""
 view=dict(row); view.update(selection_metrics(row,include_provenance=False))
 view['_selection_policy_view_id']=SELECTION_POLICY_ID
 view['_evidence_origin']='external_frozen_broad'
 view['_frozen_broad_source_sha256']=source_sha256
 return view
def audit_frozen_broad_source():
 """Strictly audit and import broad evidence only; never write to its store."""
 path=FROZEN_BROAD_OUT/'evaluations.jsonl'; before=path.stat(); raw,rows=strict_jsonl(path)
 hashes=[row.get('config_hash') for row in rows]
 if any(row.get('schema_version')!=SOURCE_RESULT_SCHEMA_VERSION for row in rows):
  raise RuntimeError('frozen source contains a non-schema-v3 row')
 schema=[index for index,row in enumerate(rows,1) if any(key not in row for key in REQUIRED_ENVELOPE) or row.get('status_category') not in VALID_CATEGORIES]
 if schema: raise RuntimeError(f'frozen source schema errors at lines {schema[:8]}')
 if len(hashes)!=len(set(hashes)): raise RuntimeError('frozen source contains duplicate full hashes')
 broad=[row for row in rows if row.get('stage')=='broad']
 ok=[row for row in broad if _source_selectable(row)]
 invalid=[row for row in broad if row.get('status_category')=='expected_algorithmic_invalid']
 other=[row for row in broad if not _source_selectable(row) and row.get('status_category')!='expected_algorithmic_invalid']
 expected=(EXPECTED_BROAD_RECORDS,EXPECTED_BROAD_SELECTABLE,EXPECTED_BROAD_INVALID)
 actual=(len(broad),len(ok),len(invalid))
 if actual!=expected or other:
  raise RuntimeError(f'frozen broad accounting mismatch: expected={expected}, actual={actual}, other={len(other)}')
 source_sha256=sha256_bytes(raw); evidence=[policy_view(row,source_sha256) for row in ok]
 ranked=rank(evidence); leader=ranked[0]
 expected_fields={'config_hash':EXPECTED_LEADER_HASH,'n_cells_pred':358,'n_cells_true':360,
  'orientation_correct_cells_at_0p02deg':280,'one_to_one_recovered_cells':295}
 mismatch={key:(leader.get(key),value) for key,value in expected_fields.items() if leader.get(key)!=value}
 numeric={'count_score':.9944444444444445,'orientation_correct_f1_at_0p02deg':.7799442896935933,
  'orientation_count_score_at_0p02deg':.8806879497595554,'identity_f1':.8217270194986073,
  'matched_iou_mean':.8290050747011372,'boundary_assd_um':.05445188129369507,
  'ari':.8698120612943164}
 mismatch.update({key:(leader.get(key),value) for key,value in numeric.items() if not math.isclose(float(leader.get(key,-1)),value,rel_tol=1e-8,abs_tol=1e-10)})
 if mismatch: raise RuntimeError(f'frozen broad revised leader mismatch: {mismatch}')
 top_counts={int(row['n_cells_pred']) for row in ranked[:25]}
 missing_counts=[value for value in EXPECTED_DIVERSE_COUNTS if value not in top_counts]
 if missing_counts: raise RuntimeError(f'frozen broad top group lacks count diversity: {missing_counts}')
 after=path.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or sha256_bytes(path.read_bytes())!=source_sha256:
  raise RuntimeError('frozen broad source changed during read-only audit')
 audit={'source_jsonl':str(path.resolve()),'source_jsonl_sha256':source_sha256,
  'source_jsonl_bytes':len(raw),'source_total_records':len(rows),
  'broad_records':len(broad),'selectable_broad_records':len(ok),
  'expected_algorithmic_invalid_broad_records':len(invalid),
  'excluded_non_broad_records':len(rows)-len(broad),
  'excluded_stage_counts':dict(collections.Counter(str(row.get('stage')) for row in rows if row.get('stage')!='broad')),
  'duplicate_full_hashes':0,'schema_errors':0,'worker_exceptions':sum(row.get('status_category')=='worker_exception' for row in broad),
  'nonfinite_records':0,'source_selection_policy_id':SOURCE_SELECTION_POLICY_ID,
  'source_result_schema_version':SOURCE_RESULT_SCHEMA_VERSION}
 return audit,evidence
def initialise_policy_namespace(audit=None):
 """Create/validate an isolated schema-v4 continuation manifest."""
 if OUT.resolve() in {LEGACY_OUT.resolve(),FROZEN_BROAD_OUT.resolve()}:
  raise RuntimeError(f'{SELECTION_POLICY_ID} must use a clean isolated result namespace')
 if audit is None: audit,_=audit_frozen_broad_source()
 source=json.loads((FROZEN_BROAD_OUT/'implementation_audit.json').read_text())
 pinned={key:source[key] for key in ('python_source_sha256','compiled_extension_sha256','latent_sha256','truth_sha256','mask_sha256','matching_source_sha256')}
 runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 manifest={**source,**pinned,'manifest_version':1,'algorithm_id':ALGORITHM,
  'matching_metrics_version':SCIENTIFIC_METRICS_VERSION,
  'selection_policy_id':SELECTION_POLICY_ID,'selection_policy_version':SELECTION_POLICY_VERSION,
  'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,
  'result_schema_version':RESULT_SCHEMA_VERSION,'runner_source_sha256':runner_hash,
  'frozen_broad_source':audit['source_jsonl'],'frozen_broad_source_sha256':audit['source_jsonl_sha256'],
  'frozen_broad_source_bytes':audit['source_jsonl_bytes'],'frozen_broad_source_total_records':audit['source_total_records'],
  'frozen_broad_record_count':audit['broad_records'],'frozen_broad_selectable_count':audit['selectable_broad_records'],
  'frozen_broad_expected_invalid_count':audit['expected_algorithmic_invalid_broad_records'],
  'source_result_schema_version':SOURCE_RESULT_SCHEMA_VERSION,'source_selection_policy_id':SOURCE_SELECTION_POLICY_ID,
  'broad_evidence_mode':'external_frozen_read_only','broad_rows_copied':False,
  'old_policy_adaptive_rows_imported':False,'old_policy_adaptive_rows_excluded':audit['excluded_non_broad_records'],
  'count_score_formula':'0.0 if either count is zero else min(n_cells_true,n_cells_pred)/max(n_cells_true,n_cells_pred)',
  'orientation_count_score_formula':'sqrt(orientation_correct_f1_at_{tag}deg * count_score)',
  'created':utc_now()}
 OUT.mkdir(parents=True,exist_ok=True); path=OUT/'implementation_audit.json'
 required_keys=(*pinned,'manifest_version','algorithm_id','matching_metrics_version','selection_policy_id','selection_policy_version','scientific_metrics_version','result_schema_version','runner_source_sha256','frozen_broad_source','frozen_broad_source_sha256','frozen_broad_source_bytes','frozen_broad_source_total_records','frozen_broad_record_count','frozen_broad_selectable_count','frozen_broad_expected_invalid_count','source_result_schema_version','source_selection_policy_id','broad_evidence_mode','broad_rows_copied','old_policy_adaptive_rows_imported','old_policy_adaptive_rows_excluded','count_score_formula','orientation_count_score_formula')
 if path.exists():
  current=json.loads(path.read_text()); mismatch={key:(current.get(key),manifest[key]) for key in required_keys if current.get(key)!=manifest[key]}
  if mismatch: raise RuntimeError(f'policy namespace manifest mismatch: {mismatch}')
  manifest=current
 else: atomic(path,manifest)
 return manifest
def update_resource_history(aggregate,max_process,available,swap_growth,worker_peak):
 path=OUT/'resource_history.json'
 previous=json.loads(path.read_text()) if path.exists() else {}
 value={'peak_aggregate_process_tree_rss_bytes':max(int(aggregate),int(previous.get('peak_aggregate_process_tree_rss_bytes',0))),'peak_individual_process_rss_bytes':max(int(max_process),int(previous.get('peak_individual_process_rss_bytes',0))),'peak_worker_reported_rss_bytes':max(int(worker_peak),int(previous.get('peak_worker_reported_rss_bytes',0))),'minimum_available_ram_bytes':min(int(available),int(previous.get('minimum_available_ram_bytes',available))),'maximum_swap_growth_bytes':max(int(swap_growth),int(previous.get('maximum_swap_growth_bytes',0))),'updated':utc_now()}
 atomic(path,value)
def append(path,row):
 with Path(path).open("a") as f: f.write(json.dumps(plain(row),sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())
def load(path):
 if not Path(path).exists(): return []
 return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def memory():
 d={}
 for line in Path('/proc/meminfo').read_text().splitlines(): k,v=line.split(':',1); d[k]=int(v.strip().split()[0])*1024
 return d['MemAvailable'],d['MemTotal'],d['SwapTotal']-d['SwapFree']
def tree_rss():
 import psutil
 root=psutil.Process(os.getpid()); values=[]
 for p in [root]+root.children(recursive=True):
  try: values.append(int(p.memory_info().rss))
  except (psutil.NoSuchProcess,psutil.AccessDenied): pass
 return sum(values),max(values,default=0)
def identity(): return json.loads((OUT/"implementation_audit.json").read_text())
def config_key(p,seed):
 payload={"algorithm":ALGORITHM,"python_source_sha256":IDENTITY["python_source_sha256"],"compiled_extension_sha256":IDENTITY["compiled_extension_sha256"],"latent_sha256":IDENTITY["latent_sha256"],"truth_sha256":IDENTITY["truth_sha256"],"mask_sha256":IDENTITY["mask_sha256"],"matching_metrics_version":SCIENTIFIC_METRICS_VERSION,"matching_source_sha256":IDENTITY["matching_source_sha256"],"selection_policy_id":SELECTION_POLICY_ID,"parameters":p,"candidate_order_seed":int(seed),"max_iterations":MAX_ITERATIONS,"stagnation_tolerance":STAGNATION,"recycle_small_grains":False,"watershed_connectivity":WATERSHED_CONNECTIVITY}
 text=json.dumps(payload,sort_keys=True,separators=(',',':')); return hashlib.sha256(text.encode()).hexdigest(),payload
def init_worker():
 global WORK,IDENTITY; IDENTITY=identity(); WORK=oc.load_workspace(CACHE)
def sizes_summary(values,prefix):
 a=np.asarray(values,float)
 return {f"{prefix}_count":int(a.size),f"{prefix}_min":float(a.min()) if a.size else None,f"{prefix}_median":float(np.median(a)) if a.size else None,f"{prefix}_p90":float(np.percentile(a,90)) if a.size else None,f"{prefix}_max":float(a.max()) if a.size else None}
def safe_ratio(numerator,denominator):
 """Return a finite probability, with an explicit empty-denominator policy."""
 value=float(numerator/denominator) if denominator>0 else 0.0
 if not math.isfinite(value) or not 0.0<=value<=1.0:
  raise ValueError(f'bounded ratio outside [0, 1]: {numerator}/{denominator}={value}')
 return value
def harmonic_f1(precision,recall):
 if precision is None or recall is None: return None
 if not math.isfinite(float(precision)) or not math.isfinite(float(recall)): return None
 if not 0.0<=float(precision)<=1.0 or not 0.0<=float(recall)<=1.0:
  raise ValueError('interface precision and recall must be within [0, 1]')
 denominator=float(precision)+float(recall)
 return float(2*precision*recall/denominator) if denominator>0 else 0.0
def bounded_score(value,name):
 value=float(value)
 if not math.isfinite(value) or not 0.0<=value<=1.0:
  raise ValueError(f'{name} must be finite and within [0, 1], got {value}')
 return value
def count_score(n_true,n_pred):
 """Symmetric cell-count agreement; an empty side has score zero."""
 n_true=int(n_true or 0); n_pred=int(n_pred or 0)
 if n_true<0 or n_pred<0: raise ValueError('cell counts must be non-negative')
 if n_true==0 or n_pred==0: return 0.0
 return bounded_score(min(n_true,n_pred)/max(n_true,n_pred),'count_score')
def selection_metrics(row,*,include_provenance=True):
 """Finite object, symmetric-count, and orientation-count policy metrics."""
 n_true=int(row.get('n_cells_true',row.get('n_true_cells',0)) or 0); n_pred=int(row.get('n_cells_pred',row.get('n_predicted_cells',0)) or 0)
 output={}
 score=count_score(n_true,n_pred); output['count_score']=score
 for tag in ORIENTATION_TAGS:
  f1_key=f'orientation_correct_f1_at_{tag}deg'
  if f1_key in row:
   f1=bounded_score(row[f1_key],f1_key)
   correct=int(row.get(f'orientation_correct_cells_at_{tag}deg',0) or 0)
   precision=safe_ratio(correct,n_pred); recall=safe_ratio(correct,n_true)
  else:
   correct=int(row.get(f'orientation_correct_cells_at_{tag}deg',0) or 0)
   precision=safe_ratio(correct,n_pred); recall=safe_ratio(correct,n_true); f1=safe_ratio(2*correct,n_true+n_pred)
  output[f'orientation_correct_precision_at_{tag}deg']=precision; output[f'orientation_correct_recall_at_{tag}deg']=recall; output[f'orientation_correct_f1_at_{tag}deg']=f1
  output[f'orientation_count_score_at_{tag}deg']=bounded_score(math.sqrt(f1*score),f'orientation_count_score_at_{tag}deg')
 recovered=int(row.get('one_to_one_recovered_cells',0) or 0); output['identity_precision']=safe_ratio(recovered,n_pred); output['identity_recall']=safe_ratio(recovered,n_true); output['identity_f1']=safe_ratio(2*recovered,n_true+n_pred)
 recall=row.get('true_facet_recall_area_weighted'); precision=row.get('interface_precision')
 output['interface_f1']=harmonic_f1(precision,recall)
 for key,value in output.items():
  if value is not None and (key=='count_score' or key.startswith('orientation_count_score_at_')): bounded_score(value,key)
 if include_provenance:
  output['selection_policy_id']=SELECTION_POLICY_ID; output['scientific_metrics_version']=SCIENTIFIC_METRICS_VERSION
 return output
def _evaluate_task(task):
 import resource
 p=task['parameters']; cfg=oc.Config(**p); key,payload=config_key(p,task['seed']); t=time.perf_counter()
 fp=WORK.footprint(cfg.footprint_radius_um); kam=WORK.kam(cfg.kam_radius_um)
 labels,markers,initial,diag,final_sizes=pipelines.run_flood_fill_two_stage(WORK.field,WORK.mask,kam,fp,local_threshold_deg=cfg.local_threshold_deg,global_threshold_deg=cfg.global_threshold_deg,footprint_tolerance=cfg.footprint_tolerance,min_cell_size=cfg.min_cell_size,max_seed_attempts=MAX_ITERATIONS,stagnation_tolerance=STAGNATION,random_seed=task['seed'],watershed_connectivity=WATERSHED_CONNECTIVITY,recycle_small_grains=False)
 cand=plain(diag.get('candidate_collection') or {}); final=plain(diag.get('final_growth') or {})
 if int(markers.max()) == 0:
  raw={"algorithm":ALGORITHM,"config_hash":key,"configuration_identity":payload,
    "config_key":cfg.key(),"candidate_order_seed":int(task['seed']),"stage":task['stage'],
    "n_cells_true":int(WORK.labels.max()),"n_cells_pred":0,
    **p,**sizes_summary(initial,'preliminary_candidate_size'),
    "preliminary_candidates_detected":int(len(initial)),"accepted_marker_count":0,
    "candidate_pass_iterations":int(cand.get('iterations',-1)),
    "final_pass_iterations":int(final.get('iterations',-1)) if final else None,
    "candidate_pass_saturated":bool(cand.get('max_iterations_reached',False)),
    "final_pass_saturated":bool(final.get('max_iterations_reached',False)) if final else False,
    "status":"invalid","error_type":"NoAcceptedMarkers",
    "error_message":"No valid seeds or no accepted markers"}
  raw.update(selection_metrics(raw)); return raw
 _,obj=oom.match_cells(WORK.labels,labels,WORK.field,WORK.spacing_um_zyx,purity_threshold=.6,completeness_threshold=.6)
 metrics=om.evaluate_partition(WORK.labels,labels,WORK.spacing_um_zyx,with_boundary=task.get('boundary',False)); cand=plain(diag['candidate_collection']); final=plain(diag['final_growth'])
 saturated=bool(cand.get('max_iterations_reached') or final.get('max_iterations_reached'))
 req=oc.neighbour_requirements(cfg.footprint_tolerance,int(fp.sum()))
 row={"algorithm":ALGORITHM,"config_hash":key,"configuration_identity":payload,"config_key":cfg.key(),"candidate_order_seed":int(task['seed']),"stage":task['stage'],**p,**metrics,**obj,**sizes_summary(initial,'preliminary_candidate_size'),**sizes_summary(final_sizes,'marker_size'),"preliminary_candidates_detected":int(len(initial)),"candidate_seeds_supplied":int(final.get('user_seeds_supplied',0)),"candidate_seeds_skipped_claimed":int(final.get('user_seeds_skipped_claimed',0)),"candidate_seeds_processed":int(final.get('user_seeds_processed',0)),"accepted_marker_count":int(markers.max()),"unlabelled_marker_stage_fraction":float(np.mean(markers==0)),"final_watershed_count":int(labels.max()),"candidate_pass_iterations":int(cand.get('iterations',-1)),"final_pass_iterations":int(final.get('iterations',-1)),"candidate_pass_saturated":bool(cand.get('max_iterations_reached',True)),"final_pass_saturated":bool(final.get('max_iterations_reached',True)),"limit_saturated":saturated,"effective_neighbour_requirement":int(req[-1]),"effective_neighbour_vector_sha256":hashlib.sha256(req.tobytes()).hexdigest(),"footprint_voxels":int(fp.sum()),"kam_footprint_voxels":int(WORK.footprint(cfg.kam_radius_um).sum()),"runtime_seconds":time.perf_counter()-t,"worker_peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),"status":"incomplete_limit_saturated" if saturated else "ok"}
 row.update(selection_metrics(row))
 return row
def evaluate_task(task):
 import resource
 started=utc_now(); clock=time.perf_counter(); pid=os.getpid(); raw={}
 p=task['parameters']; key,payload=config_key(p,task['seed'])
 try:
  raw=_evaluate_task(task)
 except Exception as exc:
  raw={"algorithm":ALGORITHM,"config_hash":key,"configuration_identity":payload,
    "config_key":oc.Config(**p).key(),"candidate_order_seed":int(task['seed']),
    "stage":task['stage'],**p,"status":"error","error_type":type(exc).__name__,
    "error_message":str(exc)}
 finally:
  raw.setdefault("config_hash",key); raw.setdefault("configuration_identity",payload)
  raw.update({"worker_pid":pid,"worker_peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
    "started":started,"finished":utc_now(),"elapsed_seconds":time.perf_counter()-clock,
    "scientific_metrics_version":SCIENTIFIC_METRICS_VERSION,"selection_policy_id":SELECTION_POLICY_ID})
 return normalize_result(raw,payload,schema_version=RESULT_SCHEMA_VERSION)
def worker_future_failure(task,exc):
 """Envelope an executor-level failure that occurred outside worker code."""
 p=task['parameters']; key,payload=config_key(p,task['seed'])
 return normalize_result({"algorithm":ALGORITHM,"config_hash":key,
   "configuration_identity":payload,"config_key":oc.Config(**p).key(),
   "candidate_order_seed":int(task['seed']),"stage":task['stage'],**p,
   "status":"error","error_type":type(exc).__name__,
   "error_message":f"worker future failed before returning an envelope: {exc}",
   "worker_pid":None,"worker_peak_rss_bytes":None,"started":None,
   "finished":utc_now(),"elapsed_seconds":None,
   "candidate_pass_saturated":False,"final_pass_saturated":False,
   "scientific_metrics_version":SCIENTIFIC_METRICS_VERSION,"selection_policy_id":SELECTION_POLICY_ID},payload,schema_version=RESULT_SCHEMA_VERSION)
def prepare_worker_result(raw,task,expected,identity_payload):
 """Validate an untrusted worker return without spreading it into errors."""
 try: row=normalize_result(raw,identity_payload,schema_version=RESULT_SCHEMA_VERSION)
 except Exception as exc:
  return minimal_schema_error(raw,identity_payload,error_type=type(exc).__name__,error_message=f'coordinator normalisation failed: {exc}',config_hash=expected,schema_version=RESULT_SCHEMA_VERSION)
 if row.get('config_hash')!=expected:
  return minimal_schema_error(raw,identity_payload,error_type='SchemaError',error_message='configuration identity mismatch',config_hash=expected,schema_version=RESULT_SCHEMA_VERSION)
 if task.get('sampling'): row['sampling_metadata']=task['sampling']
 return row
def canonical(p,w): return oc.canonical(oc.Config(**p),w).as_dict()
def broad(n,w):
 from scipy.stats import qmc
 x=qmc.Sobol(6,scramble=True,seed=20260815).random_base2(int(np.ceil(np.log2(n*2)))); out=[]; seen=set(); skipped=[]
 for sample_index,v in enumerate(x):
  p={"local_threshold_deg":float(10**(-2.7+1.9*v[0])),"global_threshold_deg":float(-1 if v[1]<.3 else 10**(-2.1+2*v[1])),"footprint_tolerance":float(.04+.92*v[2]),"footprint_radius_um":float(10**(np.log10(.3)+v[3]*np.log10(4/.3))),"min_cell_size":int(5+245*v[4]),"kam_radius_um":float(10**(np.log10(.3)+v[5]*np.log10(2/.3)))}; out.append(canonical(p,w))
  p=out.pop(); fp=w.footprint(p['footprint_radius_um']); req=oc.neighbour_requirements(p['footprint_tolerance'],int(fp.sum()))
  signature=(p['local_threshold_deg'],p['global_threshold_deg'],hashlib.sha256(req.tobytes()).hexdigest(),p['footprint_radius_um'],p['min_cell_size'],p['kam_radius_um'])
  if signature in seen:
   skipped.append({'sample_index':sample_index,'reason':'identical canonical parameters/effective neighbour vector'}); continue
  seen.add(signature); p['_sampling']={'broad_batch_id':'broad_batch_1' if sample_index<3000 else 'broad_batch_2_extended','sampler_type':'Sobol','scramble_seed':20260815,'original_sample_index':sample_index,'deduplication_reason':None}; out.append(p)
  if len(out)>=n: break
 if len(out)<n: raise RuntimeError(f'only {len(out)} unique broad configurations generated')
 atomic(OUT/'broad_deduplication.json',{'selection_policy_id':SELECTION_POLICY_ID,'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,'requested_unique':n,'generated_unique':len(out),'skipped':skipped})
 return out
def selectable(rows):
 required=('orientation_count_score_at_0p02deg','count_score','identity_f1','orientation_count_score_at_0p01deg','panoptic_quality_at_0p5','matched_iou_mean','vi_total_bits','ari')
 def valid(row):
  policy_ok=row.get('selection_policy_id')==SELECTION_POLICY_ID
  frozen_ok=(row.get('_selection_policy_view_id')==SELECTION_POLICY_ID and row.get('_evidence_origin')=='external_frozen_broad' and row.get('stage')=='broad' and row.get('schema_version')==SOURCE_RESULT_SCHEMA_VERSION)
  if row.get('status_category',row.get('status'))!='ok' or row.get('scientific_metrics_version')!=SCIENTIFIC_METRICS_VERSION or not (policy_ok or frozen_ok): return False
  try: values=[float(row[key]) for key in required]
  except (KeyError,TypeError,ValueError): return False
  bounded=('orientation_count_score_at_0p02deg','count_score','identity_f1','orientation_count_score_at_0p01deg')
  return all(math.isfinite(value) for value in values) and all(0.0<=float(row[key])<=1.0 for key in bounded)
 return [row for row in rows if valid(row)]
def minimised_metric(row,key,digits=None):
 """Numeric ordering value; an undefined error is strictly worst in memory."""
 raw=row.get(key)
 if raw is None: return math.inf
 try: value=float(raw)
 except (TypeError,ValueError): return math.inf
 if not math.isfinite(value): return math.inf
 return round(value,digits) if digits is not None else value
def maximised_metric(row,key,digits=None):
 raw=row.get(key)
 if raw is None: return -math.inf
 try: value=float(raw)
 except (TypeError,ValueError): return -math.inf
 if not math.isfinite(value): return -math.inf
 return round(value,digits) if digits is not None else value
def rank_key(row):
 """Deterministic total ordering for the orientation-count policy.

 Undefined maximisation metrics sort after every finite value; undefined
 minimisation metrics likewise sort after every finite value.  The sentinels
 are used only in memory and are never written to the result store.
 """
 seed=row.get('candidate_order_seed',row.get('seed',0))
 try: seed=int(seed or 0)
 except (TypeError,ValueError): seed=sys.maxsize
 return (-maximised_metric(row,'orientation_count_score_at_0p02deg'),-maximised_metric(row,'identity_f1'),-maximised_metric(row,'orientation_count_score_at_0p01deg'),-maximised_metric(row,'count_score'),-maximised_metric(row,'panoptic_quality_at_0p5'),-maximised_metric(row,'matched_iou_mean'),-maximised_metric(row,'interface_f1'),minimised_metric(row,'boundary_assd_um'),minimised_metric(row,'vi_total_bits'),-maximised_metric(row,'ari'),str(row.get('config_hash','')),seed)
def rank(rows): return sorted(rows,key=rank_key)
def object_vector(row):
 return np.asarray((-maximised_metric(row,'orientation_count_score_at_0p02deg'),-maximised_metric(row,'count_score'),-maximised_metric(row,'identity_f1'),-maximised_metric(row,'panoptic_quality_at_0p5'),-maximised_metric(row,'matched_iou_mean'),-maximised_metric(row,'interface_f1'),minimised_metric(row,'boundary_assd_um')),float)
def pareto_rows(rows):
 frontier=[]
 for row in rows:
  vector=object_vector(row)
  if any(np.all(old_vector<=vector) and np.any(old_vector<vector) for _,old_vector in frontier): continue
  frontier=[(old,old_vector) for old,old_vector in frontier if not (np.all(vector<=old_vector) and np.any(vector<old_vector))]; frontier.append((row,vector))
 return [row for row,_ in frontier]
def parameter_signature(p,w):
 fp=w.footprint(p['footprint_radius_um']); requirement=oc.neighbour_requirements(p['footprint_tolerance'],int(fp.sum()))
 return (p['local_threshold_deg'],p['global_threshold_deg'],hashlib.sha256(requirement.tobytes()).hexdigest(),p['footprint_radius_um'],int(p['min_cell_size']),p['kam_radius_um'])
def count_focus_key(row):
 """Balance count, orientation and identity so exact count cannot win alone."""
 values=[max(0.0,maximised_metric(row,key)) for key in ('count_score','orientation_count_score_at_0p02deg','identity_f1')]
 balanced=math.prod(values)**(1/3) if all(math.isfinite(value) for value in values) else -math.inf
 return (-balanced,-maximised_metric(row,'count_score'),rank_key(row))
def diverse_sources(rows):
 eligible=selectable(rows); frontier=pareto_rows(eligible)
 return {
  'official':rank(eligible),
  'count_focused':sorted(eligible,key=count_focus_key),
  'identity_focused':sorted(eligible,key=lambda row:(-maximised_metric(row,'identity_f1'),-maximised_metric(row,'orientation_count_score_at_0p02deg'),-maximised_metric(row,'count_score'),rank_key(row))),
  'boundary_focused':sorted(frontier,key=lambda row:(minimised_metric(row,'boundary_assd_um'),rank_key(row))),
  'full_non_dominated':rank(frontier),
 }
def diverse_pool(rows,limit=None,*,include_full_frontier=True):
 sources=diverse_sources(rows); chosen=[]; seen=set(); index=0; ordered=list(sources.values())
 def add(row):
  key=tuple(row[name] for name in oc.PARAMETER_NAMES)
  if key not in seen: seen.add(key); chosen.append(row)
 while any(index<len(source) for source in ordered) and (limit is None or len(chosen)<limit):
  for source in ordered:
   if index<len(source): add(source[index])
   if limit is not None and len(chosen)>=limit: break
  index+=1
 if include_full_frontier:
  for row in sources['full_non_dominated']: add(row)
 return chosen,sources
def adaptive(existing,n,w,batch):
 selectable_rows=selectable(existing); leaders,_=diverse_pool(selectable_rows,120,include_full_frontier=True)
 if not leaders: raise RuntimeError('no selectable frozen-broad evidence for adaptive generation')
 rng=np.random.default_rng(20260816+batch); out=[]; seen={parameter_signature({k:r[k] for k in oc.PARAMETER_NAMES},w) for r in selectable_rows if all(k in r for k in oc.PARAMETER_NAMES)}; scale=.35 if batch==0 else .16
 while len(out)<n:
  c=leaders[int(rng.integers(len(leaders)))]; p={k:c[k] for k in oc.PARAMETER_NAMES}
  for k in ('local_threshold_deg','footprint_radius_um','kam_radius_um'): p[k]=float(np.clip(np.exp(np.log(p[k])+rng.normal(0,scale)),.002 if k=='local_threshold_deg' else .3,.16 if k=='local_threshold_deg' else (4 if k=='footprint_radius_um' else 2)))
  p['global_threshold_deg']= -1.0 if p['global_threshold_deg']<=0 and rng.random()<.8 else float(np.clip(np.exp(np.log(max(p['global_threshold_deg'],.02))+rng.normal(0,scale)),.008,1.0))
  p['footprint_tolerance']=float(np.clip(p['footprint_tolerance']+rng.normal(0,.1*scale/.35),.04,.96)); p['min_cell_size']=int(np.clip(round(p['min_cell_size']+rng.normal(0,40*scale/.35)),5,250)); p=canonical(p,w); signature=parameter_signature(p,w)
  if signature not in seen: seen.add(signature); out.append(p)
 return out
def rank_configurations(rows):
 groups={}
 for row in selectable(rows): groups.setdefault(tuple(row[k] for k in oc.PARAMETER_NAMES),[]).append(row)
 ranked=[]
 for values in groups.values():
  representative={k:values[0][k] for k in oc.PARAMETER_NAMES}; n=len(values)
  maximum=lambda k:np.asarray([maximised_metric(r,k) for r in values],float); minimum=lambda k:np.asarray([minimised_metric(r,k) for r in values],float)
  oc02=maximum('orientation_count_score_at_0p02deg'); identity=maximum('identity_f1'); oc01=maximum('orientation_count_score_at_0p01deg'); count=maximum('count_score'); pq=maximum('panoptic_quality_at_0p5'); iou=maximum('matched_iou_mean'); interface=maximum('interface_f1'); boundary=minimum('boundary_assd_um'); vi=minimum('vi_total_bits'); ari=maximum('ari')
  lower=float(np.min(oc02)) if n<20 else float(np.percentile(oc02,5)); lower_name='minimum' if n<20 else 'p05'
  representative['_seed_summary']={'n_seed_orders':n,'orientation_count_score_at_0p02deg_median':float(np.median(oc02)),'orientation_count_score_at_0p02deg_minimum':float(np.min(oc02)),'orientation_count_score_at_0p02deg_p05':float(np.percentile(oc02,5)) if n>=20 else None,'orientation_count_score_at_0p02deg_sd':float(np.std(oc02,ddof=1)) if n>1 else 0.0,'lower_tail_statistic':lower_name,'lower_tail_value':lower,'identity_f1_median':float(np.median(identity)),'count_score_median':float(np.median(count)),'interface_f1_median':float(np.median(interface)),'boundary_assd_um_median':float(np.median(boundary))}
  representative['_rank']=(-float(np.median(oc02)),-lower,-float(np.median(identity)),-float(np.median(count)),float(np.median(boundary)),-float(np.median(oc01)),-float(np.median(pq)),-float(np.median(iou)),-float(np.median(interface)),float(np.median(vi)),-float(np.median(ari)),float(np.std(oc02,ddof=1)) if n>1 else 0.0,oc.Config(**{name:representative[name] for name in oc.PARAMETER_NAMES}).key())
  ranked.append(representative)
 return sorted(ranked,key=lambda p:p['_rank'])
def transition_signature(row):
 return (maximised_metric(row,'orientation_count_score_at_0p02deg',8),maximised_metric(row,'identity_f1',8),maximised_metric(row,'orientation_count_score_at_0p01deg',8),maximised_metric(row,'count_score',8),maximised_metric(row,'panoptic_quality_at_0p5',8),maximised_metric(row,'matched_iou_mean',8),maximised_metric(row,'interface_f1',8),-minimised_metric(row,'boundary_assd_um',8),-minimised_metric(row,'vi_total_bits',8),maximised_metric(row,'ari',8))
def prepare_adaptive_transition(current,n,w,batch):
 selected=selectable(current); before=rank(selected)[0]
 return before,transition_signature(before),[object_vector(row) for row in pareto_rows(selected)],adaptive(current,n,w,batch)
def finalist_pool(rows,n=50):
 return diverse_pool(rows,n,include_full_frontier=True)[0]
def audit_destination_store():
 """Validate that the continuation namespace contains new-policy rows only."""
 path=OUT/'evaluations.jsonl'
 if not path.exists(): return {'rows':0,'duplicate_full_hashes':0,'stages':{}},[]
 _,rows=strict_jsonl(path); hashes=[]
 for line,row in enumerate(rows,1):
  if row.get('stage')=='broad': raise RuntimeError(f'broad rerun/import row in continuation store at line {line}')
  if row.get('selection_policy_id')!=SELECTION_POLICY_ID: raise RuntimeError(f'foreign/old selection policy at destination line {line}')
  if row.get('scientific_metrics_version')!=SCIENTIFIC_METRICS_VERSION: raise RuntimeError(f'wrong scientific metrics version at destination line {line}')
  if row.get('schema_version')!=RESULT_SCHEMA_VERSION: raise RuntimeError(f'wrong result schema version at destination line {line}')
  identity_payload=row.get('configuration_identity') or {}
  if identity_payload.get('selection_policy_id')!=SELECTION_POLICY_ID: raise RuntimeError(f'foreign configuration identity at destination line {line}')
  expected=sha256_json(identity_payload)
  if row.get('config_hash')!=expected: raise RuntimeError(f'full-hash mismatch at destination line {line}')
  hashes.append(row.get('config_hash'))
 if len(hashes)!=len(set(hashes)): raise RuntimeError('duplicate full hashes in continuation store')
 return {'rows':len(rows),'duplicate_full_hashes':0,'stages':dict(collections.Counter(str(row.get('stage')) for row in rows))},rows
def basis_hash(rows):
 return sha256_json(sorted(str(row.get('config_hash')) for row in selectable(rows)))
def adaptive_plan_path(stage_number): return OUT/f'adaptive_{stage_number}_plan.json'
def adaptive_plan(prior,n,w,stage_number,source_sha256):
 """Create or validate an immutable deterministic adaptive-stage plan."""
 batch=stage_number-1; path=adaptive_plan_path(stage_number); parameters=adaptive(prior,n,w,batch)
 candidates=[]
 for p in parameters:
  clean={key:p[key] for key in oc.PARAMETER_NAMES}; digest,_=config_key(clean,0)
  candidates.append({'parameters':clean,'seed':0,'config_hash':digest})
 expected={'plan_version':1,'stage':f'adaptive_{stage_number}','selection_policy_id':SELECTION_POLICY_ID,
  'selection_policy_version':SELECTION_POLICY_VERSION,'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,
  'result_schema_version':RESULT_SCHEMA_VERSION,'frozen_broad_source_sha256':source_sha256,
  'generator_seed':20260816+batch,'basis_selectable_rows':len(selectable(prior)),
  'basis_full_hashes_sha256':basis_hash(prior),'generated_candidate_count':len(candidates),
  'candidate_full_hashes_sha256':sha256_json([item['config_hash'] for item in candidates]),
  'candidates':candidates}
 if path.exists():
  current=json.loads(path.read_text()); mismatch={key:(current.get(key),value) for key,value in expected.items() if key!='candidates' and current.get(key)!=value}
  if current.get('candidates')!=candidates: mismatch['candidates_sha256']=(sha256_json(current.get('candidates')),sha256_json(candidates))
  if mismatch: raise RuntimeError(f'adaptive plan mismatch for stage {stage_number}: {mismatch}')
  return current
 atomic(path,expected); return expected
def plan_parameters(plan): return [dict(item['parameters']) for item in plan['candidates']]
def plan_complete(plan,destination_rows):
 completed={row.get('config_hash') for row in destination_rows}
 return all(item['config_hash'] in completed for item in plan['candidates'])
def dry_run_report(audit,evidence,w,*,create_plan=True):
 """Freeze broad-only ranking, diversity, Pareto, and adaptive generation."""
 if any(row.get('stage')!='broad' for row in evidence): raise RuntimeError('dry run evidence is not broad-only')
 ranked=rank(evidence); frontier=pareto_rows(evidence); pool,sources=diverse_pool(evidence,120,include_full_frontier=True)
 plan=adaptive_plan(evidence,ADAPTIVE_BATCH_SIZE,w,1,audit['source_jsonl_sha256']) if create_plan else None
 def summary(row):
  keys=('config_hash','n_cells_pred','n_cells_true','count_score','orientation_correct_cells_at_0p02deg','orientation_correct_f1_at_0p02deg','orientation_count_score_at_0p02deg','one_to_one_recovered_cells','identity_f1','matched_iou_mean','boundary_assd_um','ari')
  return {key:row.get(key) for key in keys}
 report={**audit,'selection_policy_id':SELECTION_POLICY_ID,'selection_policy_version':SELECTION_POLICY_VERSION,
  'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,'result_schema_version':RESULT_SCHEMA_VERSION,
  'evidence_stage_filter':'stage == "broad"','old_policy_adaptive_rows_in_evidence':0,
  'leading_candidate':summary(ranked[0]),'top_candidates':[summary(row) for row in ranked[:25]],
  'top_25_predicted_cell_counts':[int(row['n_cells_pred']) for row in ranked[:25]],
  'pareto_count':len(frontier),'pareto_config_hashes':[row['config_hash'] for row in rank(frontier)],
  'adaptive_pool_unique_candidates':len(pool),
  'adaptive_pool_source_sizes':{key:len(value) for key,value in sources.items()},
  'generated_adaptive_candidate_count':plan['generated_candidate_count'] if plan else None,
  'adaptive_candidate_full_hashes_sha256':plan['candidate_full_hashes_sha256'] if plan else None,
  'validated':True,'created':utc_now()}
 atomic(OUT/'frozen_broad_dry_run.json',report); return report
def live_accounting(rows,planned):
 hashes=[row.get('config_hash') for row in rows if row.get('config_hash')]
 trials={(str(row.get('config_key')),int(row.get('candidate_order_seed',row.get('seed',0)) or 0)) for row in rows if row.get('config_key') is not None}
 scientific={(str(row.get('config_key')),int(row.get('candidate_order_seed',row.get('seed',0)) or 0)) for row in selectable(rows)}
 return {'durable_rows':len(rows),'superseded_pre_v3_rows':sum(row.get('scientific_metrics_version')!=SCIENTIFIC_METRICS_VERSION for row in rows),'unique_full_hashes':len(set(hashes)),'unique_trials':len(trials),'selectable_v3_trials':len(scientific),'planned_unique_v3_trials':int(planned)}
def run_tasks(parameters,stage,seeds,workers,store,target=None):
 global STOP
 if stage=='broad': raise RuntimeError('broad execution is disabled for this continuation policy')
 result_store=ResultStore(store,schema_version=RESULT_SCHEMA_VERSION); existing=list(result_store.rows); done=set(result_store.hashes); scheduled=set(done); tasks=[]
 for p in parameters:
  sampling=p.get('_sampling'); p={k:v for k,v in p.items() if not k.startswith('_')}
  for seed in seeds:
   key,_=config_key(p,seed)
   if key not in scheduled:
    scheduled.add(key); tasks.append({'parameters':p,'seed':seed,'stage':stage,'sampling':sampling,
     # Boundary metrics remain secondary to object identity, but collecting
     # them here is necessary for unbiased parameter-response profiles. Rows
     # checkpointed before this audit fix remain valid for object selection
     # and are explicitly absent, rather than imputed, in boundary profiles.
     'boundary':True})
 if target: tasks=tasks[:target]
 avail,total,swap0=memory(); started=time.monotonic(); peak=0; failures=0; ctx=mp.get_context('spawn')
 with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_worker,max_tasks_per_child=15) as pool:
  future={}
  while tasks or future:
   avail,total,swap=memory(); growth=max(0,swap-swap0); aggregate,max_process=tree_rss()
   unsafe=aggregate>=10*1024**3 or avail<max(6*1024**3,.25*total) or growth>256*1024**2
   if unsafe: STOP=True
   if aggregate>=12*1024**3: STOP=True
   while tasks and len(future)<workers and not STOP and not unsafe:
    t=tasks.pop(0); future[pool.submit(evaluate_task,t)]=t
   if not future: break
   ready,_=wait(future,timeout=2,return_when=FIRST_COMPLETED)
   for f in ready:
    task=future.pop(f)
    try: raw=f.result()
    except Exception as exc:
     raw=worker_future_failure(task,exc)
     # A dead process may make this executor unusable. Persist the failure and
     # stop submitting; an ordinary resume reconstructs all state from disk.
     if type(exc).__name__ in ('BrokenProcessPool','TerminatedWorkerError'): STOP=True
    expected,identity_payload=config_key(task['parameters'],task['seed'])
    row=prepare_worker_result(raw,task,expected,identity_payload)
    if expected in done: continue
    added,row=result_store.append(row,identity_payload)
    if not added: continue
    done=set(result_store.hashes); failures+=int(row['status_category']!='ok')
    value=row.get('worker_peak_rss_bytes')
    if value is not None: peak=max(peak,int(value))
    update_resource_history(aggregate,max_process,avail,growth,peak)
    elapsed=max(time.monotonic()-started,1e-9); rate=(len(done)-len(existing))/elapsed
    atomic(OUT/'progress.json',{"stage":stage,"selection_policy_id":SELECTION_POLICY_ID,"scientific_metrics_version":SCIENTIFIC_METRICS_VERSION,"worker_count":workers,"completed_count":len(done),"completed_this_run":len(done)-len(existing),"pending_count":len(tasks),"in_flight_count":len(future),**live_accounting(result_store.rows,len(scheduled)),"status_counts":result_store.counts(),"trials_per_second":rate,"aggregate_process_tree_rss_bytes":aggregate,"max_process_rss_bytes":max_process,"available_ram_bytes":avail,"swap_baseline_bytes":swap0,"swap_growth_bytes":growth,"max_worker_peak_rss_bytes":peak,"eta_seconds":len(tasks)/rate if rate else None,"safety_stop":bool(STOP or unsafe),"updated":dt.datetime.now().astimezone().isoformat()})
 return list(result_store.rows)
def smoke(w,workers):
 selected=json.loads((HERE/'oracle_results'/'selected_solutions.json').read_text())
 names=('balanced','max_ari','min_count','min_vi')
 ps=[canonical({k:selected[n][k] for k in oc.PARAMETER_NAMES},w) for n in names]
 serial=[]; init_worker()
 for i in range(20): serial.append(evaluate_task({'parameters':ps[i%4],'seed':i,'stage':'serial_smoke'}))
 atomic(OUT/'serial_smoke.json',serial)
 parallel=run_tasks(ps,'parallel_smoke',range(25),workers,OUT/'parallel_smoke.jsonl')
 # Ten identical configuration/seed evaluations must reproduce serial exactly.
 checks=[]
 for row in serial[:10]:
  repeat=evaluate_task({'parameters':{k:row[k] for k in oc.PARAMETER_NAMES},'seed':row['candidate_order_seed'],'stage':'serial_repeat'})
  ignore={'runtime_seconds','worker_peak_rss_bytes','stage'}; bad=[]
  for k in set(row)|set(repeat):
   if k in ignore: continue
   a,b=row.get(k),repeat.get(k)
   if isinstance(a,float) or isinstance(b,float):
    if not np.isclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True): bad.append(k)
   elif a!=b: bad.append(k)
  checks.append({'config_hash':row['config_hash'],'mismatches':bad})
 atomic(OUT/'preflight_report.json',{'serial_trials':len(serial),'parallel_trials':len(parallel),'reproducibility_checks':checks,'passed':len(serial)>=20 and len(parallel)>=100 and not any(x['mismatches'] for x in checks) and all(x['status']=='ok' for x in serial+parallel)})
def complete_all(workers):
 """Run requested downstream stages serially after primary completion."""
 commands=[
  [sys.executable,str(HERE/'two_stage_saltelli.py'),'--n','512','--workers',str(workers)],
  [sys.executable,str(HERE/'two_stage_analyze.py')],
  [sys.executable,str(HERE/'two_stage_paired_comparison.py'),'--workers',str(workers),'--configurations','32','--seeds','21'],
  [sys.executable,str(HERE/'two_stage_strain_suite.py'),'--workers',str(workers),'--broad','256','--adaptive','128'],
  [sys.executable,'-m','pytest','-q',str(HERE.parent/'tests')],
  [sys.executable,'-m','pytest','-q',*[str(path) for path in sorted(HERE.glob('test_*.py'))]],
  ['git','diff','--check'],
  [sys.executable,str(HERE/'two_stage_final_audit.py')],
 ]
 for index,command in enumerate(commands,1):
  atomic(OUT/'downstream_progress.json',{'step':index,'total_steps':len(commands),'command':command,'started':utc_now(),'status':'running'})
  subprocess.run(command,check=True,env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","VECLIB_MAXIMUM_THREADS":"1","MALLOC_ARENA_MAX":"2"})
  atomic(OUT/'downstream_progress.json',{'step':index,'total_steps':len(commands),'command':command,'finished':utc_now(),'status':'completed'})
def adaptive_stage_number(row):
 match=re.fullmatch(r'adaptive_(\d+)',str(row.get('stage','')))
 return int(match.group(1)) if match else None
def run_frozen_broad_continuation(w,audit,evidence,workers):
 """Run new-policy adaptive stages using only pinned broad evidence as input."""
 global STOP
 destination_path=OUT/'evaluations.jsonl'; no_improvement=0; convergence=[]; stage_number=1
 while True:
  _,destination_rows=audit_destination_store()
  future=[row for row in destination_rows if adaptive_stage_number(row) is not None and adaptive_stage_number(row)>stage_number]
  current_stage=[row for row in destination_rows if adaptive_stage_number(row)==stage_number]
  if future and not current_stage: raise RuntimeError(f'destination skips adaptive_{stage_number} before later stages')
  prior=evidence+[row for row in destination_rows if adaptive_stage_number(row) is not None and adaptive_stage_number(row)<stage_number]
  selected=selectable(prior)
  if not selected: raise RuntimeError(f'no selectable evidence before adaptive_{stage_number}')
  before=rank(selected)[0]; before_signature=transition_signature(before); before_front=[object_vector(row) for row in pareto_rows(selected)]
  plan=adaptive_plan(prior,ADAPTIVE_BATCH_SIZE,w,stage_number,audit['source_jsonl_sha256'])
  if future and not plan_complete(plan,destination_rows): raise RuntimeError(f'later adaptive rows exist before adaptive_{stage_number} is complete')
  destination_rows=run_tasks(plan_parameters(plan),f'adaptive_{stage_number}',[0],workers,destination_path)
  if STOP: return
  if not plan_complete(plan,destination_rows): raise RuntimeError(f'adaptive_{stage_number} returned without completing its immutable plan')
  batch_rows=[row for row in selectable(destination_rows) if adaptive_stage_number(row)==stage_number]
  combined=prior+batch_rows; after=rank(selectable(combined))[0]; signature=transition_signature(after)
  pareto_gain=any(any(np.all(object_vector(row)<=old) and np.any(object_vector(row)<old) for old in before_front) for row in batch_rows)
  meaningful=bool(signature>before_signature or pareto_gain); no_improvement=0 if meaningful else no_improvement+1
  convergence.append({'stage':f'adaptive_{stage_number}','best_signature':signature,
   'lexicographic_improvement':bool(signature>before_signature),'pareto_dominance_improvement':bool(pareto_gain),
   'meaningful_improvement':meaningful,'consecutive_batches_without_meaningful_improvement':no_improvement})
  atomic(OUT/'adaptive_convergence.json',{'selection_policy_id':SELECTION_POLICY_ID,
   'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,'result_schema_version':RESULT_SCHEMA_VERSION,
   'frozen_broad_source_sha256':audit['source_jsonl_sha256'],'batches':convergence,
   'last_batch':stage_number,'consecutive_batches_without_meaningful_improvement':no_improvement})
  if stage_number>=2 and no_improvement>=2: break
  stage_number+=1
 _,destination_rows=audit_destination_store(); current=evidence+destination_rows
 leaders=[{k:r[k] for k in oc.PARAMETER_NAMES} for r in finalist_pool(current,50)]
 screened=run_tasks(leaders,'five_seed_screen',range(5),workers,destination_path)
 if STOP: return
 leader_keys={tuple(p[k] for k in oc.PARAMETER_NAMES) for p in leaders}; screened_leaders=[r for r in screened if all(k in r for k in oc.PARAMETER_NAMES) and tuple(r[k] for k in oc.PARAMETER_NAMES) in leader_keys]; finals=[]
 for candidate in rank_configurations(screened_leaders)[:25]: finals.append({k:candidate[k] for k in oc.PARAMETER_NAMES})
 run_tasks(finals,'twenty_seed_final',range(20),workers,destination_path)
def main():
 global IDENTITY,STOP
 ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=8)
 actions=ap.add_mutually_exclusive_group(required=True); actions.add_argument('--frozen-broad-dry-run',action='store_true'); actions.add_argument('--continue-frozen-broad',action='store_true'); actions.add_argument('--search',action='store_true',help=argparse.SUPPRESS)
 ap.add_argument('--complete-all',action='store_true'); a=ap.parse_args()
 if a.search: raise RuntimeError('broad execution is disabled; use --continue-frozen-broad')
 audit,evidence=audit_frozen_broad_source(); manifest=initialise_policy_namespace(audit); IDENTITY=manifest
 audit_destination_store(); w=oc.load_workspace(CACHE); report=dry_run_report(audit,evidence,w,create_plan=True)
 if a.frozen_broad_dry_run:
  print(json.dumps(report,indent=2,sort_keys=True)); return
 if a.workers!=8: raise RuntimeError('the revised continuation requires exactly eight workers')
 lock=(OUT/'two_stage_oracle.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 signal.signal(signal.SIGTERM,lambda s,f:globals().__setitem__('STOP',True)); signal.signal(signal.SIGINT,lambda s,f:globals().__setitem__('STOP',True))
 manifest_hash=sha256_bytes((OUT/'implementation_audit.json').read_bytes())
 atomic(OUT/'pid.json',{'pid':os.getpid(),'workers':a.workers,'started':dt.datetime.now().astimezone().isoformat(),'coordinator_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'manifest_sha256':manifest_hash,'frozen_broad_source_sha256':audit['source_jsonl_sha256'],'selection_policy_id':SELECTION_POLICY_ID,'scientific_metrics_version':SCIENTIFIC_METRICS_VERSION,'result_schema_version':RESULT_SCHEMA_VERSION,'stage_mode':'adaptive_only_from_frozen_broad'})
 run_frozen_broad_continuation(w,audit,evidence,a.workers)
 if a.complete_all and not STOP: complete_all(a.workers)
if __name__=='__main__':
 try: main()
 except Exception:
  failure=traceback.format_exc()
  if OUT.resolve() not in {LEGACY_OUT.resolve(),FROZEN_BROAD_OUT.resolve()}:
   try: OUT.mkdir(parents=True,exist_ok=True); (OUT/'failure.log').write_text(failure)
   except Exception: pass
  sys.stderr.write(failure); raise

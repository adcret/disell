#!/usr/bin/env python3
"""Read-only integrity/counter snapshot of the active two-stage store."""
from __future__ import annotations
import argparse,collections,json,os
import math
from pathlib import Path
from two_stage_accounting import accounting
from two_stage_store import REQUIRED_ENVELOPE,VALID_CATEGORIES
HERE=Path(__file__).resolve().parent; DEFAULT=HERE/'two_stage_oracle_results'
def atomic(path,value):
 temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(json.dumps(value,indent=2,sort_keys=True)); temporary.replace(path)
def nonfinite(value,path='result'):
 if isinstance(value,dict):
  for key,child in value.items(): yield from nonfinite(child,f'{path}.{key}')
 elif isinstance(value,(list,tuple)):
  for index,child in enumerate(value): yield from nonfinite(child,f'{path}[{index}]')
 elif isinstance(value,float) and not math.isfinite(value): yield path
def main():
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--out-dir',type=Path,default=DEFAULT); args=parser.parse_args(); path=args.out_dir/'evaluations.jsonl'; raw=path.read_bytes(); rows=[]; malformed=[]; schema=[]
 for line_number,line in enumerate(raw.splitlines(keepends=True),1):
  try:
   if not line.endswith(b'\n'): raise ValueError('partial/non-terminated line')
   row=json.loads(line,parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'non-standard JSON constant {value}'))); rows.append(row); missing=[key for key in REQUIRED_ENVELOPE if key not in row]
   if missing or row.get('status_category') not in VALID_CATEGORIES: schema.append({'line':line_number,'missing':missing,'status_category':row.get('status_category')})
  except Exception as exc: malformed.append({'line':line_number,'error':str(exc)})
 hashes=[row.get('config_hash') for row in rows]; categories=collections.Counter(row.get('status_category') for row in rows); scientific=collections.Counter()
 for row in rows:
  category=row.get('status_category'); version=row.get('scientific_metrics_version'); scientific[f'v{version}_{category}']+=1
 nonfinite_stored=[{'line':line,'fields':list(nonfinite(row))} for line,row in enumerate(rows,1) if list(nonfinite(row))]
 policy_ids={row.get('selection_policy_id') for row in rows}
 if policy_ids=={'object_orientation_count_geomean_v1'}:
  trial_keys={(row.get('config_key'),row.get('candidate_order_seed',row.get('seed'))) for row in rows}
  versioned={'durable_rows':len(rows),'unique_full_hashes':len(set(hashes)),
   'unique_trials':len(trial_keys),'selectable_policy_trials':sum(row.get('status_category')=='ok' for row in rows),
   'adaptive_policy_trials':sum(str(row.get('stage','')).startswith('adaptive_') for row in rows),
   'broad_rows_in_continuation_namespace':sum(row.get('stage')=='broad' for row in rows),
   'result_schema_versions':dict(collections.Counter(row.get('schema_version') for row in rows)),
   'selection_policy_ids':dict(collections.Counter(row.get('selection_policy_id') for row in rows))}
 else: versioned=accounting(rows)
 report={'store':str(path),'rows':len(rows),'bytes':len(raw),'newline_terminated':raw.endswith(b'\n'),'duplicate_full_hashes':len(hashes)-len(set(hashes)),'malformed_records':malformed,'stored_nonfinite_records':nonfinite_stored,'schema_errors':schema,'status_categories':dict(categories),'scientific_version_categories':dict(scientific),**versioned,'schema_v2_schema_error_results':sum(row.get('scientific_metrics_version')==2 and row.get('status_category')=='coordinator/schema_error' for row in rows),'schema_v2_infrastructure_failures':sum(row.get('scientific_metrics_version')==2 and row.get('status_category')=='worker_exception' for row in rows),'schema_v2_expected_algorithmic_invalids':sum(row.get('scientific_metrics_version')==2 and row.get('status_category')=='expected_algorithmic_invalid' for row in rows),'schema_v2_candidate_saturated':sum(row.get('scientific_metrics_version')==2 and row.get('status_category')=='candidate_saturated' for row in rows),'schema_v2_final_saturated':sum(row.get('scientific_metrics_version')==2 and row.get('status_category')=='final_saturated' for row in rows),'pid_metadata':json.loads((args.out_dir/'pid.json').read_text()) if (args.out_dir/'pid.json').exists() else None,'coordinator_progress':json.loads((args.out_dir/'progress.json').read_text()) if (args.out_dir/'progress.json').exists() else None}
 atomic(args.out_dir/'live_integrity_audit.json',report); print(json.dumps(report,indent=2))
 if malformed or nonfinite_stored or schema or report['duplicate_full_hashes']: raise SystemExit(2)
if __name__=='__main__': main()

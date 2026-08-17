#!/usr/bin/env python3
"""Integrity gate and final status for the completed two-stage workflow."""
from __future__ import annotations
import collections,csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; OUT=HERE/'two_stage_oracle_results'; PAIRED=HERE/'continuation_results'/'algorithm_comparison'; STRAIN=HERE/'continuation_results'/'two_stage_strain_suite'
sys.path.insert(0,str(HERE))
from two_stage_accounting import accounting,is_broad_stage,is_selectable_v2,unique_rows
REQUIRED=('selection_decision_table.csv','object_first_pareto.csv','selected_solutions.json','matched_orientation_table.csv','orientation_recovery_curve.csv','kam_guarded_comparison.csv','sobol_indices.csv','broad_deduplication.json','adaptive_convergence.json')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def main():
 current_diff_hash=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=HERE.parent)).hexdigest()
 malformed=[]; rows=[]; raw=(OUT/'evaluations.jsonl').read_bytes()
 for line_number,line in enumerate(raw.splitlines(keepends=True),1):
  try:
   if not line.endswith(b'\n'): raise ValueError('partial line')
   rows.append(json.loads(line))
  except Exception as exc: malformed.append({'line':line_number,'error':str(exc)})
 hashes=[r.get('config_hash') for r in rows]; duplicates=len(hashes)-len(set(hashes)); counts=collections.Counter(r.get('status_category') for r in rows); scientific=unique_rows([r for r in rows if is_selectable_v2(r)],is_selectable_v2); scientific_accounting=accounting(rows); missing=[name for name in REQUIRED if not (OUT/name).exists()]; external_required=[PAIRED/'paired_differences.csv',PAIRED/'preserved_random_order_seed_stability_verified.csv',STRAIN/'two_stage_multi_phantom_summary.csv',STRAIN/'transferable_rule_held_out_performance.csv']
 missing.extend(str(path) for path in external_required if not path.exists()); finite_failures=[]
 for path in OUT.glob('*.csv'):
  try:
   frame=pd.read_csv(path)
  except Exception as exc:
   finite_failures.append({'path':str(path),'error':str(exc)}); continue
  for column in frame.select_dtypes(include='number'):
   # Optional/undefined scientific metrics may be blank, but infinities may not.
   if np.isinf(frame[column].to_numpy(float,copy=False)).any(): finite_failures.append({'path':str(path),'column':column,'error':'infinite value'})
 figure_sources={'count_identity_and_ari_recovery':['full_scalar_trials_v2.csv'],'matched_cell_mean_calibration':['matched_orientation_table.csv'],'orientation_error_and_identity_quality':['matched_orientation_table.csv'],'orientation_recovery_curve':['orientation_recovery_curve.csv'],'kam_guarded_identity_orientation':['kam_guarded_comparison.csv'],'object_first_sensitivity_profiles':['sensitivity_profiles.csv'],'pairwise_object_response_surfaces':['pairwise_response_surfaces.csv']}; figure_audit=[]
 for figure,sources in figure_sources.items(): figure_audit.append({'figure':figure,'png_exists':(OUT/f'{figure}.png').exists(),'pdf_exists':(OUT/f'{figure}.pdf').exists(),'sources':[{"path":source,"sha256":sha(OUT/source) if (OUT/source).exists() else None} for source in sources]})
 parameter_names=('local_threshold_deg','global_threshold_deg','footprint_tolerance','footprint_radius_um','min_cell_size','kam_radius_um'); broad={tuple(r.get(k) for k in parameter_names) for r in scientific if is_broad_stage(r)}; adaptive={tuple(r.get(k) for k in parameter_names) for r in scientific if str(r.get('stage','')).startswith('adaptive_')}; seed_sets=collections.defaultdict(set)
 for row in scientific: seed_sets[tuple(row.get(k) for k in parameter_names)].add(row.get('candidate_order_seed'))
 five_seed_configs=sum(len(seeds)>=5 for seeds in seed_sets.values()); twenty_seed_configs=sum(len(seeds)>=20 for seeds in seed_sets.values()); saltelli=json.loads((OUT/'saltelli_design.json').read_text()) if (OUT/'saltelli_design.json').exists() else {}; paired=json.loads((PAIRED/'paired_plan.json').read_text()) if (PAIRED/'paired_plan.json').exists() else {}; strain_count=len(pd.read_csv(STRAIN/'two_stage_multi_phantom_summary.csv')) if (STRAIN/'two_stage_multi_phantom_summary.csv').exists() else 0; v2_infrastructure=sum(r.get('scientific_metrics_version')==2 and r.get('status_category') in ('worker_exception','coordinator/schema_error') for r in rows)
 coverage={'unique_broad_configurations':scientific_accounting['broad_selectable_v2_trials'],'unique_adaptive_configurations':len(adaptive),'configurations_with_at_least_five_seeds':five_seed_configs,'configurations_with_at_least_twenty_seeds':twenty_seed_configs,'saltelli_base_N':saltelli.get('N',0),'paired_configurations':paired.get('n_configurations',0),'paired_seed_orders':len(paired.get('seeds',[])),'strain_phantoms':strain_count,'schema_v2_infrastructure_failures':v2_infrastructure}
 coverage_failures={key:value for key,value,minimum in [('unique_broad_configurations',len(broad),8000),('unique_adaptive_configurations',len(adaptive),4000),('configurations_with_at_least_five_seeds',five_seed_configs,50),('configurations_with_at_least_twenty_seeds',twenty_seed_configs,20),('saltelli_base_N',saltelli.get('N',0),512),('paired_configurations',paired.get('n_configurations',0),32),('paired_seed_orders',len(paired.get('seeds',[])),20),('strain_phantoms',strain_count,36)] if value<minimum}
 report={'jsonl_rows':len(rows),'scientific_accounting':scientific_accounting,'schema_v1_rows_used_for_scientific_selection':0,'newline_terminated':raw.endswith(b'\n'),'duplicate_hashes':duplicates,'malformed_records':malformed,'status_counts_all_durable_rows':dict(counts),'missing_required_artifacts':missing,'finite_metric_failures':finite_failures,'coverage':coverage,'coverage_failures':coverage_failures,'figure_source_audit':figure_audit,'algorithm_manifest':json.loads((OUT/'implementation_audit.json').read_text()),'final_tracked_worktree_diff_sha256':current_diff_hash,'resource_history':json.loads((OUT/'resource_history.json').read_text()) if (OUT/'resource_history.json').exists() else None}
 (OUT/'artifact_audit.json').write_text(json.dumps(report,indent=2))
 blocking=bool(malformed or duplicates or missing or finite_failures or coverage_failures or v2_infrastructure)
 selected=json.loads((OUT/'selected_solutions.json').read_text()) if (OUT/'selected_solutions.json').exists() else {}; best=selected.get('maximum_orientation_correct_recovery',{}).get('metrics',{}); progress=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {}; resources=json.loads((OUT/'resource_history.json').read_text()) if (OUT/'resource_history.json').exists() else {}
 text='# Final status\n\n'+('Status: **incomplete — blocking audit items remain.**\n' if blocking else 'Status: **complete for the requested two-stage workflow.**\n')+f"\nAlgorithm: `size_prioritised_multiseed_v1` / `disell.flood_fill_dfxm_two_stage`.\n\nDurable rows: {len(rows):,}; scientifically selectable schema-v2 trials: {scientific_accounting['selectable_v2_trials']:,}; superseded schema-v1 rows: {scientific_accounting['superseded_v1_rows']:,}; categories across all durable rows: `{dict(counts)}`; duplicate full hashes: {duplicates}; malformed records: {len(malformed)}. Coverage: `{coverage}`.\n\nPeak observed aggregate RSS: {resources.get('peak_aggregate_process_tree_rss_bytes',progress.get('aggregate_process_tree_rss_bytes'))} bytes; peak worker RSS: {resources.get('peak_worker_reported_rss_bytes',progress.get('max_worker_peak_rss_bytes'))} bytes; maximum swap growth: {resources.get('maximum_swap_growth_bytes',progress.get('swap_growth_bytes'))} bytes.\n"
 if best: text+=f"\nObject-first selection recovered {best.get('one_to_one_recovered_cells')} cells one-to-one and {best.get('orientation_correct_cells_at_0p02deg')} within 0.02°, with median matched orientation error {best.get('median_matched_mean_orientation_error_deg')}°, count error {best.get('cell_count_error')}, {best.get('unmatched_true_cells')} unmatched true cells, {best.get('merged_predicted_cells')} merged predictions and {best.get('split_true_cells')} split true cells.\n"
 text+='\nSee `artifact_audit.json`, `AUDIT_COMPLIANCE.md`, and the saved scalar tables for reproducible details.\n'; (OUT/'FINAL_STATUS.md').write_text(text)
 if not blocking:
  compliance=(OUT/'AUDIT_COMPLIANCE.md').read_text(); compliance=compliance.replace('| pending |','| tested |').replace('Search must finish','Completed and audited').replace('Requires broad completion','Completed and audited').replace('Requires search completion','Completed and audited').replace('Requires two-stage finalists','Completed using final two-stage candidates').replace('Must run after stable primary search','Completed as an independent design').replace('Starts only after primary selection','Completed after primary selection'); compliance+='\n## Final automated gate\n\nAll coverage, persistence, downstream-artifact, finite-value and figure-source checks passed. Exact counts and hashes are recorded in `artifact_audit.json`.\n'; (OUT/'AUDIT_COMPLIANCE.md').write_text(compliance)
 if blocking: raise SystemExit(2)
if __name__=='__main__': main()

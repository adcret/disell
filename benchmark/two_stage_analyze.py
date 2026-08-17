#!/usr/bin/env python3
"""Object-first tables, selected volumes, sensitivity and guarded KAM comparison."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; OUT=HERE/'two_stage_oracle_results'; sys.path.insert(0,str(HERE))
import object_orientation_metrics as oom,oracle_core as oc,oracle_metrics as om,pipelines
from two_stage_accounting import is_selectable_v2,unique_rows
PARAMS=list(oc.PARAMETER_NAMES)

def rows():
 out=[]
 for line in (OUT/'evaluations.jsonl').read_text().splitlines():
  r=json.loads(line)
  # Saltelli is an independent variance design and must not silently become
  # an extra optimisation batch or alter the manuscript's selected solution.
  if is_selectable_v2(r) and r.get('stage')!='saltelli': out.append(r)
 return unique_rows(out,is_selectable_v2)
def cfg_key(r): return tuple(r[p] for p in PARAMS)
def aggregate(frame):
 numeric=[c for c in frame.select_dtypes('number') if c not in ('seed','candidate_order_seed')]
 records=[]
 for key,g in frame.groupby(PARAMS,dropna=False):
  row={p:v for p,v in zip(PARAMS,key)}; row['n_seeds']=len(g)
  for c in numeric:
   values=g[c].dropna()
   if len(values):
    row[c+'_mean']=values.mean(); row[c+'_sd']=values.std(ddof=1) if len(values)>1 else 0.; row[c+'_p05']=values.quantile(.05); row[c+'_worst']=values.min() if ('recovered' in c or c=='ari') else values.max()
  records.append(row)
 return pd.DataFrame(records)
def lexsort(frame):
 c=frame.copy(); c['abs_count']=c.cell_count_error_mean.abs()
 return c.sort_values(['orientation_correct_cells_at_0p02deg_mean','one_to_one_recovered_cells_mean','unmatched_true_cells_mean','unmatched_predictions_mean','merged_predicted_cells_mean','split_true_cells_mean','excess_fragments_total_mean','abs_count','median_matched_mean_orientation_error_deg_mean','cell_mean_wasserstein_deg_mean','vi_total_bits_mean','ari_mean'],ascending=[False,False,True,True,True,True,True,True,True,True,True,False])
def pareto(frame):
 work=lexsort(frame); objectives=np.column_stack((-work.orientation_correct_cells_at_0p02deg_mean,-work.one_to_one_recovered_cells_mean,work.unmatched_true_cells_mean,work.merged_predicted_cells_mean,work.split_true_cells_mean,work.abs_count,work.vi_total_bits_mean,-work.ari_mean)); keep=np.ones(len(work),bool)
 for i in range(len(work)):
  if keep[i]: keep[i]=not np.any(np.all(objectives<=objectives[i],axis=1)&np.any(objectives<objectives[i],axis=1))
 return work.iloc[np.flatnonzero(keep)]
def selections(frame):
 ranked=lexsort(frame); strong=frame[frame.one_to_one_recovered_cells_mean>=frame.one_to_one_recovered_cells_mean.quantile(.9)]
 picks={'maximum_orientation_correct_recovery':ranked.iloc[0],'maximum_total_one_to_one_recovery':frame.sort_values(['one_to_one_recovered_cells_mean','orientation_correct_cells_at_0p02deg_mean','unmatched_true_cells_mean'],ascending=[False,False,True]).iloc[0],'minimum_identity_error':frame.sort_values(['unmatched_true_cells_mean','unmatched_predictions_mean','merged_predicted_cells_mean','split_true_cells_mean']).iloc[0],'minimum_count_error_subject_to_strong_recovery':strong.iloc[strong.cell_count_error_mean.abs().argmin()],'maximum_ari':frame.loc[frame.ari_mean.idxmax()],'minimum_vi_subject_to_strong_recovery':strong.loc[strong.vi_total_bits_mean.idxmin()]}
 exact=frame[frame.cell_count_error_mean.abs()<1e-12]
 if len(exact): picks['exact_count_counterexample']=exact.sort_values(['one_to_one_recovered_cells_mean','orientation_correct_cells_at_0p02deg_mean'],ascending=[True,True]).iloc[0]
 frontier=pareto(frame); centre=np.column_stack([(-frontier.orientation_correct_cells_at_0p02deg_mean).rank(pct=True),frontier.unmatched_true_cells_mean.rank(pct=True),frontier.merged_predicted_cells_mean.rank(pct=True),frontier.split_true_cells_mean.rank(pct=True),frontier.cell_count_error_mean.abs().rank(pct=True)]).sum(axis=1); picks['balanced_object_first_pareto_knee']=frontier.iloc[int(np.argmin(centre))]
 return picks,frontier
def tolerance_solution_table(frame):
 output=[]
 for tolerance in (.005,.01,.02,.05):
  tag=str(tolerance).replace('.','p'); target=f'orientation_correct_cells_at_{tag}deg_mean'
  ordered=frame.sort_values([target,'one_to_one_recovered_cells_mean','unmatched_true_cells_mean','merged_predicted_cells_mean','split_true_cells_mean','cell_count_error_mean'],ascending=[False,False,True,True,True,True])
  row=ordered.iloc[0]; output.append({'orientation_tolerance_deg':tolerance,**{p:row[p] for p in PARAMS},target:row[target],'one_to_one_recovered_cells_mean':row.one_to_one_recovered_cells_mean,'unmatched_true_cells_mean':row.unmatched_true_cells_mean,'cell_count_error_mean':row.cell_count_error_mean})
 return pd.DataFrame(output)
def _facet_summary(truth,labels,spacing):
 facets=om.facet_table(truth,labels,spacing); recovered=float(facets['recovered_faces'].sum()); total=float(facets['faces'].sum()); low,high=om.wilson_interval(recovered,total); area=np.asarray(facets['area_um2'],float); found=np.asarray(facets['recovered_area_um2'],float); rng=np.random.default_rng(20260831); boot=[]
 if len(area):
  for _ in range(1000):
   take=rng.integers(0,len(area),len(area)); denominator=area[take].sum(); boot.append(found[take].sum()/denominator if denominator else np.nan)
 return {'true_facet_count':int(len(facets['faces'])),'true_facet_face_count':int(total),'recovered_facet_face_count':int(recovered),'true_facet_recall_face_weighted':recovered/total if total else None,'true_facet_recall_face_weighted_wilson_ci95_low':low,'true_facet_recall_face_weighted_wilson_ci95_high':high,'true_facet_recall_area_weighted_bootstrap_ci95_low':float(np.nanpercentile(boot,2.5)) if boot else None,'true_facet_recall_area_weighted_bootstrap_ci95_high':float(np.nanpercentile(boot,97.5)) if boot else None,'facet_bootstrap_replicates':1000}
def _marker_summary(w,markers):
 pairs,obj=oom.match_cells(w.labels,markers,w.field,w.spacing_um_zyx,purity_threshold=.6,completeness_threshold=.6)
 r,c,n,_,ps=om.contingency(w.labels,markers); dominant=om.dominant_map(r,c,n,int(ps.size),by='pred')
 return {**{f'marker_{k}':v for k,v in om.marker_errors(markers,w.labels).items()},**{f'marker_identity_{k}':v for k,v in obj.items()},**{f'marker_{k}':v for k,v in om.interface_precision(w.labels,markers,w.spacing_um_zyx,dominant).items()},'marker_matched_pairs':len(pairs)}
def save_selected(picks):
 w=oc.load_workspace(OUT/'../oracle_results/cache'); matched=[]; manifest={}; label_dir=OUT/'selected_labels'; label_dir.mkdir(exist_ok=True)
 for name,row in picks.items():
  cfg=oc.Config(**{p:int(row[p]) if p=='min_cell_size' else float(row[p]) for p in PARAMS}); labels,markers,initial,diag,fs=pipelines.run_flood_fill_two_stage(w.field,w.mask,w.kam(cfg.kam_radius_um),w.footprint(cfg.footprint_radius_um),local_threshold_deg=cfg.local_threshold_deg,global_threshold_deg=cfg.global_threshold_deg,footprint_tolerance=cfg.footprint_tolerance,min_cell_size=cfg.min_cell_size,max_seed_attempts=700000,stagnation_tolerance=2000,random_seed=0,watershed_connectivity=1,recycle_small_grains=False)
  np.savez_compressed(label_dir/f'{name}.npz',labels=labels,markers=markers,preliminary_sizes=initial,marker_sizes=fs)
  pairs,summary=oom.match_cells(w.labels,labels,w.field,w.spacing_um_zyx,purity_threshold=.6,completeness_threshold=.6)
  partition=om.evaluate_partition(w.labels,labels,w.spacing_um_zyx,with_boundary=True)
  for x in pairs: matched.append({'solution':name,**x})
  manifest[name]={'parameters':cfg.as_dict(),'metrics':{**summary,**partition,**_facet_summary(w.labels,labels,w.spacing_um_zyx)},'marker_metrics':_marker_summary(w,markers),'candidate_diagnostics':diag,'preliminary_candidate_count':int(len(initial)),'candidate_region_identity_available':False}
 matched_frame=pd.DataFrame(matched); matched_frame.to_csv(OUT/'matched_orientation_table.csv',index=False); matched_frame.to_csv(OUT/'object_correspondence_table.csv',index=False); correlations=[]
 for name,g in matched_frame.groupby('solution'):
  for channel in ('chi','phi'):
   truth=g[f'true_mean_{channel}_deg']; predicted=g[f'predicted_mean_{channel}_deg']; slope,intercept=np.polyfit(truth,predicted,1); correlations.append({'solution':name,'channel':channel,'pearson_r':truth.corr(predicted),'calibration_slope':slope,'calibration_intercept_deg':intercept,'n_cells':len(g)})
 pd.DataFrame(correlations).to_csv(OUT/'matched_mean_correlation_calibration.csv',index=False); (OUT/'selected_solutions.json').write_text(json.dumps(manifest,indent=2,default=lambda x:x.item() if hasattr(x,'item') else x))
 stages=[]
 for name,value in manifest.items():
  stages.append({'solution':name,'stage':'preliminary_candidate_discovery','region_count':value['preliminary_candidate_count'],'spatial_region_metrics_available':False})
  stages.append({'solution':name,'stage':'size_prioritised_markers','region_count':value['marker_metrics'].get('marker_marker_count'),**value['marker_metrics']})
  stages.append({'solution':name,'stage':'kam_guided_watershed','region_count':value['metrics'].get('n_cells_pred'),**value['metrics']})
 pd.DataFrame(stages).to_csv(OUT/'candidate_marker_watershed_comparison.csv',index=False)
 return w,matched_frame,manifest
def sensitivity(frame):
 responses=['one_to_one_recovered_cells','orientation_correct_cells_at_0p02deg','unmatched_true_cells','split_true_cells','merged_predicted_cells','cell_count_error','median_matched_mean_orientation_error_deg','vi_total_bits','ari','boundary_f1_at_0p4um']
 rows=[]
 for p in PARAMS:
  bins=pd.qcut(frame[p],min(12,frame[p].nunique()),duplicates='drop')
  for interval,g in frame.groupby(bins,observed=True):
   for y in responses:
    if y in g: rows.append({'parameter':p,'low':interval.left,'high':interval.right,'response':y,'count':len(g),'mean':g[y].mean(),'sd':g[y].std(),'se':g[y].std()/np.sqrt(len(g))})
 profiles=pd.DataFrame(rows); profiles.to_csv(OUT/'sensitivity_profiles.csv',index=False)
 surfaces=[]
 for a,b in [('local_threshold_deg','global_threshold_deg'),('footprint_tolerance','min_cell_size'),('footprint_radius_um','kam_radius_um')]:
  x=frame.copy(); x['a_bin']=pd.qcut(x[a],8,duplicates='drop'); x['b_bin']=pd.qcut(x[b],8,duplicates='drop')
  for (ai,bi),g in x.groupby(['a_bin','b_bin'],observed=True):
   surfaces.append({'parameter_a':a,'parameter_b':b,'a_low':ai.left,'a_high':ai.right,'b_low':bi.left,'b_high':bi.right,'count':len(g),'orientation_correct_recovery_mean':g.orientation_correct_cells_at_0p02deg.mean(),'one_to_one_recovery_mean':g.one_to_one_recovered_cells.mean(),'ari_mean':g.ari.mean(),'vi_mean':g.vi_total_bits.mean()})
 surface_frame=pd.DataFrame(surfaces); surface_frame.to_csv(OUT/'pairwise_response_surfaces.csv',index=False)
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
 for axis,parameter in zip(axes.ravel(),PARAMS):
  for response,label in [('orientation_correct_cells_at_0p02deg','orientation-correct'),('one_to_one_recovered_cells','one-to-one')]:
   g=profiles[(profiles.parameter==parameter)&(profiles.response==response)]; centre=(g.low+g.high)/2; axis.errorbar(centre,g['mean'],yerr=1.96*g.se,marker='o',ms=2,lw=.7,label=label)
  axis.set(xlabel=parameter,ylabel='recovered cells'); axis.text(.02,.02,f"n={int(frame[parameter].notna().sum())}",transform=axis.transAxes,fontsize=6)
 axes[0,0].legend(fontsize=6)
 for ext in ('png','pdf'): fig.savefig(OUT/f'object_first_sensitivity_profiles.{ext}',dpi=300 if ext=='png' else None)
 plt.close(fig)
 if not surface_frame.empty:
  fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
  for axis,((a,b),g) in zip(axes,surface_frame.groupby(['parameter_a','parameter_b'])):
   points=axis.scatter((g.a_low+g.a_high)/2,(g.b_low+g.b_high)/2,c=g.orientation_correct_recovery_mean,s=np.maximum(g['count'],1),cmap='viridis'); axis.set(xlabel=a,ylabel=b); fig.colorbar(points,ax=axis,label='orientation-correct cells')
  for ext in ('png','pdf'): fig.savefig(OUT/f'pairwise_object_response_surfaces.{ext}',dpi=300 if ext=='png' else None)
  plt.close(fig)
def surrogate_analysis(frame):
 from sklearn.ensemble import ExtraTreesRegressor
 from sklearn.inspection import permutation_importance
 from sklearn.metrics import r2_score,mean_absolute_error
 from sklearn.model_selection import train_test_split
 x=frame[PARAMS].copy(); x['global_threshold_active']=(x.global_threshold_deg>0).astype(int); features=list(x.columns)
 rows=[]; importance=[]
 for response,domain in [('orientation_correct_cells_at_0p02deg','object_identification'),('median_matched_mean_orientation_error_deg','mean_orientation_accuracy'),('ari','voxel_partition_accuracy'),('vi_total_bits','voxel_partition_accuracy'),('boundary_f1_at_0p4um','boundary_placement')]:
  if response not in frame or frame[response].notna().sum()<100: continue
  valid=frame[response].notna(); train,test=train_test_split(np.flatnonzero(valid),test_size=.2,random_state=20260825); model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,n_jobs=1,random_state=20260825).fit(x.iloc[train],frame[response].iloc[train]); prediction=model.predict(x.iloc[test]); r2=r2_score(frame[response].iloc[test],prediction); mae=mean_absolute_error(frame[response].iloc[test],prediction); adequate=bool(r2>=.5)
  rows.append({'response':response,'domain':domain,'held_out_n':len(test),'held_out_r2':r2,'held_out_mae':mae,'importance_interpretable':adequate})
  if adequate:
   result=permutation_importance(model,x.iloc[test],frame[response].iloc[test],n_repeats=8,n_jobs=1,random_state=20260825)
   for name,mean,sd in zip(features,result.importances_mean,result.importances_std): importance.append({'response':response,'domain':domain,'parameter':name,'permutation_importance_mean':mean,'permutation_importance_sd':sd,'held_out_r2':r2})
 pd.DataFrame(rows).to_csv(OUT/'surrogate_validation.csv',index=False); imp=pd.DataFrame(importance); imp.to_csv(OUT/'surrogate_importance.csv',index=False)
 classifications=[]
 for domain in ('object_identification','mean_orientation_accuracy','voxel_partition_accuracy','boundary_placement'):
  subset=imp[imp.domain==domain] if len(imp) else pd.DataFrame()
  for parameter in PARAMS:
   values=subset[subset.parameter==parameter].permutation_importance_mean if len(subset) else pd.Series(dtype=float); score=float(values.mean()) if len(values) else None
   if parameter=='global_threshold_deg': label='interacting'
   elif score is None: label='unclassified (surrogate inadequate)'
   elif score<.005: label='inactive'
   elif score<.03: label='plateaued'
   else: label='constrained'
   classifications.append({'domain':domain,'parameter':parameter,'classification':label,'mean_permutation_importance':score})
 pd.DataFrame(classifications).to_csv(OUT/'parameter_classification.csv',index=False)
def kam_guarded(w):
 out=[]; d=HERE/'continuation_results'/'kam_analysis'
 parity=json.loads((d/'input_parity_audit.json').read_text()); audit=json.loads((OUT/'implementation_audit.json').read_text())
 parity_ok=(parity['primary_cache']['latent']['sha256']==audit['latent_sha256'] and parity['primary_cache']['labels']['sha256']==audit['truth_sha256'] and tuple(w.spacing_um_zyx)==tuple(audit['spacing_um_zyx']))
 for path in sorted(d.glob('kam_*.npz')):
  with np.load(path) as z: labels=z['labels']
  _,obj=oom.match_cells(w.labels,labels,w.field,w.spacing_um_zyx,purity_threshold=.6,completeness_threshold=.6); metric=om.evaluate_partition(w.labels,labels,w.spacing_um_zyx,with_boundary=True); facet=_facet_summary(w.labels,labels,w.spacing_um_zyx); true_area=float(om.facet_table(w.labels,w.labels,w.spacing_um_zyx)['area_um2'].sum()); out.append({'solution':path.stem,'input_parity_confirmed':parity_ok,**obj,**metric,**facet,'true_interface_area_um2':true_area,'count_guard_300_420':300<=labels.max()<=420,'interface_area_guard_10pct_true':abs(metric['predicted_interface_area_um2']/true_area-1)<=.1,'high_interface_precision':metric['interface_precision']>=.8,'scientifically_guarded':300<=labels.max()<=420 and abs(metric['predicted_interface_area_um2']/true_area-1)<=.1 and metric['interface_precision']>=.8})
 pd.DataFrame(out).to_csv(OUT/'kam_guarded_comparison.csv',index=False)
 (OUT/'kam_input_parity.json').write_text(json.dumps({'confirmed':parity_ok,'latent_hash':audit['latent_sha256'],'truth_hash':audit['truth_sha256'],'spacing_um_zyx':audit['spacing_um_zyx'],'kam_definition':'per-channel RMS on the identical latent field and physical footprint'},indent=2))
 return pd.DataFrame(out)
def orientation_tolerance_tables(w,manifest):
 rows=[]
 identity=[]
 for name in manifest:
  with np.load(OUT/'selected_labels'/f'{name}.npz') as z: labels=z['labels']
  for purity in (.5,.6,.7):
   for completeness in (.5,.6,.7):
    _,summary=oom.match_cells(w.labels,labels,w.field,w.spacing_um_zyx,purity_threshold=purity,completeness_threshold=completeness)
    identity.append({'solution':name,**summary})
  pairs,_=oom.match_cells(w.labels,labels,w.field,w.spacing_um_zyx,purity_threshold=.6,completeness_threshold=.6); errors=np.asarray([pair['mean_orientation_error_deg'] for pair in pairs])
  for tolerance in np.r_[np.linspace(0,.05,51),.075,.1]: rows.append({'solution':name,'orientation_tolerance_deg':float(tolerance),'orientation_correct_cells':int(np.sum(errors<=tolerance)),'orientation_correct_recovered_fraction':float(np.sum(errors<=tolerance)/max(int(w.labels.max()),1)),'equal_cell_weighting':True})
 pd.DataFrame(identity).to_csv(OUT/'identity_tolerance_decision_table.csv',index=False); curve=pd.DataFrame(rows); curve.to_csv(OUT/'orientation_recovery_curve.csv',index=False); return curve
def figures(frame,matched,curve,kam):
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(8,3.2),layout='constrained'); ax[0].scatter(frame.cell_count_error,frame.one_to_one_recovered_cells,s=5,alpha=.35); ax[0].axvline(0,color='k',lw=.6); ax[0].set(xlabel='cell-count error',ylabel='one-to-one recovered cells'); ax[1].scatter(frame.ari,frame.orientation_correct_cells_at_0p02deg,s=5,alpha=.35); ax[1].set(xlabel='ARI',ylabel='orientation-correct cells (0.02°)')
 for ext in ('png','pdf'): fig.savefig(OUT/f'count_identity_and_ari_recovery.{ext}',dpi=300 if ext=='png' else None)
 plt.close(fig)
 if not matched.empty:
  fig,ax=plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
  for axis,channel in zip(ax,('chi','phi')):
   x=matched[f'true_mean_{channel}_deg']; y=matched[f'predicted_mean_{channel}_deg']; axis.scatter(x,y,s=7,alpha=.45); lo=min(x.min(),y.min()); hi=max(x.max(),y.max()); axis.plot([lo,hi],[lo,hi],color='k',lw=.7); axis.set(xlabel=f'true mean {channel} (deg)',ylabel=f'predicted mean {channel} (deg)',aspect='equal')
  for ext in ('png','pdf'): fig.savefig(OUT/f'matched_cell_mean_calibration.{ext}',dpi=300 if ext=='png' else None)
  plt.close(fig)
  fig,ax=plt.subplots(1,2,figsize=(8,3.2),layout='constrained'); ax[0].hist(matched.mean_orientation_error_deg,bins=40); ax[0].set(xlabel='matched mean-orientation error (deg)',ylabel='cells'); ax[1].scatter(matched.prediction_purity,matched.true_cell_completeness,s=7,alpha=.4); ax[1].set(xlabel='prediction purity',ylabel='true-cell completeness',xlim=(0,1.01),ylim=(0,1.01))
  for ext in ('png','pdf'): fig.savefig(OUT/f'orientation_error_and_identity_quality.{ext}',dpi=300 if ext=='png' else None)
  plt.close(fig)
 if not curve.empty:
  fig,ax=plt.subplots(figsize=(4.5,3.4),layout='constrained')
  for name,g in curve.groupby('solution'): ax.plot(g.orientation_tolerance_deg,g.orientation_correct_recovered_fraction,label=name)
  ax.set(xlabel='orientation tolerance (deg)',ylabel='orientation-correct recovery'); ax.legend(fontsize=5)
  for ext in ('png','pdf'): fig.savefig(OUT/f'orientation_recovery_curve.{ext}',dpi=300 if ext=='png' else None)
  plt.close(fig)
 if not kam.empty:
  fig,ax=plt.subplots(figsize=(5,3.5),layout='constrained'); guarded=kam[kam.scientifically_guarded]; ax.scatter(kam.one_to_one_recovered_cells,kam.orientation_correct_cells_at_0p02deg,c=np.where(kam.scientifically_guarded,1,0),cmap='coolwarm',s=28); ax.set(xlabel='one-to-one recovered cells',ylabel='orientation-correct cells (0.02°)');
  for _,r in guarded.iterrows(): ax.annotate(r.solution.replace('kam_',''),(r.one_to_one_recovered_cells,r.orientation_correct_cells_at_0p02deg),fontsize=5)
  for ext in ('png','pdf'): fig.savefig(OUT/f'kam_guarded_identity_orientation.{ext}',dpi=300 if ext=='png' else None)
  plt.close(fig)
def synthetic_identity_examples():
 import matplotlib.pyplot as plt
 truth=np.zeros((40,60),np.int32); truth[:,:20]=1; truth[:,20:40]=2; truth[:,40:]=3
 shifted=np.zeros_like(truth); shifted[:,:18]=1; shifted[:,18:42]=2; shifted[:,42:]=3
 compensating=np.zeros_like(truth); compensating[:20,:20]=1; compensating[20:,:20]=2; compensating[:,20:]=3
 field=np.zeros(truth.shape+(2,),np.float32)
 for label,values in enumerate(((0,0),(.01,.02),(.04,.01),(.02,.06))): field[truth==label]=values
 records=[]
 for name,labels in [('shifted_boundaries_correct_identity',shifted),('compensating_split_merge_exact_count',compensating)]:
  _,obj=oom.match_cells(truth,labels,field,(.4,.4),purity_threshold=.5,completeness_threshold=.5); records.append({'example':name,**obj,**om.evaluate_partition(truth,labels,(.4,.4),with_boundary=True)})
 pd.DataFrame(records).to_csv(OUT/'synthetic_identity_counterexamples.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(7,4.8),layout='constrained')
 for row,(name,prediction) in enumerate((('shifted boundaries',shifted),('exact count: split + merge',compensating))):
  axes[row,0].imshow(truth,cmap='tab20',interpolation='nearest'); axes[row,0].set_title('truth'); axes[row,1].imshow(prediction,cmap='tab20',interpolation='nearest'); axes[row,1].set_title(name)
  for axis in axes[row]: axis.set_xticks([]); axis.set_yticks([])
 for ext in ('png','pdf'): fig.savefig(OUT/f'identity_boundary_counterexamples.{ext}',dpi=300 if ext=='png' else None)
 plt.close(fig)
def main():
 from two_stage_v2_completion import complete
 complete(workers=8)
 raw=[json.loads(line) for line in (OUT/'evaluations.jsonl').read_text().splitlines() if line.strip()]
 v2=[r for r in raw if r.get('schema_version')==2 and r.get('scientific_metrics_version')==2]
 pd.DataFrame([{'status_category':key,'count':value} for key,value in pd.Series([r.get('status_category') for r in v2]).value_counts().items()]).to_csv(OUT/'error_category_table.csv',index=False)
 pd.DataFrame([{'schema_version':r.get('schema_version'),'scientific_metrics_version':r.get('scientific_metrics_version'),'status_category':r.get('status_category')} for r in raw]).value_counts(dropna=False).rename('count').reset_index().to_csv(OUT/'error_category_table_all_durable_rows.csv',index=False)
 frame=pd.DataFrame(rows())
 if frame.empty: raise RuntimeError('no completed schema-v2 scientific rows')
 frame.to_csv(OUT/'full_scalar_trials_v2.csv',index=False); agg=aggregate(frame); picks,front=selections(agg); decision=lexsort(agg); decision.to_csv(OUT/'selection_decision_table.csv',index=False); front.to_csv(OUT/'object_first_pareto.csv',index=False); tolerance_solution_table(agg).to_csv(OUT/'tolerance_dependent_selection.csv',index=False)
 w,matched,manifest=save_selected(picks); curve=orientation_tolerance_tables(w,manifest); sensitivity(frame); surrogate_analysis(frame); kam=kam_guarded(w); figures(frame,matched,curve,kam); synthetic_identity_examples()
 text='The latent field was deliberately used to isolate identification behaviour from acquisition noise. Its piecewise-cell construction with smooth intradomain variation is favourable to region-based methods, so this is a controlled algorithmic benchmark rather than an unbiased test across all experimental fields.'
 best=manifest['maximum_orientation_correct_recovery']['metrics']; report={'algorithm_id':'size_prioritised_multiseed_v1','scientific_metrics_version':2,'n_complete_trials':len(frame),'n_unique_parameter_sets':len(agg),'selected_solutions':manifest,'object_first_best':best,'input_scope':text}; (OUT/'object_orientation_report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else x))
 result=f"# Results\n\nThe object-first analysis used {len(frame):,} completed schema-v2 trials and retained {len(front):,} non-dominated parameter sets. At the 0.02° tolerance the orientation-optimal selected solution uniquely recovered {best['one_to_one_recovered_cells']} true cells, of which {best['orientation_correct_cells_at_0p02deg']} reproduced the matched cell mean within tolerance. It left {best['unmatched_true_cells']} true cells unmatched and produced {best['merged_predicted_cells']} merged predictions and {best['split_true_cells']} substantially split true cells. Its median matched two-channel mean-orientation error was {best['median_matched_mean_orientation_error_deg']:.6g}°.\n\nExact count and marginal orientation-distribution agreement were retained only as secondary diagnostics because both can be satisfied by compensating identity errors. Boundary displacement was evaluated after identity and mean-orientation recovery.\n"
 discussion="# Discussion\n\n"+text+" The paired and guarded KAM tables should be interpreted on this controlled field only; connectivity of a tessellated high-KAM wall network is not itself evidence of KAM failure. Preliminary candidate-region overlap cannot be reconstructed from the observation-only API, which returns candidate sizes and counters but not a candidate label volume. Marker and watershed identity metrics are reported without inventing unavailable candidate-region data.\n"
 methods="# Methods\n\n"+text+" The definitive marker algorithm was `disell.flood_fill_dfxm_two_stage` (`size_prioritised_multiseed_v1`), followed exactly once by KAM-guided watershed. Identity assignment required reciprocal spatial overlap at explicit purity/completeness thresholds; wrapped orientation error was used only after spatial eligibility and could not match unrelated cells.\n"
 (OUT/'SCIENTIFIC_SUMMARY.md').write_text('# Scientific summary\n\n'+text+'\n\nThe selection is lexicographic and object-first; see `selection_decision_table.csv` and `tolerance_dependent_selection.csv`.\n'); (OUT/'RESULTS.md').write_text(result); (OUT/'DISCUSSION.md').write_text(discussion); (OUT/'METHODS.md').write_text(methods); (OUT/'FIGURE_CAPTIONS.md').write_text('# Figure captions\n\n`matched_cell_mean_calibration`: Matched predicted versus true cell-mean χ and φ. `orientation_error_and_identity_quality`: equal-cell-weighted orientation errors and reciprocal-overlap quality. `orientation_recovery_curve`: recovered true-cell fraction versus explicit orientation tolerance. `count_identity_and_ari_recovery`: count error and ARI contrasted with cell-level recovery. `kam_guarded_identity_orientation`: KAM points coloured by count, interface-area and interface-precision guards.\n')
if __name__=='__main__': main()

import numpy as np
import json
import hashlib

import oracle_core as oc
import two_stage_oracle as tso
from two_stage_store import ResultStore
from disell import flood_fill_dfxm_two_stage


def inputs():
    field=np.zeros((3,12,12,2),np.float32); field[:,:,6:]=.3
    mask=np.ones(field.shape[:3],np.uint8); footprint=np.ones((3,3,3),bool)
    return field,mask,footprint


def test_two_stage_multiple_markers_reproducible_largest_first_and_mask_safe():
    field,mask,footprint=inputs(); before=mask.copy()
    kwargs=dict(footprint=footprint,local_misorientation_threshold=.05,
      footprint_tolerance=.5,mask=mask,max_iterations=1000,min_grain_size=5,
      recycle_small_grains=False,stagnation_tolerance=200,random_seed=7)
    a,sa=flood_fill_dfxm_two_stage(field,**kwargs); b,sb=flood_fill_dfxm_two_stage(field,**kwargs)
    assert a['segmentation'].max()>=2
    np.testing.assert_array_equal(a['segmentation'],b['segmentation'])
    np.testing.assert_array_equal(sa,sb); np.testing.assert_array_equal(mask,before)
    d=a['diagnostics']; np.testing.assert_array_equal(d['seeds_sorted'],d['seeds_initial'][np.argsort(sa)])
    assert d['final_growth']['user_seeds_processed']+d['final_growth']['user_seeds_skipped_claimed']==len(sa)


def test_below_minimum_candidates_do_not_become_markers():
    field,mask,footprint=inputs()
    result,sizes=flood_fill_dfxm_two_stage(field,footprint=footprint,
      local_misorientation_threshold=.05,footprint_tolerance=.5,mask=mask,
      max_iterations=1000,min_grain_size=field[...,0].size+1,random_seed=3)
    assert len(sizes)==0 and result['segmentation'].max()==0


def test_algorithm_identity_prevents_random_order_cache_hit(monkeypatch):
    monkeypatch.setattr(tso,'IDENTITY',{'python_source_sha256':'p','compiled_extension_sha256':'e',
      'latent_sha256':'l','truth_sha256':'t','mask_sha256':'m',
      'matching_source_sha256':'metrics'})
    p=oc.Config(.01,-1,.2,1,50,1.2).as_dict(); key,_=tso.config_key(p,0)
    random_key=oc.Config(**p).key()
    assert key!=random_key and tso.ALGORITHM in tso.config_key(p,0)[1]['algorithm']


def test_policy_namespace_is_isolated_and_manifest_is_versioned(tmp_path,monkeypatch):
    frozen=tmp_path/'frozen'; destination=tmp_path/'policy'; frozen.mkdir()
    source={key:key for key in ('python_source_sha256','compiled_extension_sha256',
      'latent_sha256','truth_sha256','mask_sha256','matching_source_sha256')}
    (frozen/'implementation_audit.json').write_text(json.dumps(source))
    audit={'source_jsonl':str(frozen/'evaluations.jsonl'),'source_jsonl_sha256':'frozen-sha',
      'source_jsonl_bytes':123,'source_total_records':9159,'broad_records':8000,
      'selectable_broad_records':6930,'expected_algorithmic_invalid_broad_records':1070,
      'excluded_non_broad_records':1159}
    monkeypatch.setattr(tso,'FROZEN_BROAD_OUT',frozen); monkeypatch.setattr(tso,'OUT',destination)
    tso.initialise_policy_namespace(audit)
    manifest=json.loads((destination/'implementation_audit.json').read_text())
    assert manifest['selection_policy_id']==tso.SELECTION_POLICY_ID
    assert manifest['matching_metrics_version']==tso.SCIENTIFIC_METRICS_VERSION
    assert manifest['scientific_metrics_version']==tso.SCIENTIFIC_METRICS_VERSION
    assert manifest['result_schema_version']==4
    assert manifest['broad_evidence_mode']=='external_frozen_read_only'
    assert manifest['broad_rows_copied'] is False
    assert manifest['old_policy_adaptive_rows_imported'] is False
    assert manifest['frozen_broad_source_sha256']=='frozen-sha'
    assert not (destination/'evaluations.jsonl').exists()


def test_policy_namespace_refuses_legacy_destination(tmp_path,monkeypatch):
    monkeypatch.setattr(tso,'FROZEN_BROAD_OUT',tmp_path); monkeypatch.setattr(tso,'OUT',tmp_path)
    with np.testing.assert_raises(RuntimeError):
        tso.initialise_policy_namespace({})


def test_executor_level_worker_failure_gets_complete_envelope(monkeypatch):
    monkeypatch.setattr(tso,'IDENTITY',{'python_source_sha256':'p','compiled_extension_sha256':'e',
      'latent_sha256':'l','truth_sha256':'t','mask_sha256':'m',
      'matching_source_sha256':'metrics'})
    p=oc.Config(.01,-1,.2,1,50,1.2).as_dict()
    row=tso.worker_future_failure({'parameters':p,'seed':9,'stage':'forced'},RuntimeError('pool lost'))
    assert row['status_category']=='worker_exception'
    assert row['worker_peak_rss_bytes'] is None
    assert row['seed']==9 and row['config_hash']


def test_coordinator_prepares_nonfinite_worker_result_without_spreading_raw(monkeypatch):
    monkeypatch.setattr(tso,'IDENTITY',{'python_source_sha256':'p','compiled_extension_sha256':'e',
      'latent_sha256':'l','truth_sha256':'t','mask_sha256':'m',
      'matching_source_sha256':'metrics'})
    p=oc.Config(.01,-1,.2,1,50,1.2).as_dict(); expected,identity=tso.config_key(p,4)
    raw={
      'config_hash':expected,'configuration_identity':identity,
      'config_key':oc.Config(**p).key(),'algorithm':tso.ALGORITHM,
      'candidate_order_seed':4,'stage':'forced_nonfinite','status':'ok',
      'scientific_metrics_version':2,'scientific_metrics':{'boundary_f1':np.nan},
      'worker_pid':1,'worker_peak_rss_bytes':2,'started':'a','finished':'b',
      'elapsed_seconds':.1,'candidate_pass_saturated':False,
      'final_pass_saturated':False,
    }
    row=tso.prepare_worker_result(raw,{'parameters':p,'seed':4},expected,identity)
    assert row['status_category']=='coordinator/schema_error'
    assert row['error_type']=='NonFiniteResult'
    assert row['nonfinite_fields']==['result.scientific_metrics.boundary_f1']
    assert 'scientific_metrics' not in row


def ranking_row(name, *, recovered=300, orientation=290, n_true=360,
                n_pred=360, median=.001, **updates):
    row={
      'config_hash':name,'status_category':'ok',
      'scientific_metrics_version':tso.SCIENTIFIC_METRICS_VERSION,
      'selection_policy_id':tso.SELECTION_POLICY_ID,
      'n_cells_true':n_true,'n_cells_pred':n_pred,
      'orientation_correct_cells_at_0p005deg':min(orientation,recovered),
      'orientation_correct_cells_at_0p01deg':min(orientation,recovered),
      'orientation_correct_cells_at_0p02deg':orientation,
      'orientation_correct_cells_at_0p05deg':min(recovered,orientation+5),
      'one_to_one_recovered_cells':recovered,'unmatched_true_cells':n_true-recovered,
      'unmatched_predictions':n_pred-recovered,'merged_predicted_cells':2,
      'split_true_cells':3,'excess_fragments_total':4,
      'cell_count_error':n_pred-n_true,
      'median_matched_mean_orientation_error_deg':median,
      'cell_mean_wasserstein_deg':.002,'vi_total_bits':.4,'ari':.9,
      'panoptic_quality_at_0p5':.8,'matched_iou_mean':.85,
      'true_facet_recall_area_weighted':.9,'interface_precision':.8,
      'boundary_assd_um':.2,'candidate_order_seed':0,
      **oc.Config(.01,-1,.2,1,50,1.2).as_dict(),
    }
    row.update(updates)
    row.update(tso.selection_metrics(row))
    return row


def test_perfect_segmentation_has_unit_identity_and_orientation_f1():
    row=ranking_row('perfect',recovered=360,orientation=360,n_pred=360)
    for tag in ('0p005','0p01','0p02','0p05'):
        assert row[f'orientation_correct_precision_at_{tag}deg']==1.0
        assert row[f'orientation_correct_recall_at_{tag}deg']==1.0
        assert row[f'orientation_correct_f1_at_{tag}deg']==1.0
    assert row['identity_precision']==row['identity_recall']==row['identity_f1']==1.0


def test_zero_recovery_has_finite_zero_object_rates():
    row=ranking_row('zero',recovered=0,orientation=0,n_pred=0,median=None)
    names=['identity_precision','identity_recall','identity_f1']
    names += [f'orientation_correct_{stat}_at_{tag}deg'
              for tag in ('0p005','0p01','0p02','0p05')
              for stat in ('precision','recall','f1')]
    assert all(row[name] == 0.0 for name in names)
    assert all(np.isfinite(row[name]) for name in names)


def test_symmetric_count_penalty_and_zero_count_handling():
    assert tso.count_score(360,320)==tso.count_score(360,405)
    assert tso.count_score(320,360)==tso.count_score(405,360)
    assert tso.count_score(0,360)==tso.count_score(360,0)==tso.count_score(0,0)==0.0
    assert tso.count_score(358,360)==tso.count_score(360,358)


def test_oversegmentation_and_undersegmentation_have_equal_orientation_count_penalty():
    under=ranking_row('under',orientation=280,n_pred=320)
    over=ranking_row('over',orientation=280,n_pred=405)
    # Fix the source F1 so this isolates the symmetric count component.
    for row in (under,over):
        row['orientation_correct_f1_at_0p02deg']=.8
        row.update(tso.selection_metrics(row))
    assert under['count_score']==over['count_score']
    assert under['orientation_count_score_at_0p02deg']==over['orientation_count_score_at_0p02deg']


def test_extra_predictions_and_missing_truth_recovery_reduce_f1():
    baseline=ranking_row('baseline',recovered=280,orientation=280,n_pred=360)
    extras=ranking_row('extras',recovered=280,orientation=280,n_pred=434)
    missing=ranking_row('missing',recovered=260,orientation=260,n_pred=360)
    assert extras['orientation_correct_f1_at_0p02deg'] < baseline['orientation_correct_f1_at_0p02deg']
    assert extras['identity_f1'] < baseline['identity_f1']
    assert missing['orientation_correct_f1_at_0p02deg'] < baseline['orientation_correct_f1_at_0p02deg']
    assert missing['identity_f1'] < baseline['identity_f1']


def test_every_derived_rate_is_finite_and_bounded():
    for row in (ranking_row('ordinary'),ranking_row('zero',recovered=0,orientation=0,n_pred=0)):
        derived=tso.selection_metrics(row)
        for key,value in derived.items():
            if key=='count_score' or key.startswith('orientation_count_score_at_') or key.endswith(('_precision','_recall','_f1')) or '_precision_at_' in key or '_recall_at_' in key or '_f1_at_' in key:
                if value is not None:
                    assert np.isfinite(value) and 0.0 <= value <= 1.0


def test_nonfinite_or_out_of_bounds_orientation_input_is_rejected():
    for value in (np.nan,np.inf,-.1,1.1):
        row=ranking_row('invalid-derived')
        row['orientation_correct_f1_at_0p02deg']=value
        with np.testing.assert_raises(ValueError):
            tso.selection_metrics(row)


def test_358_cell_scientific_example_beats_351_under_revised_policy():
    balanced=ranking_row('balanced-358',recovered=295,orientation=280,n_pred=358,
                         matched_iou_mean=.829005,ari=.869812)
    under=ranking_row('under-351',recovered=287,orientation=280,n_pred=351,
                      matched_iou_mean=.818778,ari=.843018)
    assert under['orientation_correct_f1_at_0p02deg'] > balanced['orientation_correct_f1_at_0p02deg']
    assert balanced['count_score'] > under['count_score']
    assert balanced['orientation_count_score_at_0p02deg'] > under['orientation_count_score_at_0p02deg']
    assert tso.rank([under,balanced])[0]['config_hash']=='balanced-358'


def test_seed_aggregated_ranking_uses_median_then_minimum_for_few_seeds():
    one=oc.Config(.01,-1,.2,1,50,1.2).as_dict()
    two=oc.Config(.02,-1,.2,1,50,1.2).as_dict()
    rows=[]
    for seed,value in enumerate((275,200,275)):
        rows.append(ranking_row(f'one-{seed}',orientation=value,
                    recovered=max(value,300),candidate_order_seed=seed,**one))
    for seed,value in enumerate((275,275,275)):
        rows.append(ranking_row(f'two-{seed}',orientation=value,
                    recovered=300,candidate_order_seed=seed,**two))
    ranked=tso.rank_configurations(rows)
    assert len(ranked)==2
    assert ranked[0]['local_threshold_deg']==.02
    assert ranked[0]['_seed_summary']['lower_tail_statistic']=='minimum'
    assert ranked[0]['_seed_summary']['orientation_count_score_at_0p02deg_p05'] is None


def test_pareto_regions_keep_identity_and_boundary_operating_points():
    identity=ranking_row('identity',orientation=300,recovered=310,boundary_assd_um=.4)
    boundary=ranking_row('boundary',orientation=290,recovered=300,boundary_assd_um=.05)
    dominated=ranking_row('dominated',orientation=280,recovered=290,boundary_assd_um=.5,
                          panoptic_quality_at_0p5=.7,matched_iou_mean=.7,
                          true_facet_recall_area_weighted=.7,interface_precision=.7)
    frontier=tso.pareto_rows([identity,boundary,dominated])
    assert identity in frontier and boundary in frontier
    assert dominated not in frontier


def test_pareto_vector_contains_all_required_policy_objectives():
    row=ranking_row('vector')
    vector=tso.object_vector(row)
    assert vector.shape==(7,)
    np.testing.assert_allclose(vector,[
      -row['orientation_count_score_at_0p02deg'],-row['count_score'],
      -row['identity_f1'],-row['panoptic_quality_at_0p5'],
      -row['matched_iou_mean'],-row['interface_f1'],row['boundary_assd_um'],
    ])


def test_diverse_pool_combines_all_sources_and_exact_count_does_not_dominate():
    exact=ranking_row('exact-but-poor',orientation=100,recovered=100,n_pred=360,local_threshold_deg=.011)
    balanced=ranking_row('balanced',orientation=280,recovered=295,n_pred=358,local_threshold_deg=.012)
    identity=ranking_row('identity',orientation=270,recovered=320,n_pred=340,local_threshold_deg=.013)
    boundary=ranking_row('boundary',orientation=260,recovered=280,n_pred=366,boundary_assd_um=.01,local_threshold_deg=.014)
    rows=[exact,balanced,identity,boundary]
    pool,sources=tso.diverse_pool(rows,4,include_full_frontier=True)
    assert set(sources)=={'official','count_focused','identity_focused','boundary_focused','full_non_dominated'}
    assert sources['count_focused'][0]['config_hash']!='exact-but-poor'
    assert {row['config_hash'] for row in tso.pareto_rows(rows)} <= {row['config_hash'] for row in pool}


def test_round_none_crash_fix_remains_independent_of_policy_ranking():
    row=ranking_row('zero',recovered=0,orientation=0,median=None)
    assert np.isposinf(tso.minimised_metric(row,'median_matched_mean_orientation_error_deg',5))
    assert row['median_matched_mean_orientation_error_deg'] is None


def test_rank_is_total_for_zero_recovery_and_undefined_boundary():
    a=ranking_row('a',recovered=0,orientation=0,median=None,
                  boundary_assd_um=None,true_facet_recall_area_weighted=None,
                  interface_precision=None)
    b=ranking_row('b',recovered=0,orientation=0,median=None,
                  boundary_assd_um=None,true_facet_recall_area_weighted=None,
                  interface_precision=None)
    defined=ranking_row('defined',recovered=0,orientation=0,median=None,
                        boundary_assd_um=.5,true_facet_recall_area_weighted=.2,
                        interface_precision=.2)
    assert [row['config_hash'] for row in tso.rank([b,a])] == ['a','b']
    assert tso.rank([a,defined])[0]['config_hash']=='defined'


def test_malformed_required_metric_is_not_scientifically_selectable():
    row=ranking_row('malformed',panoptic_quality_at_0p5='not-a-number')
    assert tso.selectable([row])==[]
    assert tso.rank([row])[0] is row


def test_live_accounting_separates_durable_and_selectable_v3_rows():
    good={**ranking_row('good'),'config_key':'good-key'}
    invalid={**ranking_row('invalid'),'config_key':'invalid-key',
             'status_category':'expected_algorithmic_invalid'}
    old={**ranking_row('old'),'config_key':'old-key','scientific_metrics_version':2,
         'selection_policy_id':None}
    counts=tso.live_accounting([good,invalid,old],planned=8_000)
    assert counts=={
      'durable_rows':3,'superseded_pre_v3_rows':1,'unique_full_hashes':3,
      'unique_trials':3,'selectable_v3_trials':1,'planned_unique_v3_trials':8_000,
    }


def test_pareto_mixture_with_zero_recovery_and_undefined_boundary_is_safe():
    normal=ranking_row('normal')
    zero=ranking_row('zero',recovered=0,orientation=0,median=None,
                     boundary_assd_um=None,true_facet_recall_area_weighted=None,
                     interface_precision=None)
    frontier=tso.pareto_rows([zero,normal])
    assert normal in frontier and zero not in frontier


def test_complete_broad_to_adaptive_transition_with_nullable_row(monkeypatch):
    normal=ranking_row('normal')
    zero=ranking_row('zero',recovered=0,orientation=0,median=None,
                     boundary_assd_um=None,true_facet_recall_area_weighted=None,
                     interface_precision=None)
    monkeypatch.setattr(tso,'canonical',lambda parameters,_workspace:parameters)
    monkeypatch.setattr(tso,'parameter_signature',lambda p,_workspace:tuple(p[k] for k in oc.PARAMETER_NAMES))
    before,signature,frontier,parameters=tso.prepare_adaptive_transition(
        [zero,normal],4,object(),0)
    assert before['config_hash']=='normal'
    assert all(np.isfinite(value) for value in signature)
    assert frontier and len(parameters)==4


def test_documented_ranking_component_directions_are_preserved():
    fields=[
      ('orientation_count_score_at_0p02deg',.7,.8),
      ('identity_f1',.7,.8),('orientation_count_score_at_0p01deg',.7,.8),
      ('count_score',.7,.8),
      ('panoptic_quality_at_0p5',.7,.8),('matched_iou_mean',.7,.8),
      ('interface_f1',.7,.8),('boundary_assd_um',.3,.2),
      ('vi_total_bits',.5,.4),('ari',.8,.9),
    ]
    for index,(field,worse,better) in enumerate(fields):
        common={name:best for name,_,best in fields[:index]}
        a=ranking_row(f'a-{index}'); b=ranking_row(f'b-{index}')
        a.update(common); b.update(common)
        a[field]=worse; b[field]=better
        assert tso.rank([a,b])[0]['config_hash']==f'b-{index}'


def test_frozen_source_audit_filters_broad_and_excludes_old_adaptive_rows():
    path=tso.FROZEN_BROAD_OUT/'evaluations.jsonl'
    before=(path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest())
    audit,evidence=tso.audit_frozen_broad_source()
    after=(path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest())
    assert before==after
    assert audit['broad_records']==8000
    assert audit['selectable_broad_records']==6930
    assert audit['expected_algorithmic_invalid_broad_records']==1070
    assert audit['excluded_stage_counts']=={'adaptive_1':1159}
    assert len(evidence)==6930 and all(row['stage']=='broad' for row in evidence)
    assert all(row['selection_policy_id']==tso.SOURCE_SELECTION_POLICY_ID for row in evidence)
    assert all(row['_selection_policy_view_id']==tso.SELECTION_POLICY_ID for row in evidence)
    assert tso.rank(evidence)[0]['config_hash']==tso.EXPECTED_LEADER_HASH


def test_broad_execution_is_mechanically_disabled(tmp_path):
    with np.testing.assert_raises_regex(RuntimeError,'broad execution is disabled'):
        tso.run_tasks([], 'broad', [0], 8, tmp_path/'evaluations.jsonl')


def test_adaptive_plan_is_immutable_and_resumes_only_missing_candidates(tmp_path,monkeypatch):
    monkeypatch.setattr(tso,'OUT',tmp_path)
    monkeypatch.setattr(tso,'IDENTITY',{'python_source_sha256':'p','compiled_extension_sha256':'e',
      'latent_sha256':'l','truth_sha256':'t','mask_sha256':'m','matching_source_sha256':'metrics'})
    p1=oc.Config(.01,-1,.2,1,50,1.2).as_dict()
    p2=oc.Config(.02,-1,.3,1.1,60,1.3).as_dict()
    monkeypatch.setattr(tso,'adaptive',lambda *_args,**_kwargs:[p1,p2])
    prior=[ranking_row('broad-evidence')]
    plan=tso.adaptive_plan(prior,2,object(),1,'source-sha')
    assert tso.adaptive_plan(prior,2,object(),1,'source-sha')==plan
    partial=[{'config_hash':plan['candidates'][0]['config_hash']}]
    complete=partial+[{'config_hash':plan['candidates'][1]['config_hash']}]
    assert not tso.plan_complete(plan,partial)
    assert tso.plan_complete(plan,complete)
    with np.testing.assert_raises_regex(RuntimeError,'adaptive plan mismatch'):
        tso.adaptive_plan(prior,2,object(),1,'changed-source')


def test_destination_manifest_isolation_rejects_broad_and_old_policy_rows(tmp_path,monkeypatch):
    monkeypatch.setattr(tso,'OUT',tmp_path)
    monkeypatch.setattr(tso,'IDENTITY',{'python_source_sha256':'p','compiled_extension_sha256':'e',
      'latent_sha256':'l','truth_sha256':'t','mask_sha256':'m','matching_source_sha256':'metrics'})
    p=oc.Config(.01,-1,.2,1,50,1.2).as_dict(); digest,payload=tso.config_key(p,0)
    row=ranking_row(digest,**p); row.update({'configuration_identity':payload,
      'stage':'adaptive_1','candidate_order_seed':0,'status':'ok'})
    path=tmp_path/'evaluations.jsonl'; assert ResultStore(path,schema_version=4).append(row,payload)[0]
    report,rows=tso.audit_destination_store(); assert report['rows']==1 and rows[0]['schema_version']==4
    raw=json.loads(path.read_text()); raw['stage']='broad'; path.write_text(json.dumps(raw)+'\n')
    with np.testing.assert_raises_regex(RuntimeError,'broad rerun/import'):
        tso.audit_destination_store()
    raw['stage']='adaptive_1'; raw['selection_policy_id']=tso.SOURCE_SELECTION_POLICY_ID; path.write_text(json.dumps(raw)+'\n')
    with np.testing.assert_raises_regex(RuntimeError,'foreign/old selection policy'):
        tso.audit_destination_store()


def test_none_source_metric_persists_as_null_under_strict_json(tmp_path):
    identity={'algorithm':'size_prioritised_multiseed_v1','python_source_sha256':'p',
              'compiled_extension_sha256':'e','latent_sha256':'l','truth_sha256':'t'}
    row={'config_hash':'nullable','configuration_identity':identity,
         'candidate_order_seed':0,'status':'ok',
         'scientific_metrics_version':tso.SCIENTIFIC_METRICS_VERSION,
         'selection_policy_id':tso.SELECTION_POLICY_ID,
         'median_matched_mean_orientation_error_deg':None,
         'worker_pid':1,'worker_peak_rss_bytes':2,'started':'a','finished':'b',
         'elapsed_seconds':.1,'candidate_pass_saturated':False,
         'final_pass_saturated':False}
    path=tmp_path/'rows.jsonl'; assert ResultStore(path).append(row)[0]
    text=path.read_text(); parsed=json.loads(text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert parsed['median_matched_mean_orientation_error_deg'] is None
    assert 'Infinity' not in text and 'NaN' not in text

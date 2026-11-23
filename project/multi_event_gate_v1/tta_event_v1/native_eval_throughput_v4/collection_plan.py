"""Freeze an independent TRAIN performance diagnostic only after official100 release."""
import fcntl
from .common import T,P,C,O,COHORT,TASKS,read,write,sha,identity,mapped,verify
from native_calibrated_validation_v24.collection_plan import calibrated_qualification,actual_zero

def freeze_stage():
 assert COHORT=='branch12'
 O.mkdir(parents=True,exist_ok=True)
 with(O/'study_freeze.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if(O/'study.json').exists():v=read(O/'study.json');verify(v);return v
  final=T/'outputs/native_calibrated_five100_v33/official100_complete.json';result=read(final);verify(result)
  assert result['complete']and result['episodes_each_task']==100 and result['all_actual_host_wait0']
  supervisor=T/'outputs/native_calibrated_official_supervision_v15'
  actual_zero(supervisor/'actual_campaign_exit.json');assert read(supervisor/'complete.json')['complete']
  last=T/'outputs/native_calibrated_five100_v33/official_019_p1';actual_zero(last/'actual_host_wait_exit.json')
  import json
  root=mapped(read(last/'execution_state.json')['root']);release=json.loads((root/'attempt/resources.jsonl').read_text().splitlines()[-1]);assert release['own_gpu_released']and not release['remaining_own_gpu_pids']
  proof=T/'outputs/native_eval_throughput_v4/CPU_contract_checks_v2.json';checks=read(proof);assert checks['passed'];verify(checks);actual_zero(proof.with_name('CPU_actual_host_exit_v2.json'))
  baseline=T/'outputs/native_eval_throughput_v3/branch10';actual_zero(baseline/'actual_host_wait_exit.json')
  base_root=mapped(read(baseline/'execution_state.json')['root']);assert read(base_root/'terminal_audit.json')['passed']and read(base_root/'CPU_governor180_resource_audit.json')['passed']
  accepted=T/'outputs/native_eval_throughput_acceptance_v2/native_eval_throughput_v3/acceptance.json';prior=read(accepted);assert prior['passed']and prior['all_matched_RGB_commands_traces_fast_updates_byte_exact']and not prior['target80_achieved'];verify(prior)
  base_release=json.loads((base_root/'attempt/resources.jsonl').read_text().splitlines()[-1]);assert base_release['own_gpu_released']and not base_release['remaining_own_gpu_pids']and base_release['cleanup_seconds_after_start']==1887.184
  base_spec=read(base_root/'protocol.json');verify(base_spec);assert base_spec['simulation_lanes']==10 and base_spec['actual_seed_start']==7122000 and base_spec['episodes_per_task']==2 and base_spec['inherited_stage_GPU_seconds']==1743.451
  qualification,q=calibrated_qualification();template=T/'outputs/native_calibrated_five100_v33/official_019_p1/study.json'
  v=dict(read(template));verify(v);v.pop('identity');v['source_sha256']=dict(v['source_sha256']);v['input_sha256']=dict(v['input_sha256'])
  v.update(version=C.name,scope='frozen_TRAIN_throughput_diagnostic',scene_role='developed_TRAIN_performance_only_no_fit',stage=COHORT,dataset_split='TRAIN_performance_only_not_new_independent_or_official',
   actual_seed_start=7122000,initial_screen_state={t:dict(next_seed=7122000,accepted=0)for t in TASKS},episodes_per_task=2,simulation_lanes=12,cases_per_batch=10,task_slots=list(TASKS)*2,
   official_EVAL_used=False,no_official_EVAL_used=True,new_official_EVAL_scenes=0,formal_success_not_used_for_fitting=True,official_EVAL_excluded_from_fitting=True,
   inherited_stage_GPU_seconds=0.,stage_total_GPU_budget_seconds=7200,throughput_comparison_total_budget_seconds=7200,logical_batch_index=None,memory_segment_index=None,expected_primary_episodes=30,expected_zero_shadow_episodes=10,
   fixed_memory_maintenance_segments_per_remaining_logical_batch=None,external_rule8_recovery_attempt=0,known_rule4_tail_resume_attempt=0,prepared_recovery_attempt=0,one_audited_cold_to_prepared_resume_only=False,
   cold_start_scheduling_not_proven_until_actual_uncached_stage=True,performance_target_utilization=80.,performance_target_window_seconds=180,host_profile_NOT_GPU_kernel_measurement=True)
  v['cached_preparation']=dict(cached_valid_cases=0,task_cache={},requests=[],case_request_keys={},case_diagnostics={},expert_reference_files={},input_sha256={},new_automatic_recovery_attempts=0)
  v['inherited_stage_GPU_seconds']=base_spec['inherited_stage_GPU_seconds']+base_release['cleanup_seconds_after_start'];assert 0<v['inherited_stage_GPU_seconds']<7200
  v['expected_engineering_reference_episodes']=5;v['logical_batch_count']=1;v['render_wait_safe_requests_enabled']=True;v['native_original_render_overlap_safe_owner_requests']=True
  v['baseline_comparison_root']=str(base_root);v['baseline_resource_failure_retained']=True;v['previous_qualified_comparison_root']=str(base_root);v['baseline_normal_completion_NOT_claimed']=False
  v['input_sha256'][str(accepted)]=sha(accepted)
  v['independent_whole_branches_share_bounded_scene_workers']=True;v['whole_case_qualified_only_after_all_original_branches']=True
  for name in('runtime_edits.json','reuse_edits.json'):v['source_sha256'][str(C/name)]=sha(C/name)
  for name in('benefit_stage_typecheck.py','benefit_prefix_graph.py','benefit_prefix_graph_lifetime_v10.py'):v['source_sha256'][str(P/'deployment'/name)]=sha(P/'deployment'/name)
  v['static_CPU_heap_only_frozen_before_fork']=True;v['original_getImage_async_GIL_readback_with_pixel_oracle']=True;v['original_batch3_prefix_graph_with_current_card_eager_conformance']=True
  for p in(baseline/'actual_host_wait_exit.json',baseline/'pilot_complete.json',base_root/'terminal_audit.json',base_root/'protocol.json',base_root/'attempt/resources.jsonl'):v['input_sha256'][str(p)]=sha(p)
  for helper in('native_render_overlap_v4','native_guide_reuse_v1','native_image_readback_gil_v1','native_batched_prefix_graph_v1','native_direct_frames_v1'):
   for p in(T/helper).glob('*'):
    if p.is_file():v['source_sha256'][str(p)]=sha(p)
  for p in C.rglob('*'):
   if p.is_file()and p.suffix in('.py','.sh','.md'):v['source_sha256'][str(p.relative_to(P))]=sha(p)
  for p in(T/'native_owner_profile_v2').glob('*'):
   if p.is_file():v['source_sha256'][str(p)]=sha(p)
  for p in(final,supervisor/'actual_campaign_exit.json',supervisor/'complete.json',last/'actual_host_wait_exit.json',root/'attempt/resources.jsonl',proof,proof.with_name('CPU_actual_host_exit_v2.json'),qualification,template):v['input_sha256'][str(p)]=sha(p)
  v['identity']=identity(v);verify(v);write(O/'study.json',v);return v

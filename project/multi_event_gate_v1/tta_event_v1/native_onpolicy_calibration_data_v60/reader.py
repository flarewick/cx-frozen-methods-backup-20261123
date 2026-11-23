"""NumPy-only audited offline inputs from real full trajectories and recovery."""
from pathlib import Path
import importlib.util
import numpy as np
from native_policy_execution_v1.common import T,read,sha
from native_onpolicy_recovery_v3.common import mapped,verify,identity
from native_onpolicy_arm_teacher_v1.stream import packet,ARM

def arrays(path):
 with np.load(mapped(path),allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}

class Reader:
 def __init__(self):self.inputs={};self.cache={};self.kernel=None
 def file(self,path,digest):
  path=mapped(path);assert sha(path)==digest
  self.inputs[str(path)]=digest
  if str(path)not in self.cache:self.cache[str(path)]=arrays(path)
  return self.cache[str(path)]
 def qualified(self,root):
  if read(Path(root)/'protocol.json').get('version')=='native_onpolicy_arm_collection_v73':
   from .legacy73_reader import Reader as Previous
   previous=Previous();result=previous.qualified(root);self.inputs.update(previous.inputs);return result
  return self._qualified_current(root)
 def _qualified_current(self,root):
  root=Path(root).resolve();assert root.is_relative_to((T/'outputs').resolve())
  qual=read(root/'atomic_full_recovery_qualification.json');assert qual['passed']and qual['parent_terminal_passed']and qual['recovery_audit_passed']and qual['no_optimizer'];verify(qual)
  terminal=read(root/'terminal_audit.json');assert terminal['passed']and terminal['actual_host_exit_code']==terminal['actual_child_exit_code']==0 and terminal['own_GPU_released'];verify(terminal)
  out=root.parents[3];actual=read(out/'actual_host_wait_exit.json');assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  assert mapped(read(out/'execution_state.json')['root']).resolve()==root
  spec=read(root/'protocol.json');verify(spec);body=dict(spec);digest=body.pop('identity');assert identity(body)==digest==terminal['batch_protocol_identity']
  if spec.get('version')=='native_onpolicy_arm_collection_v75'and spec['stage']=='fit_000':
   assert spec['inherited_stage_GPU_seconds']==9679.983 and spec['stage_total_GPU_budget_seconds']in(10800,14400)
   assert spec['prior_CPU_postexit_failed_batch_excluded']and spec['actual_ARM72_cleanup_seconds_preserved']
   if spec['stage_total_GPU_budget_seconds']==14400:
    from native_FIT0_postexit_carry_v1.budget import AUTH
    authorization=read(AUTH);verify(authorization)
    assert authorization['explicit_user_authorization']and authorization['approved_total_seconds']==14400 and authorization['all_prior_GPU_seconds_retained']==9679.983
   assert spec['explicit_FIT0_budget10800_with_all_prior_elapsed_retained']and spec['prior_budget_stopped_science_excluded']
  if spec.get('version')=='native_onpolicy_arm_collection_v75':
   if spec['stage']=='fit_001':
    assert spec['stage_total_GPU_budget_seconds']==7200 and spec['inherited_stage_GPU_seconds']==283.246
    assert spec['actual_failed_FIT1_GPU_seconds_retained']and spec['eight_actual_prepared_inputs_all_five_tasks_before_GPU']
   assert spec['complete_postexit_audit_chain_with_eight_lane_storage']and qual['complete_eight_lane_postexit_chain']
   assert spec['documented_server_policy_user34_without_extra35_36']and not spec['governor_extra_margin_required']and spec['owner_profile_observes_existing_calls_without_GPU_sync']
   file=root/'CPU_written_policy_audit.json';v=read(file);verify(v);assert v['passed']and v['original_strict34_cumulative_rolling300_and_all_caps']and v['steady_samples']>0
   assert any(mapped(k).resolve()==file.resolve()and value==sha(file)for k,value in qual['input_sha256'].items())
   self.inputs[str(file)]=sha(file);self.inputs.update(v['input_sha256'])
  if spec.get('original_measured_physics_clock_trace_callbacks_in_native_policy_control'):
   p=root/'CPU_measured_control_audit.json';v=read(p);verify(v);assert v['passed']and v['actual_cases']==25 and v['all25_original_whole_raw_controls_required']
   assert any(mapped(name).resolve()==p.resolve()and digest==sha(p)for name,digest in qual['input_sha256'].items())
   from native_measured_policy_control_v2.audit import audit as audit_measured
   from native_audit_path_identity_v1.compare import equal as audit_equal
   assert audit_equal(audit_measured(root,spec),v)
   self.inputs[str(p)]=sha(p);self.inputs.update(v['input_sha256'])
  if spec.get('same_workload_actual_released_context_cache_only_no_NN_import'):
   value=read(root/'CPU_same_device_driver_cache_seed.json');verify(value)
   assert value['NN_compiler_fingerprint_explicitly_null']and value['original_seed_result']['fingerprint_sha256']is None and not value['original_seed_result']['compiler_sampler_artifact_imported']
   from native_released_context_cache_v2.cache import available,Context
   manifest=available(T.parents[1]/'deployment',spec['required_gpu_uuid'],value['driver']);context=Context(manifest,spec['required_gpu_uuid'],value['driver'])
   assert value['context_identity_sha256']==context.record['context_identity_sha256']
  if spec.get('original_same_physical_device_context_cache_seed_before_CUDA'):
   p=root/'CPU_same_device_driver_cache_seed.json';value=read(p);verify(value)
   assert value['passed']and value['actual_gpu_uuid']==spec['required_gpu_uuid']and value['seeded_before_CUDA_permit']and value['restores_context_files_only_NOT_parent_compiled_sampler']
   assert any(mapped(name).resolve()==p.resolve()and digest==sha(p)for name,digest in qual['input_sha256'].items())
   assert value['CPU_ready_sha256']==sha(root/'attempt/barrier/ready.json')
   self.inputs[str(p)]=sha(p);self.inputs.update(value['input_sha256'])
  if spec.get('existing_guide_reuse_100us_typecheck_static_heap_batch3_prefix'):
   from native_guide_reuse_v1.audit import validate as validate_guide
   assert validate_guide(read(root/'guide_prefix_runtime.json'),spec['tasks'],spec['required_gpu_uuid'])
   proof=root/'CPU_guide_reuse_audit.json';value=read(proof);verify(value);assert value['passed']and value['whole25_original_raw_controls_required']
   assert any(mapped(name).resolve()==proof.resolve()and digest==sha(proof)for name,digest in qual['input_sha256'].items())
   for source in[T/'native_guide_reuse_v1/prefix.py',T/'native_guide_reuse_v1/heap.py',T/'native_guide_reuse_v1/typecheck.py']:
    assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
   self.inputs[str(proof)]=sha(proof);self.inputs.update(value['input_sha256'])
  if spec.get('bounded_original_object_state_spatial_storage_only'):
   proof=root/'CPU_bounded_state_audit.json';value=read(proof);verify(value);assert value['passed']and value['whole25_original_raw_controls_still_required']and value['actual_old_spatial_tensors_offloaded']>0
   assert any(mapped(name).resolve()==proof.resolve()and digest==sha(proof)for name,digest in qual['input_sha256'].items())
   for source in [T/'native_bounded_object_state_v1/core.py',T/'native_bounded_object_state_v1/audit.py']:
    assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
   self.inputs[str(proof)]=sha(proof);self.inputs.update(value['input_sha256'])
  if spec.get('native_original_getImage_GIL_release_async_readback'):
   proof=root/'CPU_native_image_readback_audit.json';value=read(proof);verify(value)
   assert value['passed']and value['actual_same_capture_RGBA_byte_exact']and value['original_whole25_controls_still_required']and value['raw_arrays']>=36
   assert any(mapped(name).resolve()==proof.resolve()and digest==sha(proof)for name,digest in qual['input_sha256'].items())
   raw=read(root/'native_image_readback_runtime.json');assert raw['passed']and raw['same_capture_observations']==12 and len(raw['task_types'])>=2
   for source in[T/'native_image_readback_gil_v1/readback_owned_v3.cpp',T/'native_image_readback_gil_v1/readback_shadow_v2.py',T/'native_image_readback_gil_v1/loader_validated.py',T/'native_image_readback_gil_v1/shadow.py',T/'native_render_overlap_v4/scheduler.py',T/'native_render_overlap_v4/codes.py']:
    assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
   self.inputs[str(proof)]=sha(proof);self.inputs.update(value['input_sha256'])
  if spec.get('native_original_render_overlap_safe_owner_requests'):
   proof=root/'CPU_render_overlap_audit.json';value=read(proof);verify(value)
   assert value['passed']and value['original_native_worker_and_single_Torch_owner']and value['original_whole25_controls_separately_required']
   from native_render_overlap_v4.audit import validate as validate_render
   assert validate_render(read(root/'render_overlap_runtime.json'))
   assert any(mapped(name).resolve()==proof.resolve()and digest==sha(proof)for name,digest in qual['input_sha256'].items())
   for source in[T/'native_render_overlap_v4/scheduler.py',T/'native_render_overlap_v4/audit.py',T/'native_prepared_FIT_collection_v30/scope.py']:
    assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
   self.inputs[str(proof)]=sha(proof);self.inputs.update(value['input_sha256'])
  if spec.get('governor_extra_margin_required'):
   proof=root/'CPU_governor_margin_audit.json';value=read(proof)
   assert value['passed']and value['additional_cumulative35_and_process60_180_device60_36']and value['original34_and_all_original_guards_preserved']and value['steady_samples']>0
   verify(value);self.inputs[str(proof)]=sha(proof);self.inputs.update(value['input_sha256'])
   source=T/'native_governor_margin_v2/margin.py'
   assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
  if spec.get('authenticated_recent_CPU_rule4_history_required'):
   proof=root/'CPU_authenticated_history_chain_audit.json';value=read(proof);verify(value)
   from native_authenticated_history_audit_v2.audit import validate as validate_history_chain
   assert validate_history_chain(value)
   assert any(mapped(name).resolve()==proof.resolve()and digest==sha(proof)for name,digest in qual['input_sha256'].items())
   self.inputs[str(proof)]=sha(proof);self.inputs.update(value['input_sha256'])
  if spec.get('initial_ARM_path_feedback'):
   sources=[mapped(name)if Path(name).is_absolute()else T.parents[1]/name for name in spec['source_sha256']if 'native_onpolicy_arm_collection_v'in str(name)and str(name).endswith('/runtime.py')]
   assert len(sources)==1,'One explicitly source-bound operational ARM runtime required'
   expected=(T/'native_onpolicy_collection_v38/runtime.py').read_text().replace('native_onpolicy_mixed_teacher_v3','native_onpolicy_arm_teacher_v1')
   runtime=sources[0].read_text()
   if spec.get('native_batched_readback_12_lanes')or spec.get('eight_lane_with_qualified_original_native_readback'):
    assert spec['render_wait_safe_requests_bypass_ordinary_pending_only']and spec['original_setup6_expert16_and_each_class_FIFO_preserved']
    queue_proof=T/'outputs/native_render_overlap_v4/CPU_contract_checks_v1.json';qv=read(queue_proof);verify(qv)
    assert qv['passed']and qv['unsafe_ordinary_first_bypassed_only_during_render_without_execution']and qv['all_seven_lanes_and_outer_RNG_and_arrays_match_original']
    self.inputs[str(queue_proof)]=sha(queue_proof)
    assert spec['native_original_getImage_GIL_release_async_readback']
    assert spec['primary_raw_observation_priority2_original_model_feedback_priority0']
    priority_source=T/'native_policy_query_priority_v1/priority.py'
    assert any(mapped(name).resolve()==priority_source.resolve()and digest==sha(priority_source)for name,digest in spec['source_sha256'].items())
    priority_proof=T/'outputs/native_policy_query_priority_v1/CPU_contract_checks_v1.json';pp=read(priority_proof);verify(pp);assert pp['passed']and pp['four_cases60_original_calls_all_arrays_and_final_RNG_byte_exact']
    self.inputs[str(priority_proof)]=sha(priority_proof)
    assert spec['simulation_lanes']==(8 if spec.get('eight_lane_with_qualified_original_native_readback')else 12)and spec['original_native_camera_captures_and_render_scene_updates_unchanged']
    assert spec['object_state_storage_CPU_whole_original_raw_controls']and spec['unread_capture_whole_reference_each_case']
    for source in [T/'native_camera_readback_fast_v1/readback.py',T/'native_camera_readback_v2/capture.py',T/'native_prepared_FIT_collection_v30/scope.py']:
     assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
    from native_prepared_FIT_collection_v30.scope import restore_runtime as restore_readback
    runtime=restore_readback(runtime)
    proof=T/'outputs/native_onpolicy_full_ops_v2/ARM41_actual_engineering_result_v1.json';value=read(proof);verify(value)
    assert value['software_probe_passed']and value['raw_byte_comparison_passed']and value['own_GPU_released'];self.inputs[str(proof)]=sha(proof)
   if spec.get('pure11_and_direct_frame_introspection_only'):
    assert spec['object_state_storage_CPU_whole_original_raw_controls']and spec['unread_capture_whole_reference_each_case']
    for source in [T/'native_direct_frames_v1/frames.py',T/'native_prepared_FIT_collection_v2/scope.py',T/'native_fixed_FIT_inputs_v3/verify.py']:
     assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
    for proof in [T/'outputs/native_direct_frames_v1/CPU_contract_checks_v1.json',T/'outputs/native_fixed_FIT_inputs_v3/verified_FIT0_eleven_v1.json']:
     value=read(proof)
     if proof==T/'outputs/native_direct_frames_v1/CPU_contract_checks_v1.json':
      from native_direct_frames_v2.verify import verify_proof as verify_direct_source
      verify_direct_source(value)
     else:verify(value)
     assert value['passed'];self.inputs[str(proof)]=sha(proof)
   if spec.get('exact_CPU_cv2_frame_normalization'):
    assert spec['object_state_storage_CPU_whole_original_raw_controls']and spec['unread_capture_whole_reference_each_case']
    for source in [T/'native_exact_frame_cv2_v1/frames.py',T/'native_exact_frame_collection_v1/scope.py']:
     assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
    from native_exact_frame_collection_v1.scope import restore as restore_exact_frames
    runtime=restore_exact_frames(runtime)
    proof=T/'outputs/native_exact_frame_gpu_v1/actual_report.json';value=read(proof);verify(value)
    assert value['passed']and value['actual_host_exit']==0 and value['own_GPU_released']
    self.inputs[str(proof)]=sha(proof)
   if spec.get('carried_next_FIT0_scope'):
    for source in [T/'native_carried_next_FIT_v1/scope.py',T/'native_carried_next_FIT_v1/carry.py']:
     assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
    from native_carried_next_FIT_v1.scope import restore_runtime as restore_carried
    runtime=restore_carried(runtime)
    if spec.get('carried_FIT0_input_proof'):
     proof=root/'CPU_carried_FIT0_inputs_audit.json';value=read(proof);verify(value)
     assert value['passed']and value['carried_pure_inputs']==4 and value['all_required_inputs']==5
     assert value['fresh_required_task']=='open_laptop'and value['original_negative_validity_rows_preserved']and value['no_failed_policy_masks_or_labels_imported']
     assert any(mapped(name).resolve()==proof.resolve()and digest==sha(proof)for name,digest in qual['input_sha256'].items())
     self.inputs[str(proof)]=sha(proof);self.inputs.update(value['input_sha256'])
   if spec.get('object_state_storage_CPU_whole_original_raw_controls'):
    assert spec['simulation_lanes']==(8 if spec.get('eight_lane_with_qualified_original_native_readback')else 12 if spec.get('native_batched_readback_12_lanes')else 10)and spec['startup_release_after_actual_once_query_not_passive']
    assert not spec['startup_release_requires_actual_complete_initial_native32']
    for source in [T/'native_storage_capture_v1/scope.py',T/'native_storage_capture_v1/source.py',T/'native_storage_capture_v1/audit.py',T/'native_object_state_storage_v2/core.py',T/'native_used_cameras_v1/camera.py']:
     assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
    assert sources[0].with_name('startup.py').read_text().replace('assert lanes in(4,8,9,10,12)','assert lanes in(4,8,9,10)')==(T/'native_onpolicy_arm_collection_v31/startup.py').read_text()
    from native_storage_capture_v1.scope import restore as restore_storage
    runtime=restore_storage(runtime)
    proof=root/'CPU_storage_whole25_audit.json';value=read(proof)
    assert value['passed']and value['whole_pairs']==25 and value['all_raw_feedback_and_original_trajectory_arrays_exact']
    assert value['no_extra_queries_or_frames']and value['frames']>0;verify(value)
    assert any(mapped(name).resolve()==proof.resolve()and digest==sha(proof)for name,digest in qual['input_sha256'].items())
    self.inputs[str(proof)]=sha(proof);self.inputs.update(value['input_sha256'])

   if spec.get('unread_capture_whole_reference_each_case'):
    assert spec['every_original_actual_get_obs_update_random_draw_preserved']
    for source in [T/'native_unread_capture_full_v1/scope.py',T/'native_unread_capture_full_v1/audit.py',T/'native_discard_capture_v1/defer.py']:
     assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
    from native_unread_capture_full_v1.scope import restore_runtime
    runtime=restore_runtime(runtime)
    proof=root/'CPU_unread_capture_whole_reference_audit.json';value=read(proof)
    assert value['passed']and value['whole_original_reference_pairs']==25
    assert value['all_whole_observations_actions_noise_physics_RNG_exact']and value['original_reference_never_elides_captures']and value['reference_excluded_from_labels']
    assert value['skipped_unread_captures']>0;verify(value)
    assert any(mapped(name).resolve()==proof.resolve()and digest==sha(proof)for name,digest in qual['input_sha256'].items())
    self.inputs[str(proof)]=sha(proof);self.inputs.update(value['input_sha256'])
   if spec.get('independent_required_next_population_preparation'):
    assert spec['minimum_actual_current_queries_before_early_preparation']==9
    assert spec['early_fixed_next_population_preparation']==bool(spec.get('future_preparation_population'))
    helper=T/'native_early_next_preparation_v1/scope.py'
    assert any(mapped(name).resolve()==helper.resolve()and digest==sha(helper)for name,digest in spec['source_sha256'].items())
    if spec.get('early_preparation_wait_seconds') is not None:
     assert spec['early_preparation_wait_seconds']==.05
     wait='    future_ready.wait(.05)\n';assert runtime.count(wait)==1
     runtime=runtime.replace(wait,'')
    from native_early_next_preparation_v1.scope import restore as restore_next_schedule
    runtime=restore_next_schedule(runtime)
   if spec.get('future_preparation_uses_unfinished_whole_current_cases'):
    assert spec['future_preparation_minimum_unfinished_current_cases']==4
    assert spec['scheduling_only_revision']['maximum_foreground_before_required_future']==1
    assert runtime.count('with primary_lock:return case_count-len(whole_done)>=4')==1
    runtime=runtime.replace('with primary_lock:return case_count-len(whole_done)>=4','with primary_lock:return case_count-len(primary_done)>=4').replace('primary_done=set();whole_done=set();reference_tasks=set()','primary_done=set();reference_tasks=set()').replace("   with primary_lock:whole_done.add(item['case_id'])\n",'')
   if spec.get('actual_query_gated_startup_lanes'):
    assert spec['actual_query_gated_startup_lanes']==[[0],[1,2,3],[4,5,6,7,8]]
    old_log='dict(lanes=[0],remaining_wait_for_actual_first_query=True'
    assert runtime.count(old_log)==1
    runtime=runtime.replace(old_log,'dict(lanes=[0,1,2,3],remaining_wait_for_actual_first_query=True')
    startup=sources[0].with_name('startup.py')
    assert startup.read_text()==(T/'native_onpolicy_arm_collection_v22/startup.py').read_text()
    assert any(mapped(name).resolve()==startup.resolve()for name in spec['source_sha256'])
   assert runtime==expected,'Scientific runtime change requires new dataset qualification'
   assert (sources[0].parent/'episode.py').read_text()==(T/'native_onpolicy_arm_collection_v5/episode.py').read_text()
  lifecycle=T/'native_expert_case_release_v1/release.py'
  if str(lifecycle.relative_to(T.parents[1]))in spec['source_sha256']:
   proof=root/'CPU_closed_expert_task_release_audit.json';released=read(proof);assert released['passed']and released['all_closed_scene_and_task_and_proxy_weakrefs_dead']and released['one_original_shared_pair_preserved']and released['original_model_and_RNG_preserved'];verify(released)
   self.inputs[str(proof)]=sha(proof);self.inputs.update(released['input_sha256'])
   if spec.get('q0_scene_handoff_with_whole_native_reference_controls'):
    cached=spec.get('cached_preparation')or{}
    expected=spec['cases_per_batch']-cached.get('cached_valid_cases',0)
    tasks={t for t in spec['tasks']if len(cached.get('task_cache',{}).get(t,{}).get('items',[]))<5}
    assert released['closed_after_full_not_after_screen']and released['transferred_q0_tasks']==expected
    if spec.get('unread_capture_whole_reference_each_case'):
     assert released['whole_native_reference_controls']==spec['cases_per_batch']==25
     assert released['cached_revalidated_cases']==cached.get('cached_valid_cases',0)and released['all_cached_policy_entries_byte_exact']
    elif spec.get('cached_q0_semantics_lazy_revalidation'):
     assert released['cached_revalidated_cases']==cached.get('cached_valid_cases',0)
     assert released['all_cached_policy_entries_byte_exact']
     regenerated=released['regenerated_cached_cases'];assert len(regenerated)==len(set(regenerated))
     assert released['cached_regeneration_whole_controls']==len(regenerated)
     assert released['whole_native_reference_controls']==len(tasks)+len(regenerated)
    elif spec.get('cached_q0_regenerated_from_actual_current_native_scene'):
     assert released['current_cached_q0_tasks']==cached.get('cached_valid_cases',0)
     assert released['all_actual_current_q0_tasks']==spec['cases_per_batch']
     assert released['old_cache_physics_RNG_commands_exact']
     assert released['whole_native_reference_controls']==len(spec['tasks'])
    else:assert released['whole_native_reference_controls']==len(tasks)
    if spec.get('obsolete_MPLIB_world_detached_after_expert_q0'):
     assert released['all_obsolete_MPLIB_scenes_released']
  CPU_admission_sources=[mapped(name)if Path(name).is_absolute()else T.parents[1]/name for name in spec['source_sha256']if str(name).endswith('/cpu_admission.py')]
  if CPU_admission_sources:
   assert all(sha(f)==sha(T/'native_onpolicy_collection_v26/cpu_admission.py')for f in CPU_admission_sources),'New CPU admission body requires new reader qualification'
   previous=[r for r in root.parent.iterdir()if r.is_dir()and r!=root];assert len(previous)<=2
   for prior in previous:
    proof=read(prior/'CPU_only_replan_qualified.json');assert proof['passed']and proof['no_GPU_or_scientific_work']and proof['prepared_and_semantic_processes_actually_waited']and proof['no_GPU_clock_or_budget_reset']and proof['not_native_data_qualification']
    assert not(prior/'attempt/barrier/permit.json').exists()
    f=prior/'attempt/resources.jsonl';assert not f.exists()or f.stat().st_size==0
    assert not(prior/'GPU_episodes_complete.json').exists()and not(prior/'terminal_audit.json').exists()
    verify(read(prior/'protocol.json'))
    for name in('CPU_admission_race.json','CPU_only_replan_qualified.json','CPU_admission_prepared_exit.json','semantic_exit.json','source_bound_CPU_preload_proof.json','protocol.json'):
     f=prior/name;self.inputs[str(f)]=sha(f)
    for name in('CPU_admission_prepared_exit.json','semantic_exit.json'):
     actual=read(prior/name);assert actual['actual_Popen_wait_return']and actual.get('proc_absent_after_wait',True)
  if spec.get('CPU_semantic_adaptive_total8_threads'):
   proof=root/'CPU_semantic_thread_budget_audit.json';bounded=read(proof);assert bounded['passed']and bounded['limit8_preserved']and bounded['maximum_active_threads']<=8 and bounded['workers_actually_waited0']
   assert bounded['actual_requests']==read(root/'semantic/result.json')['requests']and bounded['actual_requests']>0;verify(bounded)
   assert all(mapped(name).resolve().is_relative_to((root/'semantic').resolve())for name in bounded['input_sha256'])
   self.inputs[str(proof)]=sha(proof);self.inputs.update(bounded['input_sha256'])
  self.admission_proof(root,spec)
  missing_guard=T/'native_process_missing_guard_v2/guard.py'
  if str(missing_guard.relative_to(T.parents[1]))in spec['source_sha256']:
   assert spec['fresh_valid_NOT_FOUND_unknown_interval_zero_bound_v2']and spec['resource_guard_revision']=='fresh_validated_NOT_FOUND_dense_clock_entire_unknown_interval_zero_lower_bound_v2'
  idle=T/'native_process_idle_guard_v1/guard.py'
  if str(idle.relative_to(T.parents[1]))in spec['source_sha256']:
   assert spec['conservative_idle_missing_lower_bound_guard']
   proof=root/'CPU_idle_missing_lower_bound_audit.json';bounded=read(proof);assert bounded['passed']and bounded['actual_samples']>0 and bounded['steady_samples']>0 and bounded['all_process_windows_recomputed_exact']and bounded['unknown_interval_integrated_at_zero_not_prior_positive']and bounded['instantaneous_missing_values_preserved']and bounded['all_original_average_floors_34']and bounded['no_device_to_process_substitution']
   assert any(mapped(name).resolve()==(root/'protocol.json').resolve()and digest==sha(root/'protocol.json')for name,digest in bounded['input_sha256'].items())
   assert any(mapped(name).resolve()==(root/'attempt/resources.jsonl').resolve()and digest==sha(root/'attempt/resources.jsonl')for name,digest in bounded['input_sha256'].items())
   verify(bounded);self.inputs[str(proof)]=sha(proof);self.inputs.update(bounded['input_sha256'])
  rule4=[T/package/'history.py'for package in('native_server_rule4_v2','native_server_rule4_v3','native_server_rule4_v4','native_server_rule4_v5','native_server_rule4_v6','native_server_rule4_v7','native_server_rule4_v8','native_server_rule4_v9','native_server_rule4_v10')]
  if any(str(source.relative_to(T.parents[1]))in spec['source_sha256']for source in rule4):
   proof=root/'CPU_cautious_rule4_terminal.json';resource=read(proof);assert resource['passed']and resource['actual_active_samples']>0 and resource['original_scientific_resource_thresholds_unchanged'];verify(resource)
   self.inputs[str(proof)]=sha(proof);self.inputs.update(resource['input_sha256'])
  account=T/'native_account_stream_v1/transport.py'
  if str(account.relative_to(T.parents[1]))in spec['source_sha256']:
   proof=root/'CPU_account_transport_terminal.json';stream=read(proof);assert stream['passed']and stream['actual_normal_CPU_workers']==3 and stream['actual_fresh_account_scans']>0 and stream['original_collection_bodies_and_three_second_deadline_unchanged']and stream['no_GPU_metrics_or_accounting_clock_changed'];verify(stream)
   self.inputs[str(proof)]=sha(proof);self.inputs.update(stream['input_sha256'])
  report=read(root/'CPU_recovery_stream_audit.json');assert report['passed']and report['optimizer_steps']==0
  cases=read(root/'GPU_episodes_complete.json');assert cases['complete']and cases['frozen_parameter_pointers_and_mutation_counters_unchanged']and cases['protocol_identity']==digest
  if spec.get('q0_once_policy_overlaps_initial_CPU_semantics_before_any_action'):
   for case in cases['cases']:self.overlap_proof(root/'cases'/case['case_id'])
  assert len(report['records'])==len(cases['cases'])and {x['case_id']for x in report['records']}=={x['case_id']for x in cases['cases']}
  for p in(root/'atomic_full_recovery_qualification.json',root/'terminal_audit.json',root/'CPU_recovery_stream_audit.json',root/'protocol.json',out/'actual_host_wait_exit.json'):self.inputs[str(p)]=sha(p)
  return spec,cases['cases'],report
 def overlap_proof(self,folder):
  result=folder/'M_event_full/result.json';record=read(result);self.inputs[str(result)]=sha(result)
  proof=folder/'M_event_full_q0_policy_semantic_overlap.json'
  if not record['queries']:
   assert not proof.exists(),'No query may claim a policy/semantic overlap'
   return
  first=record['queries'][0];policy=first['policy'];assert first['query']==first['tick']==policy['query']==policy['actor_observation_tick']==0
  assert policy['base_infer_calls']==policy['actual_internal_noise_calls']==1 and policy['environment_audit']['native_environment_unchanged']
  row=read(proof);assert row['query']==row['physical_tick']==0
  for key in('actual_once_q0_before_semantic_resolution','no_actor_execution_before_semantic_resolution','no_additional_query_or_native_step'):assert row[key] is True
  assert type(row['forward_file_precedes_CPU_answer'])is bool  # Cached answers may predate the policy; no latency selection.
  receipt=folder/'M_event_full/semantic_initial_receipt.json';semantic=read(receipt)
  assert semantic['query']==semantic['tick']==0 and semantic['all_three_RGB_command_language_exact']
  for pathkey,hashkey,expectedpath,expectedhash in[
   ('actual_once_policy_file','actual_once_policy_sha256',policy['policy_file'],policy['policy_sha256']),
   ('CPU_semantic_file','CPU_semantic_sha256',semantic['CPU_result_file'],semantic['CPU_result_sha256'])]:
   path=mapped(row[pathkey]);assert path.resolve()==mapped(expectedpath).resolve()and row[hashkey]==expectedhash==sha(path)
   self.inputs[str(path)]=expectedhash
  for path in(proof,receipt):self.inputs[str(path)]=sha(path)
 def admission_proof(self,root,spec):
  if spec.get('version')=='native_onpolicy_arm_collection_v75':assert spec['simulation_lanes']==8 and spec['eight_lane_with_qualified_original_native_readback']and not spec['native_batched_readback_12_lanes']
  if spec.get('verified_native_q0_releases_remaining_real_lanes_before_CPU_semantic_wait'):
   lanes=spec['simulation_lanes'];assert lanes in(8,9,10)or(lanes==12 and spec.get('native_batched_readback_12_lanes'))
   admission=root/('startup_release_'+str(lanes-4)+'.json');row=read(admission)
   assert row['released_lanes']==list(range(4,lanes))and row['continuous_resource_clock']
   if row['reason']=='actual_verified_native_q0_RGB_command_prefetch':
    assert row['native_action_count']==0 and row['no_original_actor_query_fabricated']
    cid=row['case_id'];assert read(root/'accepted'/(cid+'.json'))['case_id']==cid
    initial=root/'initial_prefetch'/(cid+'.json');meta=read(initial);assert meta['content_key']==row['content_key']
    req=root/'semantic/requests'/(row['content_key']+'.json');request=read(req);frame=mapped(request['frame_file']);assert sha(frame)==request['frame_sha256']
    for f in(initial,req,frame):self.inputs[str(f)]=sha(f)
   else:
    assert (spec.get('cached_preparation')or{}).get('cached_valid_cases',0)>0
    assert row['reason']in('actual_once_policy_query_complete','actual_case_terminal')and 0<=row['lane']<4
    if row['reason']=='actual_once_policy_query_complete':assert row['query']==row['tick']==0
   self.inputs[str(admission)]=sha(admission)
 def predictor(self):
  if self.kernel is None:
   root=T/'outputs/native_phase_feedback_v1/frozen_auxiliary';protocol=read(root/'protocol.json');assert protocol['auxiliary_candidate_frozen']and protocol['primary_model']=='correct'and protocol['ridge']==1.
   file=root/'predictor.npz';assert sha(file)==protocol['checkpoint_sha256']=='86069427fa4a4e2f9386227e35bdaebbcd18a70773dff4b7686ef83b854fb0c1'
   source=T/'native_local_response_v2/model.py';assert any(mapped(p).resolve()==source.resolve()and h==sha(source)for p,h in protocol['source_sha256'].items())
   spec=importlib.util.spec_from_file_location('actual_frozen_onpolicy_coordinates',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   state=arrays(file)
   for a in state.values():a.setflags(write=False)
   self.kernel=(module,state)
   for p in(root/'protocol.json',file,source):self.inputs[str(p)]=sha(p)
  return self.kernel
 def example(self,root,spec,case,row):
  # The caller must have verified this exact atomic parent, not a short probe.
  assert str(Path(root)/'atomic_full_recovery_qualification.json')in self.inputs
  assert row['case_id']==case['case_id']and row['label_available']and row['provisional_until_entire_parent_batch_qualifies']
  folder=Path(root)/'cases'/case['case_id']/'M_event_full';p=packet(spec,folder,case);assert p is not None and p['query']==row['query']and p['choice']['application_mode']==row['application_mode']
  record=read(folder/'result.json');q=p['choice']['packet']['query'];source=p['choice']['packet']['source_query'];packetrow=p['choice']['packet'];byq={r['query']:r for r in record['queries']}
  assert len(byq)==len(record['queries'])and sorted(byq)==list(range(len(byq)))
  local=self.file(packetrow['source_local_file'],packetrow['source_local_sha256']);ids=np.asarray(packetrow['row_ids'],np.int64);assert len(ids)>0 and len(set(ids.tolist()))==len(ids)
  issuing=local['command_query'][ids];assert set(issuing.tolist())<=set((source,source+1))and np.all(issuing<q)
  hidden=[]
  for commandq in issuing:
   policy=byq[int(commandq)]['policy'];a=self.file(policy['policy_file'],policy['policy_sha256']);assert a['hidden'].shape==(32,64);hidden.append(a['hidden'])
  selected=byq[p['query']];current=byq[q]
  def policy(row):
   v=row['policy'];obs=self.file(row['observation']['file'],row['observation']['sha256']);actual=self.file(v['observation_file'],v['observation_sha256']);assert v['actor_observation_tick']==row['tick']and actual.keys()==obs.keys()and all(np.array_equal(actual[k],obs[k])for k in obs)
   return self.file(v['policy_file'],v['policy_sha256'])
  current=policy(current);action=policy(selected);label=self.file(row['target_file'],row['target_sha256']);valid=label['valid_offsets'];assert valid.dtype==np.int64 and len(valid)>0 and np.all((valid>=0)&(valid<32))
  assert np.array_equal(label['once_base_ARM'],action['base'][valid][:,ARM])and np.array_equal(label['direction_target_ARM'],np.clip(label['teacher_ARM']-label['once_base_ARM'],-.1,.1).astype(np.float32))
  module,state=self.predictor();xx,u,kind,_=module.coordinates(local['context'][ids],local['command'][ids],local['start_points'][ids],'correct');prediction=module.predict(state,xx,u,kind)
  result=dict(hidden=np.stack(hidden),context=xx,command=u,kind=kind,kernel_prediction=prediction,target=local['target'][ids],next_hidden=current['hidden'][None],next_logits=current['logits'][:,ARM][None],next_reference=current['raw_delta'][:,ARM][None],action_hidden=action['hidden'][None],action_logits=action['logits'][:,ARM][None],action_reference=action['raw_delta'][:,ARM][None],action_valid_offsets=valid.copy(),action_teacher_target=label['direction_target_ARM'])
  assert all(np.isfinite(a).all()for a in result.values())and result['hidden'].shape==(len(ids),32,64)and result['command'].shape==(len(ids),194)and result['action_teacher_target'].shape==(len(valid),12)
  result.update(application_mode=row['application_mode'],actual_feedback_query=q,offline_action_query=p['query'])
  return result

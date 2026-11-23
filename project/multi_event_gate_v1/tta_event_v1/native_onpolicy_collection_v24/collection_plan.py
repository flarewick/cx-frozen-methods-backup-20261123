"""Freeze each independent25-case stage under one fixed250 population/recipe."""
from pathlib import Path
from .common import T,P,C,O,COHORT,TASKS,read,write,sha,mapped,verify,identity
def _freeze_stage():
 from .contracts import check
 from .known_preparation import collect
 check();split,number=COHORT.split('_');number=int(number);assert split in('fit','val')and 0<=number<(8 if split=='fit'else 2)
 if(O/'study.json').exists():value=read(O/'study.json');verify(value);return value
 checks=T/'outputs/native_onpolicy_collection_v24/CPU_collection_contract_checks_v2.json';cpu=read(checks);assert cpu['passed']and cpu['sam2_original_allocator_check']['passed']and cpu['actual_dynamic_dispatch_and_case_RNG']['native_lanes_fixture']==9;verify(cpu)
 base=T/'outputs/native_onpolicy_coverage_v8/pilot_fit';r=mapped(read(base/'execution_state.json')['root']);parent=read(r/'protocol.json');verify(parent);audit=read(r/'CPU_recovery_stream_audit.json')
 assert read(r/'atomic_full_recovery_qualification.json')['passed']and read(base/'actual_host_wait_exit.json')['exit_code']==0
 for task in TASKS:
  if task!='grab_roller':assert audit['groups'][task]['labels']>0
 live=T/'outputs/native_live_joint_teacher_v1/pilot_fit';lr=mapped(read(live/'execution_state.json')['root']);proof=read(live/'coverage_result.json')
 assert proof['coverage_passed']and read(live/'actual_host_wait_exit.json')['exit_code']==0 and read(lr/'terminal_audit.json')['passed']
 assert proof['bounded_interface_only']and proof['no_sustained_resource_or_bulk_dataset_qualification'],'Short probe never becomes fitting data'
 start=7120000 if split=='fit'else 7121000;initial={t:dict(next_seed=start,accepted=0)for t in TASKS};previous=[]
 if number:
  from .prefix_stage import qualified
  prior=qualified(split,number-1)
  pr=mapped(read(prior/'execution_state.json')['root']);assert read(pr/'atomic_full_recovery_qualification.json')['passed']and read(prior/'actual_host_wait_exit.json')['exit_code']==0
  initial=read(pr/'screening_complete.json')['state'];assert all(v['accepted']==number*5 for v in initial.values());previous=[pr/'atomic_full_recovery_qualification.json',pr/'screening_complete.json',prior/'actual_host_wait_exit.json']
 cached=collect(initial,TASKS,(number+1)*5,start);value=dict(parent);value.pop('identity');value.pop('cached_preparation',None)
 value.update(version='native_onpolicy_collection_v24',scope='frozen_full_onpolicy_TRAIN_capture',stage=COHORT,scene_role='complete_predeclared_FIT40_VAL10_current_full_native_recovery',dataset_split='adapter_fit'if split=='fit'else'adapter_validation',actual_seed_start=start,initial_screen_state=initial,episodes_per_task=(number+1)*5,simulation_lanes=9,cases_per_batch=25,task_slots=list(TASKS)*5,batch_index=0,logical_batch_count=1,perception_interface='native_object_association_v1',cached_preparation=cached,future_preparation_each_task=0,future_preparation_is_required_next_fixed_dataset_expert_work=True,coverage_expansion_only=False,teacher_strategy='single-source nearest reference; actual two-sided source live current-geometry opening+official expert+independent native replay',all_teacher_and_GT_offline_only=True,full_fixed_population=dict(FIT_each_task=40,VAL_each_task=10,FIT_seed_start=7120000,VAL_seed_start=7121000),old_short_probe_labels_excluded_from_training=True,no_automatic_retry_of_unknown_failure=True)
 value['inherited_stage_GPU_seconds']=68.287 if split=='fit'and number==1 else 0.
 value['known_governor_interruption_preserved']=True
 if split=='fit'and number==2:
  lost=[]
  for version in('native_onpolicy_collection_v8','native_onpolicy_collection_v9'):
   path=T/'outputs'/version/COHORT;oldroot=mapped(read(path/'execution_state.json')['root']);import json
   actual=read(path/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
   logs=[json.loads(x)for x in(oldroot/'attempt/resources.jsonl').read_text().splitlines()];assert logs[-1]['own_gpu_released'];lost.append(logs[-1]['cleanup_seconds_after_start'])
  value['inherited_stage_GPU_seconds']=sum(lost)
 if split=='fit'and number==3:
  import json
  prior=T/'outputs/native_onpolicy_collection_v13/fit_003';r0=mapped(read(prior/'execution_state.json')['root']);actual=read(prior/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  resource=r0/'attempt/resources.jsonl';rr=[json.loads(x)for x in resource.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
  value['inherited_stage_GPU_seconds']=rr[-1]['cleanup_seconds_after_start'];value['preserved_cold_start_resource_failure']=True
 if split=='fit'and number==4:
  import json
  prior=T/'outputs/native_onpolicy_collection_v14/fit_004';oldroot=mapped(read(prior/'execution_state.json')['root']);actual=read(prior/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['proc_absent_after_wait']
  rows=[json.loads(x)for x in(oldroot/'attempt/resources.jsonl').read_text().splitlines()];assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']
  value['inherited_stage_GPU_seconds']=rows[-1]['cleanup_seconds_after_start']
  prior=T/'outputs/native_onpolicy_collection_v15/fit_004';oldroot=mapped(read(prior/'execution_state.json')['root']);actual=read(prior/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['proc_absent_after_wait']
  rows=[json.loads(x)for x in(oldroot/'attempt/resources.jsonl').read_text().splitlines()];assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']
  value['inherited_stage_GPU_seconds']+=rows[-1]['cleanup_seconds_after_start']
  prior=T/'outputs/native_onpolicy_collection_v16/fit_004';oldroot=mapped(read(prior/'execution_state.json')['root']);actual=read(prior/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['proc_absent_after_wait']
  rows=[json.loads(x)for x in(oldroot/'attempt/resources.jsonl').read_text().splitlines()];assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']
  value['inherited_stage_GPU_seconds']+=rows[-1]['cleanup_seconds_after_start']
 if split=='fit'and number==4:
  for version in('native_onpolicy_collection_v17','native_onpolicy_collection_v18','native_onpolicy_collection_v19','native_onpolicy_collection_v20','native_onpolicy_collection_v21','native_onpolicy_collection_v22','native_onpolicy_collection_v23'):
   prior=T/'outputs'/version/'fit_004';oldroot=mapped(read(prior/'execution_state.json')['root']);actual=read(prior/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
   rows=[json.loads(x)for x in(oldroot/'attempt/resources.jsonl').read_text().splitlines()];assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids'];value['inherited_stage_GPU_seconds']+=rows[-1]['cleanup_seconds_after_start']
 value['stage_total_GPU_budget_seconds']=9000 if split=='fit'and number==4 else 7200
 value['stage_walltime_budget_extension_preserves_all_elapsed']=True
 value['sam2_uses_original_unused_allocator_check']=True
 value['sam2_tracker_owner_dispatch_repaired_after_actual_C22_failure']=True
 value['conservative_idle_missing_lower_bound_guard']=True
 value['fresh_valid_NOT_FOUND_unknown_interval_zero_bound_v2']=True
 value['resource_guard_revision']='fresh_validated_NOT_FOUND_dense_clock_entire_unknown_interval_zero_lower_bound_v2'
 value['account_scan_transport_only_revision']=False
 value['cached_q0_semantics_lazy_revalidation']=True
 value['old_RGB_change_requires_same_scene_whole_fresh_full_reference']=True
 value['native_setup_stage_profiling_no_scientific_input']=True
 value['q0_scene_handoff_with_whole_native_reference_controls']=True
 value['obsolete_MPLIB_world_detached_after_expert_q0']=True
 value['account_scan_transport_only_revision']=True
 value['extra_future_expert_scene_disabled_after_actual_render_buffer_failure']=True
 value['scheduling_only_revision']=dict(cached_prefix_barrier='actual first original policy query or actual case terminal',current_recovery_priority_over_future=True,future_dispatch_min_remaining_full_cases=4,partial_future_inputs_audited_by_actual_count=True,unstarted_future_cases_still_required=True)
 value['source_sha256']=dict(parent['source_sha256']);value['input_sha256']=dict(parent['input_sha256']);value['input_sha256'].update(cached['input_sha256'])
 for directory in(C,T/'native_onpolicy_mixed_teacher_v3',T/'native_expert_case_release_v1',T/'native_metadata_stream_v1',T/'native_server_rule4_v5',T/'native_account_stream_v1',T/'native_expert_request_priority_v1',T/'native_q0_task_handoff_v3',T/'native_scene_free_topp_v2',T/'native_process_missing_guard_v2',T/'native_server_rule4_v8'):
  for f in directory.rglob('*'):
   if f.is_file()and f.suffix in('.py','.sh','.md'):value['source_sha256'][str(f.relative_to(P))]=sha(f)
 for f in [T/'outputs/native_scene_free_topp_v2/CPU_actual_MPLIB_scene_release_checks.json',T/'outputs/native_scene_free_topp_v2/CPU_actual_host_exit.json',T/'outputs/native_q0_task_handoff_v3/CPU_actual_checks.json',T/'outputs/native_q0_task_handoff_v3/CPU_actual_host_exit_v2.json',T/'outputs/native_expert_request_priority_v1/CPU_actual_dispatch_checks.json',T/'outputs/native_expert_request_priority_v1/CPU_actual_host_exit.json',T/'outputs/native_server_rule4_v5/CPU_actual_V11_history_handoff_checks.json',T/'outputs/native_server_rule4_v5/CPU_actual_host_exit.json',T/'outputs/native_account_stream_v1/CPU_account_terminal_audit_checks.json',T/'outputs/native_account_stream_v1/CPU_account_transport_checks.json',T/'outputs/native_account_stream_v1/CPU_account_actual_host_exit.json',T/'outputs/native_onpolicy_collection_v24/CPU_scheduling_contract_checks_v2.json',T/'outputs/native_onpolicy_collection_v24/CPU_collection_actual_host_exit_v2.json',T/'outputs/native_server_rule4_v3/CPU_actual_handoff_admission_checks_v2.json',T/'outputs/native_server_rule4_v3/CPU_actual_frozen_history_handoff_checks.json',T/'outputs/native_onpolicy_full_ops_v1/GPU3_readonly_health_bound.json',T/'outputs/native_server_rule4_v2/V6_actual_governor_interruption.json',T/'outputs/native_onpolicy_mixed_teacher_v3/CPU_actual_live_lifetime_checks.json',T/'outputs/native_onpolicy_mixed_teacher_v3/CPU_actual_host_exit.json',T/'outputs/native_expert_case_release_v1/CPU_actual_closed_expert_release_checks.json',T/'outputs/native_expert_case_release_v1/CPU_actual_host_exit_v2.json',T/'outputs/native_metadata_stream_v1/CPU_transport_checks.json',checks,base/'actual_host_wait_exit.json',r/'atomic_full_recovery_qualification.json',r/'CPU_recovery_stream_audit.json',live/'coverage_result.json',live/'actual_host_wait_exit.json',lr/'terminal_audit.json',*previous]:value['input_sha256'][str(f)]=sha(f)
 plan=T/'outputs/native_feedback_after_train_v1/data_plan.json';v=read(plan);assert v['identity']=='cfc1090bbe3a0a6a81f748c71361574e744b1edab9b5e525ed23dfcd31cfe66e';value['input_sha256'][str(plan)]=sha(plan)
 for f in(T/'native_fast_inventory_v1/worker.py',T/'native_fast_inventory_v1/inventory.py',T/'native_server_rule4_v1/worker.py',T/'native_resource_policy_v4/admission.py'):value['input_sha256'][str(f)]=sha(f)
 for name in('CPU_actual_C20_idle_lower_bound_checks.json','CPU_actual_host_exit.json','CPU_actual_host_guard_load_checks_v2.json','CPU_guard_load_actual_host_exit_v2.json','CPU_actual_resource_audit_roundtrip_checks.json','CPU_resource_roundtrip_actual_host_exit.json'):
  file=T/'outputs/native_process_idle_guard_v1'/name;proof=read(file);assert proof.get('passed',proof.get('exit_code')==0);verify(proof);value['input_sha256'][str(file)]=sha(file)
 for name in('CPU_actual_C23_history_handoff_checks.json','CPU_actual_host_exit.json'):
  file=T/'outputs/native_server_rule4_v8'/name;proof=read(file);assert proof.get('passed',proof.get('exit_code')==0);verify(proof);value['input_sha256'][str(file)]=sha(file)
 for name in('CPU_actual_C20_idle_lower_bound_checks.json','CPU_actual_host_exit_v3.json','CPU_actual_host_guard_load_checks_v2.json','CPU_host_load_actual_host_exit_v3.json','CPU_actual_resource_audit_roundtrip_checks.json','CPU_roundtrip_actual_host_exit_v3.json','CPU_actual_C23_resource_audit_roundtrip_checks.json','CPU_C23_roundtrip_actual_host_exit_v3.json'):
  f=T/'outputs/native_process_missing_guard_v2'/name;proof=read(f);assert proof.get('passed',proof.get('exit_code')==0);verify(proof);value['input_sha256'][str(f)]=sha(f)
 value['input_sha256'].update(cpu['lazy_current_q0_regression']['input_sha256'])
 value['identity']=identity(value);verify(value);write(O/'study.json',value);return value


def freeze_stage():
 import fcntl
 O.mkdir(parents=True,exist_ok=True)
 with(O/'study_freeze.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _freeze_stage()

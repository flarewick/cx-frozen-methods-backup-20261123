"""Prospective fixed75 augmentation after separate native mechanism validation."""
import fcntl
from .common import T,C,P,O,COHORT,TASKS,read,write,sha,identity,verify,mapped

def VAL_budget_authorization():
 from native_unread_capture_full_v1.budget import authorization
 return authorization(T)

def freeze_stage():
 O.mkdir(parents=True,exist_ok=True)
 with(O/'study_freeze.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if(O/'study.json').exists():v=read(O/'study.json');verify(v);return v
  split,number=COHORT.split('_');number=int(number);assert split in('fit','val')and 0<=number<(2 if split=='fit'else 1)
  budget_file,budget=VAL_budget_authorization() if COHORT.startswith('val_')else(None,None)
  checks=T/'outputs/native_onpolicy_arm_collection_v35/CPU_contract_checks_v2.json';assert read(checks)['passed'];verify(read(checks))
  actual=read(T/'outputs/native_onpolicy_arm_collection_v35/CPU_actual_host_exit_v2.json');assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  mechanism=T/'outputs/native_initial_arm_mechanism_v4/mechanism';actual=read(mechanism/'actual_host_wait_exit.json');assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  report=read(mechanism/'mechanism_complete.json');assert report['complete']and report['coverage_passed']and report['all_whole_native_controls_exact']and report['excluded_from_calibration']
  parent=mapped(read(mechanism/'execution_state.json')['root']);assert read(parent/'terminal_audit.json')['passed'];verify(read(parent/'protocol.json'))
  v=dict(read(T/'outputs/native_onpolicy_collection_v24/fit_004/study.json'));v.pop('identity');v['source_sha256']=dict(v['source_sha256']);v['input_sha256']=dict(v['input_sha256'])
  start=7124000 if split=='fit'else 7125000;initial={t:dict(next_seed=start,accepted=0)for t in TASKS};previous=[]
  if number:
   from .prefix_stage import qualified
   old=qualified(split,number-1);root=mapped(read(old/'execution_state.json')['root'])
   from native_onpolicy_calibration_data_v28.reader import Reader
   Reader().qualified(root);initial=read(root/'screening_complete.json')['state'];assert all(s['accepted']==number*5 for s in initial.values());previous=[old/'actual_host_wait_exit.json',root/'atomic_full_recovery_qualification.json',root/'screening_complete.json']
  from native_onpolicy_calibration_data_v17.dataset import BASE
  original=next((T/'outputs'/v/'fit_complete.json'for v in BASE if(T/'outputs'/v/'fit_complete.json').exists()),None);assert original is not None
  done=read(original);assert done['complete']and len(done['stages'])==8
  cold=original.parent/'fit_007';coldroot=mapped(read(cold/'execution_state.json')['root']);coldactual=read(cold/'actual_host_wait_exit.json');assert coldactual['exit_code']==0 and coldactual['actual_Popen_wait_return']and coldactual['proc_absent_after_wait']
  from native_onpolicy_calibration_data_v28.reader import Reader
  coldspec,_,_=Reader().qualified(coldroot);assert coldspec['cached_preparation']['no_policy_outputs_masks_or_labels_imported']
  admission=coldroot/('startup_release_'+str(coldspec['simulation_lanes']-4)+'.json')
  assert coldspec['completed_expert_q0_preparation_reused_not_failed_scientific_results']and coldspec['legacy_all_cold_expert_startup_NOT_qualified_or_claimed']
  assert read(admission)['reason'] in ('actual_verified_native_q0_RGB_command_prefetch','actual_once_policy_query_complete','actual_case_terminal')
  previous.extend([original,cold/'actual_host_wait_exit.json',coldroot/'atomic_full_recovery_qualification.json',admission])
  v.update(version='native_onpolicy_arm_collection_v35',scope='frozen_full_onpolicy_TRAIN_capture',scene_role='fixed_additional_ARM_FIT10_VAL5_original250_retained',stage=COHORT,dataset_split='adapter_fit'if split=='fit'else'adapter_validation',actual_seed_start=start,initial_screen_state=initial,episodes_per_task=(number+1)*5,simulation_lanes=10,cases_per_batch=25,task_slots=list(TASKS)*5,conditions=['M_event_full'],future_preparation_each_task=0,inherited_stage_GPU_seconds=0.,initial_ARM_path_feedback=True,all_original250_identities_still_required=True,teacher_strategy='initial ARM source live current-geometry opening+official expert+exact native replay; original grip sources unchanged',all_teacher_and_GT_offline_only=True)
  from .known_preparation import collect
  cached=collect(initial,TASKS,(number+1)*5,start);v['cached_preparation']=cached;v['input_sha256'].update(cached['input_sha256'])
  assert cached['no_policy_outputs_masks_or_labels_imported']
  if split=='fit':assert cached['cached_valid_cases']>=5,'Required genuine next-FIT preparation not complete; no cold failed-workload retry'
  if split=='val':
   import json
   failed=T/'outputs/native_onpolicy_arm_collection_v2/val_000';fr=mapped(read(failed/'execution_state.json')['root']);actual=read(failed/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
   resource=fr/'attempt/resources.jsonl';rows=[json.loads(line)for line in resource.read_text().splitlines()];assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']=rows[-1]['cleanup_seconds_after_start'];assert v['inherited_stage_GPU_seconds']==281.788
   previous.extend([resource,failed/'actual_host_wait_exit.json',fr/'failure.json'])
   failed4=T/'outputs/native_onpolicy_arm_collection_v4/val_000';fr4=mapped(read(failed4/'execution_state.json')['root']);x=read(failed4/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res4=fr4/'attempt/resources.jsonl';rr=[json.loads(line)for line in res4.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-568.543)<1e-8
   assert cached['cached_valid_cases']>=9 and cached['no_policy_outputs_masks_or_labels_imported']
   previous.extend([res4,failed4/'actual_host_wait_exit.json',fr4/'failure.json'])
   failed5=T/'outputs/native_onpolicy_arm_collection_v5/val_000';fr5=mapped(read(failed5/'execution_state.json')['root']);x=read(failed5/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res5=fr5/'attempt/resources.jsonl';rr=[json.loads(line)for line in res5.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-1608.243)<1e-8
   assert cached['cached_valid_cases']>=15
   previous.extend([res5,failed5/'actual_host_wait_exit.json',fr5/'failure.json'])
   failed6=T/'outputs/native_onpolicy_arm_collection_v6/val_000';fr6=mapped(read(failed6/'execution_state.json')['root']);x=read(failed6/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res6=fr6/'attempt/resources.jsonl';rr=[json.loads(line)for line in res6.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-2693.686)<1e-8
   sig6=fr6/'signal_received_21.json';assert read(sig6)['signal']==2 and read(sig6)['sender_unknown']
   assert cached['cached_valid_cases']>=17
   previous.extend([res6,failed6/'actual_host_wait_exit.json',sig6,fr6/'failure.json'])
   failed7=T/'outputs/native_onpolicy_arm_collection_v7/val_000';fr7=mapped(read(failed7/'execution_state.json')['root']);x=read(failed7/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res7=fr7/'attempt/resources.jsonl';rr=[json.loads(line)for line in res7.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-2979.083)<1e-8
   previous.extend([res7,failed7/'actual_host_wait_exit.json',fr7/'failure.json'])
   failed8=T/'outputs/native_onpolicy_arm_collection_v8/val_000';fr8=mapped(read(failed8/'execution_state.json')['root']);x=read(failed8/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res8=fr8/'attempt/resources.jsonl';rr=[json.loads(line)for line in res8.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-3852.288)<1e-8
   assert 'No actual eligible endpoint pair' in read(fr8/'failure.json')['traceback']and (fr8/'SIGINT_kernel_receipts.bin').stat().st_size==0
   previous.extend([res8,failed8/'actual_host_wait_exit.json',fr8/'failure.json'])
   failed9=T/'outputs/native_onpolicy_arm_collection_v9/val_000';fr9=mapped(read(failed9/'execution_state.json')['root']);x=read(failed9/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res9=fr9/'attempt/resources.jsonl';rr=[json.loads(line)for line in res9.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-5232.506)<1e-8
   assert 'Per-process cumulative/rolling GPU utilization' in read(fr9/'failure.json')['traceback']and (fr9/'SIGINT_kernel_receipts.bin').stat().st_size==0
   assert cached['cached_valid_cases']>=20
   previous.extend([res9,failed9/'actual_host_wait_exit.json',fr9/'failure.json'])
   failed10=T/'outputs/native_onpolicy_arm_collection_v10/val_000';fr10=mapped(read(failed10/'execution_state.json')['root']);x=read(failed10/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res10=fr10/'attempt/resources.jsonl';rr=[json.loads(line)for line in res10.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-6760.585)<1e-8
   sig10=fr10/'signal_received_21.json';assert read(sig10)['signal']==2 and read(fr10/'attempt/barrier/normal_child_exit.json')['exit_code']==1
   receipt10=fr10/'SIGINT_kernel_receipts.bin';assert receipt10.stat().st_size==40
   assert cached['cached_valid_cases']==25
   previous.extend([res10,failed10/'actual_host_wait_exit.json',sig10,receipt10,fr10/'failure.json'])
   failed11=T/'outputs/native_onpolicy_arm_collection_v11/val_000';fr11=mapped(read(failed11/'execution_state.json')['root']);x=read(failed11/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res11=fr11/'attempt/resources.jsonl';rr=[json.loads(line)for line in res11.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-7667.655)<1e-8
   assert 'Conservative ongoing server rule4 cannot be verified' in read(fr11/'failure.json')['traceback']and (fr11/'SIGINT_kernel_receipts.bin').stat().st_size==0
   previous.extend([res11,failed11/'actual_host_wait_exit.json',fr11/'failure.json'])
   failed13=T/'outputs/native_onpolicy_arm_collection_v13/val_000';fr13=mapped(read(failed13/'execution_state.json')['root']);x=read(failed13/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res13=fr13/'attempt/resources.jsonl';rr=[json.loads(line)for line in res13.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-7954.392)<1e-8
   assert 'Sustained GPU utilization/VRAM threshold would not be met' in read(fr13/'failure.json')['traceback']and (fr13/'SIGINT_kernel_receipts.bin').stat().st_size==0
   previous.extend([res13,failed13/'actual_host_wait_exit.json',fr13/'failure.json'])
   failed14=T/'outputs/native_onpolicy_arm_collection_v14/val_000';fr14=mapped(read(failed14/'execution_state.json')['root']);x=read(failed14/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res14=fr14/'attempt/resources.jsonl';rr=[json.loads(line)for line in res14.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-8236.662)<1e-8
   assert 'Sustained GPU utilization/VRAM threshold would not be met' in read(fr14/'failure.json')['traceback']and (fr14/'SIGINT_kernel_receipts.bin').stat().st_size==0
   previous.extend([res14,failed14/'actual_host_wait_exit.json',fr14/'failure.json'])
  else:
   failed=T/'outputs/native_onpolicy_arm_collection_v2/fit_000';fr=mapped(read(failed/'execution_state.json')['root']);actual=read(failed/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
   assert not(fr/'attempt/barrier/permit.json').exists()and not(fr/'attempt/resources.jsonl').exists()
   v['inherited_stage_GPU_seconds']=0.;previous.extend([failed/'actual_host_wait_exit.json',fr/'CPU_admission_race.json',fr/'source_bound_CPU_preload_proof.json'])
  v['future_preparation_each_task']=2 if split=='val' or number==0 else 0
  v['future_preparation_population']=dict(split='fit',seed_start=7124000,first_index=0) if split=='val' else None
  v['pure_preparation_overlap_only_for_predeclared_remaining_FIT']=True
  v['no_legacy_all_cold_qualification_claim']=True

  for directory in(C,T/'native_sigint_receipt_v2',T/'native_nvml_query_order_v3',T/'native_foreign_gpu_contention_v1',T/'native_semantic_budget_pool_v1',T/'native_semantic_host_audit_v1',T/'native_q0_handoff_atomic_v1',T/'native_q0_policy_overlap_v1',T/'native_arm_phase_feedback_v1',T/'native_onpolicy_arm_teacher_v1',T/'native_server_rule4_v10',T/'native_server_rule4_v11',T/'native_retained_cpu_wait_v1'):
   for f in directory.rglob('*'):
    if f.is_file()and f.suffix in('.py','.sh','.md'):v['source_sha256'][str(f.relative_to(P))]=sha(f)
  files=[T/'outputs/native_server_rule4_v11/CPU_actual_scope_checks_v4.json',T/'outputs/native_server_rule4_v11/CPU_actual_host_exit_v4.json',T/'outputs/native_nvml_query_order_v3/CPU_actual_query_order_checks_v1.json',T/'outputs/native_foreign_gpu_contention_v1/CPU_actual_collision_checks_v1.json',T/'outputs/native_semantic_budget_pool_v1/CPU_production_arrival_checks_v1.json',T/'outputs/native_semantic_budget_pool_v1/CPU_actual_host_exit_v1.json',T/'outputs/native_semantic_host_audit_v1/CPU_actual_host_path_checks_v1.json',T/'outputs/native_semantic_host_audit_v1/CPU_actual_host_exit_v1.json',T/'outputs/native_q0_handoff_atomic_v1/CPU_actual_lock_handoff_checks_v1.json',T/'outputs/native_q0_handoff_atomic_v1/CPU_actual_host_exit_v1.json',T/'outputs/native_onpolicy_calibration_data_v17/CPU_reader_regression_v2.json',T/'outputs/native_onpolicy_calibration_data_v17/CPU_actual_host_exit_v2.json',T/'outputs/native_onpolicy_collection_v25/CPU_collection_contract_checks_v3.json',T/'outputs/native_onpolicy_collection_v25/CPU_collection_actual_host_exit_v3.json',T/'outputs/native_server_rule4_v10/CPU_actual_C27_observer_history_checks_v1.json',T/'outputs/native_server_rule4_v10/CPU_actual_host_exit_v1.json',checks,T/'outputs/native_onpolicy_arm_collection_v35/CPU_actual_host_exit_v2.json',mechanism/'actual_host_wait_exit.json',mechanism/'mechanism_complete.json',parent/'CPU_full_capture_audit.json',parent/'terminal_audit.json',T/'outputs/native_arm_phase_feedback_v1/CPU_integration_checks.json',T/'outputs/native_onpolicy_arm_teacher_v1/CPU_legacy40_packet_parity.json',T/'outputs/native_server_rule4_v6/CPU_actual_C16_history_handoff_checks.json',*previous]
  for f in files:v['input_sha256'][str(f)]=sha(f)
  assert coldspec['q0_once_policy_overlaps_initial_CPU_semantics_before_any_action']
  assert coldspec['expert_q0_detach_under_original_lock_before_future_screen']
  v['expert_q0_detach_under_original_lock_before_future_screen']=True
  v['q0_once_policy_overlaps_initial_CPU_semantics_before_any_action']=True
  v['verified_native_q0_releases_remaining_real_lanes_before_CPU_semantic_wait']=True
  v['cached_q0_semantics_lazy_revalidation']=True
  v['cached_q0_regenerated_from_actual_current_native_scene']=False
  v['full_fixed_population']=dict(original_FIT_each_task=40,original_VAL_each_task=10,original_FIT_seed_start=7120000,original_VAL_seed_start=7121000,additional_FIT_each_task=10,additional_VAL_each_task=5,additional_FIT_seed_start=7124000,additional_VAL_seed_start=7125000)
  v['CPU_semantic_adaptive_total8_threads']=True
  v['original250_qualified_with_pure_preparation_reuse_not_legacy_cold_proof']=True
  CPUfailed=T/'outputs/native_onpolicy_arm_collection_v12/val_000';actual=read(CPUfailed/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  roots=list((CPUfailed/'batches/000/runs').iterdir());assert len(roots)==3
  CPUfiles=[CPUfailed/'actual_host_wait_exit.json',CPUfailed/'campaign_failure.json']
  for prior in roots:
   assert read(prior/'CPU_only_replan_qualified.json')['passed']and not(prior/'attempt/barrier/permit.json').exists()and not(prior/'attempt/resources.jsonl').exists()
   CPUfiles.extend(prior/name for name in ['protocol.json','CPU_admission_race.json','CPU_only_replan_qualified.json','CPU_admission_prepared_exit.json','semantic_exit.json','source_bound_CPU_preload_proof.json'])
  v['input_sha256'].update({str(p):sha(p)for p in CPUfiles})
  v['CPU_placement_restored_original_after_actual_load_aware_failure']=True
  v['GPU0_temporarily_excluded_after_two_actual_first280_failures']=True
  v['preferred_GPU_indices']=[1,6,5,4,2,3]
  v['CPU_preloaded_process_retained_while_waiting_same_selected_card']=True
  v['previous_ARM12_CPU_only_GPU_seconds']=0
  v['server_rule4_strict_admission_original_and_later_jobs_scope_v11']=True
  v['own_actual_native_ARM_workload_qualification_required']=True;v['own_actual_cold_ARM_workload_qualification_required']=False
  if split=='val':
   failed15=T/'outputs/native_onpolicy_arm_collection_v15/val_000';fr15=mapped(read(failed15/'execution_state.json')['root']);x=read(failed15/'actual_host_wait_exit.json');assert x['exit_code']==1 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
   res15=fr15/'attempt/resources.jsonl';rr=[json.loads(line)for line in res15.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   assert abs(v['inherited_stage_GPU_seconds']-8236.662)<1e-8
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-8522.780)<1e-8
   v['input_sha256'].update({str(p):sha(p)for p in [res15,failed15/'actual_host_wait_exit.json',fr15/'failure.json']})
  marker=T/'outputs/native_onpolicy_full_ops_v2/ARM16_CPU_no_CUDA_v1.json';m=read(marker);assert m['no_CUDA_permit']and m['actual_GPU_samples']==0 and m['actual_GPU_seconds']==0;verify(m)
  v['input_sha256'][str(marker)]=sha(marker)
  v['source_sha256'].update({str(p):sha(p)for p in (T/'native_admission_commit_margin_v1').glob('*.py')})
  if split=='val':v['inherited_stage_GPU_seconds']+=m['conservative_guard_bookkeeping_seconds'];assert abs(v['inherited_stage_GPU_seconds']-8524.096)<1e-8
  v['CPU_prepermit_original55percent_with_extra8seconds_unknown_zero_margin']=True
  if split=='val':
   failed17=T/'outputs/native_onpolicy_arm_collection_v17/val_000';fr17=mapped(read(failed17/'execution_state.json')['root']);actual17=read(failed17/'actual_host_wait_exit.json');assert actual17['exit_code']==1 and actual17['actual_Popen_wait_return']and actual17['proc_absent_after_wait']
   resource17=fr17/'attempt/resources.jsonl';rr=[json.loads(line)for line in resource17.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rr[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-10723.073)<1e-6
   v['input_sha256'].update({str(p):sha(p)for p in [resource17,failed17/'actual_host_wait_exit.json',fr17/'failure.json',budget_file]})
  failed18=T/'outputs/native_onpolicy_arm_collection_v18/val_000'
  if split=='val':
   actual18=read(failed18/'actual_host_wait_exit.json');assert actual18['exit_code']==1 and actual18['actual_Popen_wait_return']and actual18['proc_absent_after_wait']
   root18=mapped(read(failed18/'execution_state.json')['root']);resource18=root18/'attempt/resources.jsonl';rows18=[json.loads(line)for line in resource18.read_text().splitlines()];assert rows18[-1]['own_gpu_released']and not rows18[-1]['remaining_own_gpu_pids']
   v['inherited_stage_GPU_seconds']+=rows18[-1]['cleanup_seconds_after_start'];assert abs(v['inherited_stage_GPU_seconds']-11008.947)<1e-6
   v['input_sha256'].update({str(p):sha(p)for p in [resource18,failed18/'actual_host_wait_exit.json',root18/'failure.json']})
  no_GPU_marker=T/'outputs/native_onpolicy_full_ops_v2/ARM20_CPU_only_no_GPU_final_v1.json';no_GPU=read(no_GPU_marker);assert no_GPU['passed']and no_GPU['GPU_seconds']==0;verify(no_GPU);v['input_sha256'][str(no_GPU_marker)]=sha(no_GPU_marker)
  if split=='val':
   for failed_version in (21,22):
    failed=T/('outputs/native_onpolicy_arm_collection_v'+str(failed_version))/'val_000';actual=read(failed/'actual_host_wait_exit.json');assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    failed_root=mapped(read(failed/'execution_state.json')['root']);resources=failed_root/'attempt/resources.jsonl';samples=[json.loads(line)for line in resources.read_text().splitlines()];assert samples[-1]['own_gpu_released']and not samples[-1]['remaining_own_gpu_pids']
    v['inherited_stage_GPU_seconds']+=samples[-1]['cleanup_seconds_after_start']
    v['input_sha256'].update({str(p):sha(p)for p in [resources,failed/'actual_host_wait_exit.json',failed_root/'failure.json']})
   assert abs(v['inherited_stage_GPU_seconds']-11579.580)<1e-6
  marker=T/'outputs/native_onpolicy_full_ops_v2/ARM24_CPU_only_no_GPU_final_v1.json';assert read(marker)['passed']and read(marker)['GPU_seconds']==0;verify(read(marker));v['input_sha256'][str(marker)]=sha(marker)
  v['early_preparation_wait_seconds']=.05
  if split=='val':
   marker=T/'outputs/native_onpolicy_full_ops_v2/ARM25_actual_first280_failure_v1.json';m=read(marker);assert m['passed']and m['own_GPU_released'];verify(m);v['inherited_stage_GPU_seconds']+=m['actual_GPU_seconds'];assert abs(v['inherited_stage_GPU_seconds']-11862.916)<1e-6;v['input_sha256'][str(marker)]=sha(marker)
  parity_file=T/'outputs/native_onpolicy_full_ops_v2/render_parity_v1/software_parity_report.json'
  parity=read(parity_file);assert parity['software_parity_passed']and parity['own_GPU_released']and not parity['scientific_qualification'];verify(parity)
  v['input_sha256'][str(parity_file)]=sha(parity_file)
  if split=='val':
   marker26=T/'outputs/native_onpolicy_full_ops_v2/ARM26_actual_first280_failure_v1.json';value26=read(marker26);assert value26['passed']and value26['own_GPU_released'];verify(value26)
   profile_file=T/'outputs/native_onpolicy_full_ops_v2/startup_profile_v1/engineering_profile_report.json';profile=read(profile_file);assert profile['engineering_profile_completed']and profile['own_GPU_released'];verify(profile)
   v['inherited_stage_GPU_seconds']+=value26['actual_GPU_seconds']+profile['actual_GPU_seconds']+parity['actual_GPU_seconds']
   assert abs(v['inherited_stage_GPU_seconds']-12409.593)<1e-6
   v['input_sha256'].update({str(p):sha(p)for p in [marker26,profile_file]})
  front_file=T/'outputs/native_onpolicy_full_ops_v2/render_parity_front_v1/software_parity_report.json';front=read(front_file);assert front['software_parity_passed']and front['own_GPU_released'];verify(front)
  storage_file=T/'outputs/native_object_state_storage_v2/actual_report.json';storage=read(storage_file);assert storage['passed']and storage['own_GPU_released'];verify(storage)
  storage_wait=storage_file.with_name('CPU_GPU_host_actual_wait_v1.json');assert read(storage_wait)['exit_code']==0
  if split=='val':v['inherited_stage_GPU_seconds']=front['cumulative_VAL_seconds']
  failure33=T/'outputs/native_onpolicy_full_ops_v2/ARM33_actual_first280_failure_v1.json';prior=read(failure33);verify(prior)
  assert prior['release']['own_gpu_released']and not prior['release']['remaining_own_gpu_pids']
  if split=='val':v['inherited_stage_GPU_seconds']=prior['cumulative_VAL_GPU_seconds']
  v['task_interleaved_original_cached_prefix_order']=True
  v['input_sha256'][str(failure33)]=sha(failure33)
  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_balanced_task_schedule_v1').glob('*')if p.is_file()})
  v['object_state_storage_CPU_whole_original_raw_controls']=True
  v['startup_release_after_actual_once_query_not_passive']=True
  v['input_sha256'].update({str(p):sha(p)for p in [front_file,storage_file,storage_wait]})
  v['source_sha256'].update({str(p):sha(p)for folder in ['native_used_cameras_v1','native_object_state_storage_v2','native_storage_capture_v1']for p in(T/folder).rglob('*')if p.is_file()and p.suffix in('.py','.md','.sh')})
  v['unread_capture_whole_reference_each_case']=True
  v['every_original_actual_get_obs_update_random_draw_preserved']=True
  v['source_sha256'].update({str(p):sha(p)for folder in ['native_discard_capture_v1','native_unread_capture_full_v1']for p in (T/folder).glob('*')if p.is_file()})
  v['future_preparation_each_task']=0
  v['next_fixed_population_preparation_deferred_to_its_own_stage']=True
  v['startup_release_requires_actual_complete_initial_native32']=False
  v['independent_required_next_population_preparation']=True
  v['early_fixed_next_population_preparation']=bool(v.get('future_preparation_population'))
  v['minimum_actual_current_queries_before_early_preparation']=9
  helper=T/'outputs/native_early_next_preparation_v1/CPU_scope_checks_v1.json';assert read(helper)['passed'];verify(read(helper))
  helper_exit=helper.with_name('CPU_actual_host_exit_v3.json');actual=read(helper_exit);assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  v['input_sha256'].update({str(p):sha(p)for p in [helper,helper_exit]})
  v['source_sha256'].update({str(p):sha(p)for p in (T/'native_early_next_preparation_v1').glob('*')if p.is_file()})
  v['governor_extra_margin_required']=True
  step_file=P/'outputs/evaluations/structured_action/beat_block_hammer/L0/eval_20261003_164626_759241/robotwin/task_config/_eval_step_limit.yml'
  caps={name:int(value)for line in step_file.read_text().splitlines()if line.strip()for name,value in [line.split(':',1)]};v['offline_scheduling_step_caps']={task:int(caps[task])for task in TASKS}
  v['input_sha256'][str(step_file)]=sha(step_file)
  evidence=T/'outputs/native_onpolicy_full_ops_v2/governor_SIGINT_confirmed_20261105_v1.json';assert read(evidence)['confirmed_governor_SIGINT_for_ARM17_and_ARM10'];verify(read(evidence));v['input_sha256'][str(evidence)]=sha(evidence)
  margin_proof=T/'outputs/native_governor_margin_v2/CPU_live_key_checks_v1.json';assert read(margin_proof)['passed'];verify(read(margin_proof));v['input_sha256'][str(margin_proof)]=sha(margin_proof)
  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_governor_margin_v2').glob('*')if p.is_file()})
  signal_proof=T/'outputs/native_signal_event_wait_v1/checks_v2/CPU_actual_signal_wait_checks_v1.json';signal_value=read(signal_proof);assert signal_value['passed'];verify(signal_value)
  signal_actual=T/'outputs/native_signal_event_wait_v1/CPU_actual_host_exit_v1.json';actual=read(signal_actual);assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  v['input_sha256'].update({str(p):sha(p)for p in [signal_proof,signal_actual]})
  v['source_sha256'].update({str(p):sha(p)for p in (T/'native_signal_event_wait_v1').glob('*')if p.is_file()})
  v['future_preparation_uses_unfinished_whole_current_cases']=True
  v['future_preparation_minimum_unfinished_current_cases']=4
  v['scheduling_only_revision']['maximum_foreground_before_required_future']=1
  v['signal_only_parent_wait_not_syscall_tracing']=False
  transfer_proof=T/'outputs/native_oneway_model_transfer_v1/CPU_contract_checks_v1.json';assert read(transfer_proof)['passed'];verify(read(transfer_proof))
  v['input_sha256'][str(transfer_proof)]=sha(transfer_proof)
  v['source_sha256'].update({str(p):sha(p)for p in (T/'native_oneway_model_transfer_v1').glob('*.py')})
  v['original_model_transfer_queued_once_completion_fence_before_return']=True
  readonly=T/'outputs/native_onpolicy_full_ops_v2/GPU5_readonly_20261105_v1/56b1407e8c0d43da9f0cb1171fcb2c47/result.json';e=read(readonly);assert e['passed']and e['metadata_only'];verify(e)
  marker=T/'outputs/native_onpolicy_full_ops_v2/ARM3_CPU_only_switch_v1.json';assert read(marker)['all_no_GPU_permit_or_resource_file']and read(marker)['GPU_seconds']==0
  actual3=T/'outputs/native_onpolicy_arm_collection_v3/val_000/actual_host_wait_exit.json';x=read(actual3);assert x['exit_code']==-2 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']
  v['input_sha256'].update({str(p):sha(p)for p in [readonly,marker,actual3]})
  v['quarantined_unreliable_PID_sensor_UUIDs']=[]
  v['historical_GPU5_PID_missing_preserved_NOT_reclassified']=True
  v['GPU5_only_readonly_metadata_passed_not_active_PID_qualification']=True
  v['GPU5_active_PID_resource_and_normal_exit_require_original_full_guard']=True
  v['scheduling_only_revision']=dict(v['scheduling_only_revision'],current_recovery_priority_over_future=False,current_screen_burst_maximum2=True,current_recovery_nonstarvation=True,maximum_foreground_before_required_future=1,future_tail_rechecked_inside_original_serial_lock=True,no_planner_concurrency=True)
  traceproof=T/'outputs/native_sigint_receipt_v2/checks_v1/CPU_actual_signal_checks_v1.json';checktrace=read(traceproof);assert checktrace['passed'];verify(checktrace)
  v['input_sha256'].update({str(p):sha(p)for p in [traceproof,T/'native_sigint_receipt_v2/libreceipt.so',T/'native_sigint_receipt_v2/receipt.c']})
  v['SIGINT_diagnostics_only_no_injection_or_ignoring']=True
  v['external_signal_sender_PID_can_be0_due_to_namespace_NOT_guessed']=True
  v['stage_total_GPU_budget_seconds']=budget['approved_total_seconds']if split=='val'else 7200
  v['single_continuous_GPU_seconds_cap']=7200
  v['cumulative_budget_amendment']=dict(original_VAL_total_seconds=10800,new_VAL_total_seconds=budget['approved_total_seconds'],all_failed_seconds_retained=v['inherited_stage_GPU_seconds'],explicit_authorization_file=str(budget_file),no_resource_or_scientific_limit_change=True)if split=='val'else None
  v['fresh_valid_NOT_FOUND_unknown_interval_zero_bound_v2']=True
  v['resource_guard_revision']='fresh_validated_NOT_FOUND_dense_clock_entire_unknown_interval_zero_lower_bound_v2'
  failure34=T/'outputs/native_onpolicy_full_ops_v2/ARM34_actual_resource_result_v1.json';prior=read(failure34);verify(prior)
  assert prior['release']['own_gpu_released']and not prior['release']['remaining_own_gpu_pids']
  if split=='val':v['inherited_stage_GPU_seconds']=prior['cumulative_VAL_GPU_seconds']
  if split=='val':v['cumulative_budget_amendment']['all_failed_seconds_retained']=v['inherited_stage_GPU_seconds']
  v['future_preparation_each_task']=1 if split=='val' or number==0 else 0
  v['carried_next_FIT0_scope']=True
  v['carried_FIT0_input_proof']=str(T/'outputs/native_fixed_FIT_inputs_v1/verified_old_FIT0_prefix_v1.json')if split=='val'else None
  if split=='val':
   from native_carried_next_FIT_v1.carry import proof as carried_proof
   carried_file,carried=carried_proof(v);v['input_sha256'][str(carried_file)]=sha(carried_file);v['input_sha256'].update(carried['input_sha256'])
  v['input_sha256'][str(failure34)]=sha(failure34)
  v['source_sha256'].update({str(p):sha(p)for p in(T/'native_carried_next_FIT_v1').glob('*')if p.is_file()})
  from native_unread_capture_full_v1.budget import require_planned_full_batch
  require_planned_full_batch(v)
  v['identity']=identity(v);verify(v);write(O/'study.json',v);return v

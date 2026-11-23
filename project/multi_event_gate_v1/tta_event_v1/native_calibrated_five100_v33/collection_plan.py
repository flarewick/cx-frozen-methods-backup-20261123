"""Official seed0, unseen language, frozen confirmed candidate, no EVAL tuning."""
import fcntl
from .common import T,P,C,O,COHORT,TASKS,CANON,read,write,sha,identity,mapped,verify,stage_folder,PRIMARY,PREPARED_RECOVERY,segment_folder,decoded_stage
from native_calibrated_validation_v24.collection_plan import actual_zero,calibrated_qualification


def confirmed_candidate():
 file=T/'outputs/native_calibrated_validation_v24/confirmation_complete.json'
 report=read(file);verify(report)
 assert report['complete'] and report['all_actual_host_wait0'] and report['interface_qualified']
 # The historical net=-1 remains negative; latest user explicitly authorizes measuring100.
 from native_calibrated_official_authorization_v1.authority import require
 authorization=require(report,report['candidate_checkpoint_sha256'])
 assert report['no_confirmation_tuning'] and report['formal_EVAL_not_used']
 qualification,q=calibrated_qualification()
 assert report['candidate_checkpoint_sha256']==q['training_checkpoint_sha256']
 return file,qualification,q


def freeze_stage():
 O.mkdir(parents=True,exist_ok=True)
 with(O/'study_freeze.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if(O/'study.json').exists():value=read(O/'study.json');verify(value);return value
  number,part,count=decoded_stage(COHORT)
  proof=T/'outputs/native_calibrated_five100_v33/CPU_contract_checks_v1.json'
  checks=read(proof);assert checks['passed'];verify(checks)
  CPUexit=proof.with_name('CPU_actual_host_exit_v1.json');actual_zero(CPUexit)
  confirmation,qualification,q=confirmed_candidate()
  parent=T/'outputs/native_calibrated_validation_v24/confirmation_001';actual_zero(parent/'actual_host_wait_exit.json')
  root=mapped(read(parent/'execution_state.json')['root']);assert read(root/'terminal_audit.json')['passed']
  value=dict(read(parent/'study.json'));value.pop('identity')
  value['source_sha256']=dict(value['source_sha256']);value['input_sha256']=dict(value['input_sha256'])
  initial={t:dict(next_seed=100000,accepted=0)for t in TASKS};previous=[]
  if number:
   prior=segment_folder(number,0)if part==1 else (stage_folder(number-1)if number==7 else segment_folder(number-1,1));actual_zero(prior/'actual_host_wait_exit.json')
   priorroot=mapped(read(prior/'execution_state.json')['root']);protocol=read(priorroot/'protocol.json');verify(protocol)
   assert protocol['candidate_checkpoint_sha256']==q['training_checkpoint_sha256']
   assert read(priorroot/'terminal_audit.json')['passed']
   initial=read(priorroot/'screening_complete.json')['state'];assert all(v['accepted']==number*5+(3 if part else 0)for v in initial.values())
   previous=[prior/'actual_host_wait_exit.json',priorroot/'protocol.json',priorroot/'screening_complete.json']
  value.update(version='native_calibrated_five100_v33',scope='calibrated_five_task_official_seed0_paired100',
   scene_role='frozen_official_EVAL_no_fit',stage=COHORT,dataset_split='official_EVAL_forbidden_for_fit',
   official_seed=0,actual_seed_start=100000,max_screen_attempts_per_task=1000,initial_screen_state=initial,
   episodes_per_task=number*5+(3 if part==0 else 5),simulation_lanes=8,cases_per_batch=count*5,task_slots=list(TASKS)*count,
   logical_batch_count=20,expected_primary_episodes=1500,expected_zero_shadow_episodes=10,
   expected_engineering_reference_episodes=165,official_unseen_language=True,
   future_preparation_each_task=0,inherited_stage_GPU_seconds=0.,stage_total_GPU_budget_seconds=7200,
   official_EVAL_used=True,formal_success_not_used_for_fitting=True,official_EVAL_excluded_from_fitting=True,
   candidate_checkpoint_sha256=q['training_checkpoint_sha256'],trained_qualification=str(CANON/qualification.relative_to(P)))
  for f in C.rglob('*'):
   if f.is_file()and f.suffix in('.py','.sh','.md'):value['source_sha256'][str(f.relative_to(P))]=sha(f)
  for f in[proof,CPUexit,confirmation,qualification,parent/'actual_host_wait_exit.json',root/'terminal_audit.json',*previous]:value['input_sha256'][str(f)]=sha(f)
  for helper in ('native_signal_event_wait_v1','native_oneway_model_transfer_v1'):
   value['source_sha256'].update({str(p):sha(p)for p in (T/helper).glob('*')if p.is_file()})
  for proof in (T/'outputs/native_signal_event_wait_v1/checks_v2/CPU_actual_signal_wait_checks_v1.json',T/'outputs/native_signal_event_wait_v1/CPU_actual_host_exit_v1.json',T/'outputs/native_oneway_model_transfer_v1/CPU_contract_checks_v1.json'):
   checked=read(proof);verify(checked)
   if 'exit_code'in checked:assert checked['exit_code']==0 and checked['actual_Popen_wait_return']and checked['proc_absent_after_wait']
   else:assert checked['passed']
   value['input_sha256'][str(proof)]=sha(proof)
  value['signal_only_parent_wait_not_syscall_tracing']=False
  value['governor_extra_margin_required']=False
  value['documented_server_policy_user34_without_extra35_36']=False
  value['current_server_policy']='governor180_v15'
  value['required_reference_before_primary_tail']=True
  value['fixed_task_and_branch_scheduling_only']='group_all_switch_then_laptop_first_full_reference_TTA_baseline'
  value['resource_limits']=dict(server_window_seconds=180,minimum_utilization_by_idle_GPU=[35,30,25],minimum_mean_VRAM_MiB_by_idle_GPU=[4096,3072,2048],extra_utilization_by_idle_GPU=[70,60,50],extra_mean_VRAM_percent_by_idle_GPU=[50,30,20],GPU_VRAM_cap_MiB=24563,CPU_percent=[40,1600],maximum_CPU_cores=16,maximum_RSS_GiB=128,maximum_account_GPUs=3,attempt_seconds=7200,maximum_process_lifetime_seconds=129600)
  from native_governor180_v2.observer import verify_reference
  verify_reference()
  for p in(T/'native_governor180_v2').glob('*'):
   if p.is_file():value['source_sha256'][str(p)]=sha(p)
  import json
  value['source_sha256'].update(json.loads((T/'native_governor180_v2/reference.json').read_text())['source_sha256'])
  value['server_rules_sha256']=sha('/home/qiufw/服务器使用注意事项.md')
  value['input_sha256'][str(T/'outputs/native_calibrated_official_progress_v1/qualified_065.json')]=sha(T/'outputs/native_calibrated_official_progress_v1/qualified_065.json')
  failed_CPU=T/'outputs/native_calibrated_five100_v26/official_013_p1'
  a=read(failed_CPU/'actual_host_wait_exit.json');assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
  assert not(failed_CPU/'execution_state.json').exists()and not list((failed_CPU/'batches').glob('*/runs/*'))
  for p in(failed_CPU/'actual_host_wait_exit.json',failed_CPU/'campaign_failure.json'):value['input_sha256'][str(p)]=sha(p)
  value['predecessor_CPU_only_no_GPU_elapsed']=True
  from native_calibrated_official_authorization_v1.authority import require,AUTH
  value['input_sha256'].update(require(read(confirmation),q['training_checkpoint_sha256']))
  value['explicit_user_authorized_official100_despite_confirmation_net_minus1']=True
  value['positive_confirmation_NOT_claimed']=True
  value['cached_preparation']=dict(cached_valid_cases=0,task_cache={},requests=[],case_request_keys={},case_diagnostics={},expert_reference_files={},input_sha256={},new_automatic_recovery_attempts=0)
  value['future_preparation_population']=None
  value['source_sha256'].update({str(p):sha(p)for p in(T/'native_calibrated_official_authorization_v1').glob('*.py')})
  value['logical_batch_index']=number;value['memory_segment_index']=part
  value['fixed_memory_maintenance_segments_per_remaining_logical_batch']=[3,2]
  from .oidn_carry import carry as OIDN_carry
  old_elapsed,old_bindings=OIDN_carry();value['input_sha256'].update(old_bindings)
  if number==7 and part==0:value['inherited_stage_GPU_seconds']=old_elapsed
  if part==1:
   import json
   records=[json.loads(s)for s in(priorroot/'attempt/resources.jsonl').read_text().splitlines()]
   assert records[-1]['own_gpu_released']and not records[-1]['remaining_own_gpu_pids']
   value['inherited_stage_GPU_seconds']=protocol['inherited_stage_GPU_seconds']+records[-1]['cleanup_seconds_after_start']
   value['input_sha256'][str(priorroot/'attempt/resources.jsonl')]=sha(priorroot/'attempt/resources.jsonl')
  if(number,part)==(19,1):
   from .rule8_resume import evidence
   total,bindings,failed_protocol=evidence()
   assert failed_protocol['initial_screen_state']==initial
   value['inherited_stage_GPU_seconds']=total;value['input_sha256'].update(bindings)
   value['external_rule8_recovery_attempt']=1
  else:value['external_rule8_recovery_attempt']=0
  if(number,part)==(18,1):
   from .rule8_resume import evidence
   total,bindings,failed_protocol=evidence()
   assert failed_protocol['initial_screen_state']==initial
   value['inherited_stage_GPU_seconds']=total;value['input_sha256'].update(bindings)
   value['required_reference_before_primary_tail']=True
  value['fixed_task_and_branch_scheduling_only']='group_all_switch_then_laptop_first_full_reference_TTA_baseline'
  assert 0<=value['inherited_stage_GPU_seconds']<7200
  failed=None
  if PREPARED_RECOVERY:failed=PRIMARY
  if failed is not None:
   from .cold_resume import evidence
   elapsed,bindings,failed_spec=evidence(failed)
   value['input_sha256'].update(bindings)
   value['inherited_stage_GPU_seconds']=elapsed
  from .preparation import collect
  cached=collect(value,0,initial)
  if failed is not None:
   assert cached and cached['cached_valid_cases']>=8 and set(cached['task_cache'])==set(TASKS)
  if(number,part)==(19,1):assert cached and cached['cached_valid_cases']==9 and set(cached['task_cache'])==set(TASKS)
  if(number,part)==(18,1):assert cached and cached['cached_valid_cases']==10 and set(cached['task_cache'])==set(TASKS)
  if cached:
   value['cached_preparation']=cached
   value['input_sha256'].update(cached['input_sha256'])
  from .cpu_admission import historical_final_admission_failure
  value['input_sha256'].update(historical_final_admission_failure())
  from .cpu_admission import historical_three_CPU_races
  value['input_sha256'].update(historical_three_CPU_races())
  value['one_audited_cold_to_prepared_resume_only']=True
  value['prepared_recovery_attempt']=1 if PREPARED_RECOVERY else 0
  value['qualified_prefix_first4_readonly_from_formal20']=True
  value['failed_policy_results_imported']=False
  value['original_model_transfer_queued_once_completion_fence_before_return']=True
  value['identity']=identity(value);verify(value);write(O/'study.json',value);return value

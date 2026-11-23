"""Freeze the predeclared fresh pools only after actual complete calibration."""
import fcntl
from .common import T,P,C,O,COHORT,TASKS,CANON,read,write,sha,identity,mapped,verify


def actual_zero(path):
 value=read(path)
 assert value['exit_code']==0 and value['actual_Popen_wait_return'] and value['proc_absent_after_wait']


def calibrated_qualification():
 file=T/'outputs/native_fit7_postexit_recovery_v2/qualification.json';q=read(file)
 assert q['postfit_only_recovery']and q['original_training_host_exit_code']==1
 assert q['additional_optimizer_steps_after_original1000']==0 and q['minimum_distinct_FIT_cases']==7
 actual_zero(file.parent/'actual_finalize_host_exit.json')
 assert q['frozen'] and q['actual_training_host_exit0'] and q['actual_updates']==1000
 assert q['new_population_identities']==325 and q['original093_checkpoint_preserved']
 assert set(q['new_offline_training_tasks'])==set(TASKS)
 assert q['independent_action_and_task_benefit_still_required']
 actual_zero(mapped(q['training_host_exit_file']));verify(q)
 assert sha(mapped(q['training_checkpoint']))==q['training_checkpoint_sha256']
 return file,q


def freeze_stage():
 O.mkdir(parents=True,exist_ok=True)
 with(O/'study_freeze.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if(O/'study.json').exists():value=read(O/'study.json');verify(value);return value
  role,number=COHORT.rsplit('_',1);number=int(number)
  assert role=='confirmation'and number in(0,1),'Completed DEV40 is read-only; confirmation continuation only'
  proof=T/'outputs/native_calibrated_validation_v24/CPU_contract_checks_v8.json'
  checks=read(proof);assert checks['passed'];verify(checks)
  CPUexit=proof.with_name('CPU_actual_host_exit_v8.json');actual_zero(CPUexit)
  qualification,q=calibrated_qualification()
  dataset=T/'outputs/native_onpolicy_calibration_data_v60/qualified_dataset.json'
  manifest=read(dataset);assert manifest['complete'] and manifest['actual_build_host_exit0'] and manifest['identities']==325;verify(manifest)
  selected=next(row for row in manifest['stages']if row['population']=='initial_ARM_fixed75'and row['split']=='val')
  root=mapped(selected['root']);parent=root.parents[3];actual_zero(parent/'actual_host_wait_exit.json')
  assert mapped(read(parent/'execution_state.json')['root']).resolve()==root.resolve()
  from native_onpolicy_calibration_data_v60.reader import Reader
  Reader().qualified(root)
  template=T/'outputs/native_onpolicy_arm_collection_v35/val_000/study.json'
  value=dict(read(template));verify(value);value.pop('identity') # Original validation runtime configuration only; no failed trajectories imported.
  value['source_sha256']=dict(value['source_sha256']);value['input_sha256']=dict(value['input_sha256'])
  start=7122000 if role=='development'else 7123000
  initial={t:dict(next_seed=start,accepted=0)for t in TASKS}
  previous=[]
  if number:
   previous_stage=O.parent/(role+'_000');actual_zero(previous_stage/'actual_host_wait_exit.json')
   previous_root=mapped(read(previous_stage/'execution_state.json')['root'])
   assert read(previous_root/'terminal_audit.json')['passed']
   previous_spec=read(previous_root/'protocol.json');verify(previous_spec)
   assert previous_spec['candidate_checkpoint_sha256']==q['training_checkpoint_sha256']
   initial=read(previous_root/'screening_complete.json')['state'];assert all(v['accepted']==4 for v in initial.values())
   previous=[previous_stage/'actual_host_wait_exit.json',previous_root/'screening_complete.json',previous_root/'protocol.json']
  if role=='confirmation':
   development=T/'outputs/native_calibrated_validation_v23/development_complete.json'
   done=read(development);verify(done)
   assert done['complete'] and done['all_actual_host_wait0'] and done['interface_qualified']
   assert done['candidate_checkpoint_sha256']==q['training_checkpoint_sha256']
   assert done['paired_net_success_change']>0,'No independent confirmation expansion without positive development pilot'
   previous.append(development)
  value.update(version='native_calibrated_validation_v24',scope='new_calibrated_native_validation',
   scene_role='new_fixed_'+role+'_TRAIN_no_fit',stage=COHORT,tasks=list(TASKS),
   actual_seed_start=start,initial_screen_state=initial,episodes_per_task=(number+1)*4,
   simulation_lanes=8,cases_per_batch=20,task_slots=list(TASKS)*4,conditions=['baseline','M_event_full','TTA','zero_update','shadow_update'],
   trained_qualification=str(CANON/qualification.relative_to(P)),candidate_checkpoint_sha256=q['training_checkpoint_sha256'],
   future_preparation_each_task=0,inherited_stage_GPU_seconds=0.,stage_total_GPU_budget_seconds=7200,
   all_teacher_and_GT_offline_only=True,validation_cases_excluded_from_fitting=True,
   cold_start_scheduling_not_proven_until_actual_uncached_stage=False,
   initial_ARM_path_feedback=True,q0_once_policy_overlaps_initial_CPU_semantics_before_any_action=True)
  from native_calibrated_eval_resume_v2.admission import carry
  elapsed,bindings=carry()
  if number==0:value['inherited_stage_GPU_seconds']=elapsed
  value['input_sha256'].update(bindings)
  value['source_sha256'].update({str(p):sha(p)for p in(T/'native_calibrated_eval_resume_v2').glob('*.py')})
  from native_calibrated_confirm_resume_v1.carry import carry as confirmation_carry
  prior,previous_bindings=confirmation_carry()
  if number==0:value['inherited_stage_GPU_seconds']=prior
  value['input_sha256'].update(previous_bindings)
  value['source_sha256'].update({str(p):sha(p)for p in(T/'native_calibrated_confirm_resume_v1').glob('*.py')})
  value['cumulative_budget_amendment']=None
  value['single_continuous_GPU_seconds_cap']=7200
  value['CPU_semantic_adaptive_total8_threads']=True
  value['cached_q0_semantics_lazy_revalidation']=False
  value['future_preparation_population']=None
  value['cached_preparation']=dict(cached_valid_cases=0,task_cache={},requests=[],case_request_keys={},case_diagnostics={},expert_reference_files={},input_sha256={},new_automatic_recovery_attempts=0)
  from .preparation import collect
  cached=collect(value,0,initial)
  if number==0:
   assert cached and cached['cached_valid_cases']>=8 and set(cached['task_cache'])==set(TASKS)
  if cached:
   value['cached_preparation']=cached
   value['input_sha256'].update(cached['input_sha256'])
  for f in C.rglob('*'):
   if f.is_file()and f.suffix in('.py','.sh','.md'):value['source_sha256'][str(f.relative_to(P))]=sha(f)
  for f in[proof,CPUexit,qualification,dataset,parent/'actual_host_wait_exit.json',root/'atomic_full_recovery_qualification.json',*previous]:value['input_sha256'][str(f)]=sha(f)
  for helper in ('native_signal_event_wait_v1','native_oneway_model_transfer_v1'):
   value['source_sha256'].update({str(p):sha(p)for p in (T/helper).glob('*')if p.is_file()})
  for proof in (T/'outputs/native_signal_event_wait_v1/checks_v2/CPU_actual_signal_wait_checks_v1.json',T/'outputs/native_signal_event_wait_v1/CPU_actual_host_exit_v1.json',T/'outputs/native_oneway_model_transfer_v1/CPU_contract_checks_v1.json'):
   checked=read(proof);verify(checked)
   if 'exit_code'in checked:assert checked['exit_code']==0 and checked['actual_Popen_wait_return']and checked['proc_absent_after_wait']
   else:assert checked['passed']
   value['input_sha256'][str(proof)]=sha(proof)
  value['signal_only_parent_wait_not_syscall_tracing']=False
  value['governor_extra_margin_required']=False
  value['documented_server_policy_user34_without_extra35_36']=True
  value['exploratory_support7_explicit_user_authorization']=True
  value['validation_runtime_template35_not_current_collector75']=True
  from native_fit_support_review_v1.support import evidence as support_evidence
  value['input_sha256'].update(support_evidence())
  value['input_sha256'][str(template)]=sha(template)
  value['original_model_transfer_queued_once_completion_fence_before_return']=True
  if number==0:
   assert value['cached_preparation']['cached_valid_cases']>=8 and set(value['cached_preparation']['task_cache'])==set(TASKS),'Final frozen cache cannot be overwritten by defaults'
  value['identity']=identity(value);verify(value);write(O/'study.json',value);return value

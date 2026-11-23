"""Official seed0 paired100; freeze before observing any current-stage outcomes."""
import fcntl,json
from pathlib import Path
from .common import T,P,C,O,COHORT,GROUP,INDEX,TASKS,CANON,read,write,sha,identity,mapped,verify
from .configuration import GROUPS,bounds
from .qualification import qualified_stage
from .adoption import folder as stage_folder,check as adoption_check,IMPORTED_PLAIN_SESSIONS

def read_release(file):return json.loads(file.read_text().splitlines()[-1])

def freeze_stage():
 O.mkdir(parents=True,exist_ok=True)
 with(O/'study_freeze.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if(O/'study.json').exists():v=read(O/'study.json');verify(v);return v
  proof=O.parent/'CPU_contract_checks_v1.json';checks=read(proof);verify(checks);assert checks['passed']
  actual=read(proof.with_name('CPU_actual_host_exit_v1.json'));assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  assert GROUP!='plain'or INDEX>=IMPORTED_PLAIN_SESSIONS,'Imported normal prefix must not be repeated'
  adoption_inputs=adoption_check()
  cfg=GROUPS[GROUP];trainfolder=T/'outputs'/cfg['TRAIN_version']/cfg['TRAIN_stage'];trainroot,train=qualified_stage(trainfolder)
  start,end=bounds(INDEX);previous=None
  if INDEX:
   previous=qualified_stage(stage_folder(GROUP,INDEX-1))
   pr,ps=previous;assert ps['group']==GROUP and ps['episodes_per_task']==start and ps['candidate_checkpoint_sha256']==train['candidate_checkpoint_sha256']
   initial=read(pr/'screening_complete.json')['state'];assert all(v['accepted']==start for v in initial.values())
  else:initial={t:dict(next_seed=100000,accepted=0)for t in TASKS}
  v=dict(train);v.pop('identity');v['source_sha256']=dict(train['source_sha256']);v['input_sha256']=dict(train['input_sha256'])
  count=end-start
  for name in('same_six_accepted_cases_locked_by_preparation','same_eight_accepted_cases_locked_by_preparation','inherited_performance_GPU_seconds','performance_total_GPU_budget_seconds','simultaneously_charged_to_TRAIN_and_performance_ledgers','qualified_prior_TRAIN_GPU_seconds','prior_failed_TRAIN_GPU_seconds','TRAIN_GPU_budget_inherits_qualified_plain_validation'):
   v.pop(name,None)
  v.update(version=C.name,scope='seven_task_official_seed0_paired100',stage=COHORT,group=GROUP,stage_index=INDEX,scene_role='official_seed0_unseen_paired100_no_fit',dataset_split='EVAL',official_seed=0,actual_seed_start=100000,initial_screen_state=initial,
   tasks=list(TASKS),task_slots=list(TASKS)*count,episodes_this_stage=count,episodes_per_task=end,cases_per_batch=len(TASKS)*count,simulation_lanes=cfg['lanes'],conditions=['M_event_full','TTA','zero_update','shadow_update'],
   expected_primary_episodes=2*len(TASKS)*count,expected_zero_shadow_episodes=2*len(TASKS),expected_engineering_reference_episodes=len(TASKS),expected_engineering_reference_episodes_maximum_with_regeneration=len(TASKS)*count,
   logical_batch_count=1,official_EVAL_used=True,no_official_EVAL_used=False,new_official_EVAL_scenes=len(TASKS)*count,formal_success_not_used_for_fitting=True,official_EVAL_excluded_from_fitting=True,
   inherited_stage_GPU_seconds=0.,stage_total_GPU_budget_seconds=7200,future_preparation_each_task=0 if end==100 else min(2,100-end),cached_preparation=None,
   functional_validation_not_official100_or_TTA_benefit=False,mandatory_first_case_whole_reference=True,formal_efficacy_not_required_or_preclaimed_for_run=True,old_failed_GPU_seconds_retained=None,
   prospective_optimization='fixed_whole_branch_priority_shared_model_owner_pure_next_expert_q0_preparation_no_added_policy_queries',exact_governor180_backend='native_governor180_v5_observer_Rule8_creation_grace_v1',
   comparison='frozen_full_all_event_v118_gate_vs_same_plus_TTA'if GROUP=='gated'else'frozen_full_vs_same_plus_TTA',early_pairing=GROUP=='plain')
  from .prepared_recovery import collect
  v['cached_preparation']=collect(previous,initial,count)
  extra_failed_seconds=0.
  if GROUP=='plain'and INDEX==1 and IMPORTED_PLAIN_SESSIONS==1:
   from .carry import recovered
   from .adoption import failed_second
   cache,spent,dependencies=recovered(v,initial)
   v['cached_preparation']=cache;v['input_sha256'].update(dependencies)
   extra_failed_seconds=failed_second()[2]['cleanup_seconds_after_start']
   v['retained_failed_C5_second_GPU_seconds']=extra_failed_seconds
   from .adoption import failed_current
   _,_,failed_release,failed_inputs=failed_current()
   v['retained_failed_C6_second_GPU_seconds']=failed_release['cleanup_seconds_after_start']
   extra_failed_seconds=round(extra_failed_seconds+failed_release['cleanup_seconds_after_start'],3)
   v['input_sha256'].update(failed_inputs)
   from .adoption import failed_coordination
   _,_,coord_release,coord_inputs=failed_coordination()
   v['retained_failed_C7_second_GPU_seconds']=coord_release['cleanup_seconds_after_start']
   extra_failed_seconds=round(extra_failed_seconds+coord_release['cleanup_seconds_after_start'],3)
   v['input_sha256'].update(coord_inputs)
  v['input_sha256'].update(adoption_inputs)
  v['prospective_project_budget_authorization']='Prior explicit user permission to increase experiment time budgets; current frozen C5 untouched; bounded future allocation retains all failed and qualified elapsed time, unchanged model, seed0, physics, controls and server rules'
  from .budget import accounting
  if INDEX%2:
   release_file=pr/'attempt/resources.jsonl';release=read_release(release_file)
   v.update(accounting(GROUP,INDEX,previous_spec=ps,previous_release=release,retry_seconds=extra_failed_seconds))
   v['input_sha256'][str(release_file)]=sha(release_file)
  else:v.update(accounting(GROUP,INDEX,initial_failed=v['inherited_stage_GPU_seconds']))
  if v['cached_preparation']:v['input_sha256'].update(v['cached_preparation']['input_sha256'])
  for folder in(C,T/'native_governor180_v5',T/'native_rule8_startup_v1',T/'gated_arm_context_v1'):
   for f in folder.rglob('*'):
    if f.is_file()and f.suffix in('.py','.sh','.md','.json'):v['source_sha256'][str(f.relative_to(P))]=sha(f)
  inputs=[T/'outputs/native_seven_early_queue_scope_v1/CPU_queue_checks_v1.json',T/'outputs/native_seven_early_queue_scope_v1/CPU_actual_exit_v2.json',T/'outputs/native_seven_early_acceptance_v1/acceptance.json',T/'outputs/native_seven_early_acceptance_driver_v1/actual_exit_v1.json',proof,proof.with_name('CPU_actual_host_exit_v1.json'),trainroot/'attempt/resources.jsonl',T/'outputs/native_governor180_v5/CPU_checks_v1.json',T/'outputs/native_governor180_v5/CPU_actual_exit_v1.json',T/'outputs/native_gpu_drained_guard_v4/CPU_checks_v1.json',T/'outputs/native_gpu_drained_guard_v4/CPU_actual_exit_v1.json']
  inputs.extend([T/'outputs/native_rule8_startup_v1/CPU_checks_v1.json',T/'outputs/native_rule8_startup_v1/CPU_actual_exit_v1.json'])
  if GROUP=='gated':inputs.append(trainfolder/'qualification.json')
  else:inputs.extend([trainfolder/'actual_host_wait_exit.json',trainroot/'terminal_audit.json',trainroot/'CPU_episode_audit.json'])
  for folder in(T/'native_gpu_drained_guard_v4',T/'gated_postexit_recovery_v2'):
   for f in folder.glob('*'):
    if f.is_file():v['source_sha256'][str(f)]=sha(f)
  if previous:inputs.extend([pr/'protocol.json',pr/'terminal_audit.json',stage_folder(GROUP,INDEX-1)/'actual_host_wait_exit.json'])
  for f in inputs:v['input_sha256'][str(f)]=sha(f)
  v['identity']=identity(v);verify(v);write(O/'study.json',v);return v

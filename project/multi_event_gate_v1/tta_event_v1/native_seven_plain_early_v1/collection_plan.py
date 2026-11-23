"""Official seed0 paired100; freeze before observing any current-stage outcomes."""
import fcntl
from pathlib import Path
from .common import T,P,C,O,COHORT,GROUP,INDEX,TASKS,CANON,read,write,sha,identity,mapped,verify
from .configuration import GROUPS,bounds
from .qualification import qualified_stage

def freeze_stage():
 assert GROUP=='plain'and INDEX==0,'Only one TRAIN early-dispatch probe; no official or gated campaign'
 O.mkdir(parents=True,exist_ok=True)
 with(O/'study_freeze.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if(O/'study.json').exists():v=read(O/'study.json');verify(v);return v
  proof=O.parent/'CPU_contract_checks_v1.json';checks=read(proof);verify(checks);assert checks['passed']
  actual=read(proof.with_name('CPU_actual_host_exit_v1.json'));assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  cfg=GROUPS[GROUP];trainfolder=T/'outputs'/cfg['TRAIN_version']/cfg['TRAIN_stage'];trainroot,train=qualified_stage(trainfolder)
  start,end=bounds(INDEX);previous=None
  if INDEX:
   previous=qualified_stage(O.parent/(GROUP+'_'+str(INDEX-1).zfill(3)))
   pr,ps=previous;assert ps['group']==GROUP and ps['episodes_per_task']==start and ps['candidate_checkpoint_sha256']==train['candidate_checkpoint_sha256']
   initial=read(pr/'screening_complete.json')['state'];assert all(v['accepted']==start for v in initial.values())
  else:initial={t:dict(next_seed=7136000,accepted=0)for t in TASKS}
  v=dict(train);v.pop('identity');v['source_sha256']=dict(train['source_sha256']);v['input_sha256']=dict(train['input_sha256'])
  count=end-start
  for name in('same_six_accepted_cases_locked_by_preparation','same_eight_accepted_cases_locked_by_preparation'):
   v.pop(name,None)
  v.update(version=C.name,scope='seven_task_TRAIN_plain_early8',stage=COHORT,group=GROUP,stage_index=INDEX,scene_role='fixed_TRAIN_engineering_q0_dispatch8_no_fit',dataset_split='TRAIN',official_seed=None,actual_seed_start=7136000,initial_screen_state=initial,
   tasks=list(TASKS),task_slots=list(TASKS)*count,episodes_this_stage=count,episodes_per_task=end,cases_per_batch=len(TASKS)*count,simulation_lanes=cfg['lanes'],conditions=['M_event_full','TTA','zero_update','shadow_update'],
   expected_primary_episodes=2*len(TASKS)*count,expected_zero_shadow_episodes=2*len(TASKS),expected_engineering_reference_episodes=len(TASKS),expected_engineering_reference_episodes_maximum_with_regeneration=len(TASKS)*count,
   logical_batch_count=1,official_EVAL_used=False,no_official_EVAL_used=True,new_official_EVAL_scenes=0,formal_success_not_used_for_fitting=True,official_EVAL_excluded_from_fitting=True,
   inherited_stage_GPU_seconds=0.,stage_total_GPU_budget_seconds=7200,future_preparation_each_task=0,cached_preparation=None,
   functional_validation_not_official100_or_TTA_benefit=True,mandatory_first_case_whole_reference=True,formal_efficacy_not_required_or_preclaimed_for_run=True,old_failed_GPU_seconds_retained=None,
   prospective_optimization='fixed_whole_branch_priority_shared_model_owner_pure_next_expert_q0_preparation_no_added_policy_queries',exact_governor180_backend='native_governor180_v4_TTA_owned_observer',
   comparison='frozen_full_all_event_v118_gate_vs_same_plus_TTA'if GROUP=='gated'else'frozen_full_vs_same_plus_TTA')
  from .prepared_recovery import collect
  v['cached_preparation']=collect(previous,initial,count)
  if GROUP=='plain'and INDEX==0:
   from .carry import recovered
   cache,spent,dependencies=recovered(v,initial)
   assert v['cached_preparation']is None
   v['cached_preparation']=cache;v['inherited_stage_GPU_seconds']=spent;v['stage_total_GPU_budget_seconds']=4926.847
   v['inherited_performance_GPU_seconds']=6061.735;v['performance_total_GPU_budget_seconds']=7200
   v['simultaneously_charged_to_TRAIN_and_performance_ledgers']=True
   v['qualified_prior_TRAIN_GPU_seconds']=1555.513;v['prior_failed_TRAIN_GPU_seconds']=2233.069;v['early_pairing']=True
   v['input_sha256'].update(dependencies)
   v['TRAIN_GPU_budget_inherits_qualified_plain_validation']=True
  if v['cached_preparation']:v['input_sha256'].update(v['cached_preparation']['input_sha256'])
  for folder in(C,T/'native_governor180_v4',T/'gated_arm_context_v1'):
   for f in folder.rglob('*'):
    if f.is_file()and f.suffix in('.py','.sh','.md','.json'):v['source_sha256'][str(f.relative_to(P))]=sha(f)
  inputs=[proof,proof.with_name('CPU_actual_host_exit_v1.json'),trainroot/'attempt/resources.jsonl',T/'outputs/native_governor180_v4/CPU_checks_v1.json',T/'outputs/native_governor180_v4/CPU_actual_exit_v1.json',T/'outputs/native_gpu_drained_guard_v2/CPU_checks_v1.json',T/'outputs/native_gpu_drained_guard_v2/CPU_actual_exit_v1.json']
  if GROUP=='gated':inputs.append(trainfolder/'qualification.json')
  else:inputs.extend([trainfolder/'actual_host_wait_exit.json',trainroot/'terminal_audit.json',trainroot/'CPU_episode_audit.json'])
  for folder in(T/'native_gpu_drained_guard_v2',T/'gated_postexit_recovery_v2'):
   for f in folder.glob('*'):
    if f.is_file():v['source_sha256'][str(f)]=sha(f)
  if previous:inputs.extend([pr/'protocol.json',pr/'terminal_audit.json',O.parent/(GROUP+'_'+str(INDEX-1).zfill(3))/'actual_host_wait_exit.json'])
  for f in inputs:v['input_sha256'][str(f)]=sha(f)
  v['identity']=identity(v);verify(v);write(O/'study.json',v);return v

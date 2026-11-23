"""Sequential fixed official batches; resume only an actual qualified prefix."""
import argparse,os,sys,subprocess
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify
from .common import TASKS,stage_folder,segment_folder


def qualified(folder):
 actual=read(folder/'actual_host_wait_exit.json')
 assert actual['exit_code']==0 and actual['actual_Popen_wait_return'] and actual['proc_absent_after_wait']
 root=mapped(read(folder/'execution_state.json')['root']);spec=read(root/'protocol.json');verify(spec)
 assert read(root/'terminal_audit.json')['passed']
 report=read(folder/'pilot_complete.json');assert report['complete']
 return root,spec,report


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--start-stage',type=int,default=0);parser.add_argument('--start-part',type=int,default=1);args=parser.parse_args();assert args.start_part in(0,1);assert 13<=args.start_stage<20
 base=T/'outputs/native_calibrated_five100_v33';complete=base/'official100_complete.json';assert not complete.exists()
 for number in range(7):qualified(stage_folder(number))
 for number in range(7,args.start_stage):
  for part in(0,1):qualified(segment_folder(number,part))
 if args.start_part==1:qualified(segment_folder(args.start_stage,0))
 for number in range(args.start_stage,20):
  for part in(0,1):
   if number==args.start_stage and part<args.start_part:continue
   stage='official_'+str(number).zfill(3)+'_p'+str(part);folder=base/stage
   assert not(folder/'actual_host_wait_exit.json').exists(),'Keep failed actual stage; no unchanged retry'
   process=subprocess.Popen([sys.executable,'-B','-u','-m','native_calibrated_five100_v33.launch'],env=dict(os.environ,CX_ONPOLICY_STAGE=stage))
   code=process.wait()
   if code!=0:
    # Prepared attempts never enter the audited cold-only recovery path.
    from .cold_resume import evidence
    elapsed,bindings,failed_spec=evidence(folder)
    from .preparation import collect
    import importlib
    prep=importlib.import_module('native_calibrated_five100_v33.preparation')
    saved_O,saved_PRIMARY,saved_COHORT=prep.O,prep.PRIMARY,prep.COHORT
    try:
     prep.O=folder/'prepared_recovery_001';prep.PRIMARY=folder;prep.COHORT=stage
     cached=collect(failed_spec,0,failed_spec['initial_screen_state'])
    finally:prep.O,prep.PRIMARY,prep.COHORT=saved_O,saved_PRIMARY,saved_COHORT
    assert cached and cached['cached_valid_cases']>=8 and set(cached['task_cache'])==set(TASKS)
    recovery=folder/'prepared_recovery_001';assert not recovery.exists()
    write(folder/'prepared_recovery_decision.json',dict(only_changed_pure_preparation_configuration=True,maximum_one_per_stage=True,failed_elapsed_seconds=elapsed,remaining_stage_seconds=7200-elapsed,cached_valid_cases=cached['cached_valid_cases'],no_partial_policy_results_imported=True,failed_source_inputs=bindings))
    process=subprocess.Popen([sys.executable,'-B','-u','-m','native_calibrated_five100_v33.launch'],env=dict(os.environ,CX_ONPOLICY_STAGE=stage,CX_OFFICIAL_PREPARED_RECOVERY='1'))
    code=process.wait();assert code==0,'Prepared continuation failed; retained with no further retry'
    folder=recovery
   assert code==0
   root,spec,report=qualified(folder)
   write(base/('qualified_segment_'+str(number).zfill(3)+'_p'+str(part)+'.json'),dict(qualified_logical_batch=number,memory_segment=part,qualified_episodes_each_task=number*5+(3 if part==0 else 5),candidate_checkpoint_sha256=spec['candidate_checkpoint_sha256'],actual_host_exit_file=str(folder/'actual_host_wait_exit.json'),actual_host_exit_sha256=sha(folder/'actual_host_wait_exit.json')))
   print('CALIBRATED_OFFICIAL_PAIRED100_PROGRESS',number,part,number*5+(3 if part==0 else 5),'/100/task',flush=True)
 reports=[];inputs={};checkpoint=None
 folders=[stage_folder(n)for n in range(7)]+[segment_folder(n,p)for n in range(7,20)for p in(0,1)]
 for folder in folders:
  root,spec,report=qualified(folder)
  if checkpoint is None:checkpoint=spec['candidate_checkpoint_sha256']
  assert checkpoint==spec['candidate_checkpoint_sha256'];reports.append(report)
  for f in(folder/'actual_host_wait_exit.json',folder/'pilot_complete.json',root/'protocol.json',root/'terminal_audit.json'):inputs[str(f)]=sha(f)
 groups={}
 for task in TASKS:
  rows=[r for report in reports for r in report['groups'][task]['records']]
  assert len(rows)==100 and len({r['case_id']for r in rows})==100 and {r['episode_index']for r in rows}==set(range(100))
  groups[task]={k:sum(report['groups'][task][k]for report in reports)for k in('scenes','baseline','full','TTA','rescues','harms','semantic_available','accepted_update_scenes','changed_action_scenes')}
  groups[task].update(TTA_minus_full=groups[task]['TTA']-groups[task]['full'],records=rows)
 write(complete,dict(complete=True,official_seed=0,actual_seed_start=100000,episodes_each_task=100,
  all_actual_host_wait0=True,groups=groups,candidate_checkpoint_sha256=checkpoint,
  original_zero_shadow_and_engineering_controls_exact=True,formal_success_not_used_for_fitting=True,
  input_sha256=inputs))
 print('CALIBRATED_OFFICIAL_PAIRED100_COMPLETE',flush=True)


if __name__=='__main__':main()

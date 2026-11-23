"""Two fixed native batches per pool; actual waits and no failed-stage retry."""
import argparse,os,sys,subprocess
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify
from .common import TASKS


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--split',choices=('development','confirmation'),required=True);args=parser.parse_args()
 base=T/'outputs/native_calibrated_validation_v24';complete=base/(args.split+'_complete.json');assert not complete.exists()
 reports=[];inputs={};checkpoint=None
 for number in range(2):
  stage=args.split+'_'+str(number).zfill(3);folder=base/stage
  assert not(folder/'actual_host_wait_exit.json').exists(),'Keep failed actual stage; no unchanged retry'
  process=subprocess.Popen([sys.executable,'-B','-u','-m','native_calibrated_validation_v24.launch'],env=dict(os.environ,CX_ONPOLICY_STAGE=stage))
  code=process.wait();assert code==0,'Native stage failed; no automatic GPU retry'
  actual=read(folder/'actual_host_wait_exit.json');assert actual['exit_code']==0 and actual['actual_Popen_wait_return'] and actual['proc_absent_after_wait']
  root=mapped(read(folder/'execution_state.json')['root']);spec=read(root/'protocol.json');verify(spec)
  assert read(root/'terminal_audit.json')['passed']
  report=read(folder/'pilot_complete.json');assert report['complete']
  if checkpoint is None:checkpoint=spec['candidate_checkpoint_sha256']
  assert checkpoint==spec['candidate_checkpoint_sha256'];reports.append(report)
  for f in(folder/'actual_host_wait_exit.json',folder/'pilot_complete.json',root/'protocol.json',root/'terminal_audit.json'):inputs[str(f)]=sha(f)
 groups={}
 for task in TASKS:
  rows=[r for report in reports for r in report['groups'][task]['records']]
  assert len(rows)==8 and len({r['case_id']for r in rows})==8 and {r['episode_index']for r in rows}==set(range(8))
  groups[task]={k:sum(report['groups'][task][k]for report in reports)for k in('scenes','baseline','full','TTA','rescues','harms','semantic_available','accepted_update_scenes','changed_action_scenes')}
  groups[task]['records']=rows
 interface=all(g['semantic_available']>0 and g['accepted_update_scenes']>0 and g['changed_action_scenes']>0 for g in groups.values())
 value=dict(complete=True,stage=args.split,all_actual_host_wait0=True,groups=groups,interface_qualified=interface,
  paired_net_success_change=sum(g['TTA']-g['full']for g in groups.values()),
  candidate_checkpoint_sha256=checkpoint,original_zero_shadow_and_engineering_controls_exact=True,
  no_confirmation_tuning=True,formal_EVAL_not_used=True,small_pilot_not_proof_of_general_benefit=True,input_sha256=inputs)
 write(complete,value)
 print('FIXED_CALIBRATED_NATIVE_POOL_COMPLETE',args.split,'interface',interface,'net',value['paired_net_success_change'],flush=True)


if __name__=='__main__':main()

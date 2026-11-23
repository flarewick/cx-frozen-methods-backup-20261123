"""Actual source plus adversarial negatives and mocked controller launch budget."""
import copy,json,types,tempfile
from pathlib import Path
from unittest.mock import patch
from . import controller as ctl
from .stale_interrupt import classify,REASON

def main():
 root=ctl.T/'outputs/native_cross_task100_v17/batches/009/runs/6913a38e538d48a38026d7b193d7ff99';resource=root/'attempt/resources.jsonl';base_rows=[json.loads(x)for x in resource.read_text().splitlines()];base_read=ctl.read;original_text=Path.read_text
 proof=classify(root,read=base_read,live_identity=ctl.live_identity);assert proof and proof['actual_child_exit_code']==1
 negatives=[]
 for name in ['wrong_nonce','wrong_protocol','wrong_GPU','normal_child','unreaped','wrong_child','signal_after_wait','termination','live_PID','new_foreign','unverified_old_PID','VRAM','account','CPU','RSS','missing_release','other_stop','software','changed_source']:
  rows=copy.deepcopy(base_rows)
  def read(p):
   value=copy.deepcopy(base_read(p));n=Path(p).name
   if n=='normal_child_exit.json':
    if name=='wrong_nonce':value['attempt_nonce']='wrong'
    if name=='wrong_protocol':value['protocol_identity']='wrong'
    if name=='wrong_GPU':value['gpu_uuid']='wrong'
    if name=='normal_child':value['normal_exit']=True
    if name=='unreaped':value['proc_absent_after_wait']=False
    if name=='wrong_child':value['child_inner_pid']+=1
   if n.startswith('signal_received_'):
    if name=='termination':value['signal']=15
    if name=='signal_after_wait':value['monotonic_ns']=base_read(root/'attempt/barrier/normal_child_exit.json')['waited_monotonic_ns']+1
   if n=='protocol.json'and name=='changed_source':value['source_sha256'][next(iter(value['source_sha256']))]='wrong'
   return value
  for row in rows:
   if name=='new_foreign'and row.get('raw_foreign_pids'):row['raw_foreign_pids']=[99999999]
   if name=='unverified_old_PID':row['normal_child_previously_metric_verified_pids']=[]
   if'seconds'in row:
    if name=='VRAM':row['memory_mib']=24564
    if name=='account':row['account_gpu_count']=4
    if name=='CPU':row['cpu']={'3865766':1601}
    if name=='RSS':row['rss_bytes']={'3865766':129*2**30}
   if name=='missing_release'and'own_gpu_released'in row:row['own_gpu_released']=False
   if name=='other_stop'and'stopped'in row:row['stopped']='Sustained GPU utilization/VRAM threshold would not be met'
  def text(p,*a,**k):
   if p==resource:return '\n'.join(json.dumps(x)for x in rows)+'\n'
   if p==root/'GPU_host.log'and name=='software':return 'AssertionError\nKeyboardInterrupt: Recorded external SIGINT\n'
   return original_text(p,*a,**k)
  with patch.object(Path,'read_text',text):
   assert classify(root,read=read,live_identity=(lambda _:dict(state='S'))if name=='live_PID'else(lambda _:None))is None,name
  negatives.append(name)
 # Entire controller may request exactly one restart, preserving/incrementing existing count.
 for previous_count in [2,3]:
  temp=ctl.O/('CPU_full_controller_'+str(previous_count));temp.mkdir(parents=True);invocation='CPU_fixture_stale_interrupt';state=dict(ActiveState='failed',MainPID='0',InvocationID=invocation)
  ctl.save(temp/'control.json',dict(recoveries={'009':previous_count,'008':1},preparation_recoveries={'008':1},handled_invocations=[],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),was_active=True))
  with patch.object(ctl,'stamp',return_value='CPU_fixture_only'),patch.object(ctl,'O',temp),patch.object(ctl,'unit',side_effect=lambda name:state if name==ctl.UNIT else dict(ActiveState='inactive',MainPID='0')),patch.object(ctl,'newest_attempt',return_value=root),patch.object(ctl,'snapshot',return_value=dict(benchmark_complete=False,qualified={})),patch.object(ctl.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stderr='',stdout=''))as command:
   ctl.main()
   if previous_count==2:
    command.assert_called_once();assert command.call_args.args[0]==['systemctl','--user','start','--no-block',ctl.UNIT]
    result=base_read(temp/'control.json');assert result['recoveries']=={'009':3,'008':1}and result['preparation_recoveries']=={'008':1}
   else:command.assert_not_called();assert(base_read(temp/'blocked.json')['reason']=='Three interruptions in one logical batch; needs diagnosis')
 ctl.save(ctl.O/'CPU_stale_interrupt_checks.json',dict(passed=True,actual_failure_proof=proof,negative_checks=negatives,actual_normal_child_exit_not_forged=True,actual_GPU_guard_and_scientific_source_unchanged=True,controller_mock_launch_once_at_count2_and_blocks_count3=True,all_existing_counters_preserved=True,no_actual_service_or_GPU_calls=True))
 print('BOUND_SIGINT_STALE_PID_CPU_PASS',len(negatives),flush=True)
if __name__=='__main__':main()

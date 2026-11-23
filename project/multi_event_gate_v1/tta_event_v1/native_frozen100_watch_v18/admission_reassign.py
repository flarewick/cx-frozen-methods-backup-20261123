"""Replan only a proven CPU-only wait after its selected card became occupied."""
from pathlib import Path
from native_frozen100_v18.common import COHORT,O as CAMPAIGN_OUTPUT
import fcntl,json,os,subprocess,time

def eligible(root,fleet,read,cgroup):
 if fleet['own_gpu_count']>=3:return False,'account limit; keep CPU wait'
 spec=read(root/'protocol.json');target=spec['required_gpu_uuid']
 if not any(c['available']and c['uuid']!=target for c in fleet['cards']):return False,'no other empty card'
 card=next(c for c in fleet['cards']if c['uuid']==target)
 if card['available']or not card['pids']:return False,'selected card not occupied'
 for name in('attempt/barrier/permit.json','attempt/barrier/cuda_ready.json','attempt/resources.jsonl'):
  if(root/name).exists():return False,'GPU permit/runtime exists; never cancel'
 ready=root/'source_bound_CPU_preload_proof.json'
 if not ready.exists():return False,'CPU preload not proved'
 r=read(ready)
 if not(r['passed']and r['cpu_fork_child_passed']and not r['cuda_initialized']and r['protocol_identity']==spec['identity']):return False,'CPU proof mismatch'
 unit='cx-tta-frozen100-v18-'+COHORT+'.service'
 if any(unit in cgroup(pid)for pid in card['pids']):return False,'GPU occupant may be this service'
 return True,'CPU-only selected card taken by another job; fresh empty card available'

def run(status,*,T,O,UNIT,ENV,read,save,event,unit):
 if not status.get('active_attempt'):return False
 root=Path(status['active_attempt'])
 # Cheap immutable stage checks before acquiring inventory or locks.
 if not(root/'source_bound_CPU_preload_proof.json').exists():return False
 if any((root/n).exists()for n in('attempt/barrier/permit.json','attempt/barrier/cuda_ready.json','attempt/resources.jsonl')):return False
 control=read(O/'control.json')if(O/'control.json').exists()else dict(recoveries={},handled_invocations=[])
 counts=control.setdefault('CPU_admission_reassignments',{});batch=root.parents[1].name
 if counts.get(batch,0)>=3:return False
 from native_fast_inventory_v1.inventory import inventory
 from native_frozen100_v18.common import verify
 from native_frozen100_v18.host import container
 def cgroup(pid):
  try:return(Path('/proc')/str(pid)/'cgroup').read_text()
  except FileNotFoundError:return ''
 lock=container.WORK/'gpu_locks/account_admission.lock'
 with lock.open('a')as account:
  try:fcntl.flock(account,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
  fleet=inventory();ok,reason=eligible(root,fleet,read,cgroup)
  if not ok:return False
  state=unit(UNIT)
  if state['ActiveState']!='active'or state['InvocationID']!=status['service']['InvocationID']:return False
  verify(read(CAMPAIGN_OUTPUT/'study.json'))
  # Holding the same account lock prevents the waiting host from granting GPU
  # permission between this final stage recheck and the service cancellation.
  ok,reason=eligible(root,fleet,read,cgroup)
  if not ok:return False
  witness=dict(time=__import__('datetime').datetime.now().astimezone().isoformat(),attempt=str(root),reason=reason,old_service=state,fresh_fleet=fleet,CPU_only_no_GPU_permit_or_resource_log=True,other_jobs_untouched=True)
  save(O/('CPU_reassign_'+state['InvocationID']+'_before.json'),witness)
  stopped=subprocess.run(['systemctl','--user','stop',UNIT],env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=35)
  if stopped.returncode:raise RuntimeError(stopped.stderr)
  after=unit(UNIT)
  if after['ActiveState']not in('inactive','failed')or after['MainPID']!='0':raise RuntimeError('CPU-only service not proved stopped')
  if any((root/n).exists()for n in('attempt/barrier/permit.json','attempt/barrier/cuda_ready.json','attempt/resources.jsonl')):raise RuntimeError('CPU cancellation stage proof changed unexpectedly')
  witness.update(actual_stop_returncode=stopped.returncode,service_after=after,old_attempt_preserved=True)
  save(O/('CPU_reassign_'+state['InvocationID']+'_after.json'),witness)
 counts[batch]=counts.get(batch,0)+1
 control['handled_invocations'].append(state['InvocationID']);save(O/'control.json',control)
 started=subprocess.run(['systemctl','--user','start','--no-block',UNIT],env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
 if started.returncode:raise RuntimeError(started.stderr)
 event('CPU_only_wait_reassigned',witness=witness)
 save(O/'last_action.json',dict(time=witness['time'],action='CPU_only_wait_reassigned_to_fresh_empty_card',reason=reason,previous_attempt=str(root),old_CPU_only_attempt_preserved=True,other_jobs_untouched=True))
 return True

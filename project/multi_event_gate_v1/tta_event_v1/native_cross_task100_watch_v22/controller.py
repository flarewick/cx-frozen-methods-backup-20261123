"""Persistent low-cost status and safe interruption recovery; no CUDA allocations."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys,time,traceback
from zoneinfo import ZoneInfo
T=Path(__file__).resolve().parents[1];P=T.parents[1];O=T/'outputs/native_cross_task100_watch_v22';R=T/'outputs/native_cross_task100_v19'
UNIT='cx-tta-cross-task100-v19.service';ENV=dict(os.environ,XDG_RUNTIME_DIR='/run/user/1015',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1015/bus')

def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def stamp():return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def event(kind,**data):
 with(O/'events.jsonl').open('a')as f:f.write(json.dumps(dict(time=stamp(),kind=kind,**data),ensure_ascii=False)+'\n')
def unit(name):
 result=subprocess.run(['systemctl','--user','show',name,'-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus','-p','Result','-p','InvocationID'],env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
 if result.returncode:raise RuntimeError(result.stderr)
 return dict(line.split('=',1)for line in result.stdout.splitlines()if'='in line)
def resources(folder):
 p=folder/'attempt/resources.jsonl'
 if not p.exists():return []
 text=p.read_text();return [json.loads(line)for line in text.split('\n')[:-1]if line]
def live_identity(pid):
 try:
  p=Path('/proc')/str(pid);f=(p/'stat').read_text().split(') ',1)[1].split();return dict(pid=pid,started_ticks=int(f[19]),state=f[0],cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
 except FileNotFoundError:return None

def interruption(folder,confirmed_reboot=False):
 rows=resources(folder)
 if any('stopped'in r for r in rows):
  from .stale_interrupt import classify
  proof=classify(folder,read=read,live_identity=live_identity)
  if proof:
   save(O/'verified_stale_interrupt'/(folder.parents[1].name+'_'+folder.name+'.json'),proof)
   return True,'Bound SIGINT child reaped; only formerly verified own PID lingered in NVML; real resource failures excluded'
  return False,'Resource guard stopped workload; diagnose before any retry'
 log=(folder/'GPU_host.log').read_text()if(folder/'GPU_host.log').exists()else''
 errors=('CUDA out of memory','OutOfMemoryError','AssertionError','ValueError:','TypeError:','FileNotFoundError:','MemoryError:','Changed evaluation dependency','same-prefix','Nonfinite','No healthy native','source changed')
 if any(x in log for x in errors):return False,'GPU/software/data error; no automatic unchanged retry'
 if confirmed_reboot and not(folder/'failure.json').exists():return True,'Confirmed kernel boot-ID change after an active run; unfinished batch excluded; old processes cannot survive reboot'
 signals=[read(p)for p in folder.glob('signal_received_*.json')]
 known=any(s.get('signal')==2 for s in signals)
 if any('RuntimeError:'in line and 'RuntimeError: Traceback (most recent call last):'not in line for line in log.splitlines()):return False,'Unclassified runtime error; no automatic retry'
 if not known:return False,'Unclassified failure; preserve state and diagnose'
 if any(s.get('signal')in(1,15)for s in signals):return False,'Termination/HUP may be intentional; no automatic restart'
 if not rows or not any(r.get('own_gpu_released')is True for r in rows):return False,'GPU release proof missing'
 for r in rows:
  for pid in r.get('gpu_pids',[]):
   if live_identity(pid) is not None:return False,'Former GPU PID still present; no duplicate launch'
 return True,'Recognized interruption; no resource stop; former CUDA processes gone'

def newest_attempt():
 values=list((R/'batches').glob('*/runs/*'))
 return max(values,key=lambda p:p.stat().st_mtime)if values else None

def snapshot(state):
 from native_cross_task100_v19.report import summarize
 progress=summarize();qualified=progress['groups'];active=newest_attempt();raw=[]
 if active:
  for case in sorted((active/'cases').glob('*')):
   branches=[]
   for branch in case.iterdir():
    if not branch.is_dir():continue
    done=(branch/'completed_task_release.json').exists()and(branch/'result.json').exists()
    row=read(branch/'result.json')if done else None
    branches.append(dict(condition=branch.name,queries=len(list(branch.glob('policy*.json'))),whole_released=done,success=row['success']if row else None))
   raw.append(dict(case_id=case.name,branches=branches,whole_pair_bundle=(case/'bundle.json').exists()))
 latest=read(R/'live_status.json')if(R/'live_status.json').exists()else None
 proc=live_identity(int(state.get('MainPID','0')))if int(state.get('MainPID','0'))else None
 status=dict(time=stamp(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),service=state,service_process=proc,qualified=qualified,benchmark_complete=progress['complete'],active_attempt=str(active)if active else None,raw_current_attempt=raw,latest_resource_observation=latest,automatic_recovery='Classified interrupts; bounded startup low-utilization recovery ONLY with strictly more CPU-verified preparation; unknown/unchanged/code/VRAM/late failures block',poll_seconds=30)
 if active and(active/'attempt/resources.jsonl').exists():
  status['telemetry_age_seconds']=time.time()-(active/'attempt/resources.jsonl').stat().st_mtime
 save(O/'status.json',status);return status

def main():
 O.mkdir(parents=True,exist_ok=True)
 with(O/'controller.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  if(O/'stop_requested.json').exists():event('explicit_stop_honored');return
  for previous_version in(11,12,13,14,15,16,17,18):
   previous_new=unit('cx-tta-cross-task100-v'+str(previous_version)+'.service')
   if previous_new.get('ActiveState')in('active','activating','deactivating'):event('previous_current_still_live_no_duplicate',service=previous_new);return
  previous_precision=unit('cx-tta-cross-task100-v9.service')
  if previous_precision.get('ActiveState')in('active','activating','deactivating'):event('previous_precision_still_live_no_duplicate',service=previous_precision);return
  previous_interleaved=unit('cx-tta-cross-task100-v8.service')
  if previous_interleaved.get('ActiveState')in('active','activating','deactivating'):event('previous_interleaved_still_live_no_duplicate',service=previous_interleaved);return
  previous_graph=unit('cx-tta-cross-task100-v7.service')
  if previous_graph.get('ActiveState')in('active','activating','deactivating'):event('previous_graph_still_live_no_duplicate',service=previous_graph);return
  previous_allocator=unit('cx-tta-cross-task100-v6.service')
  if previous_allocator.get('ActiveState')in('active','activating','deactivating'):event('previous_allocator_still_live_no_duplicate',service=previous_allocator);return
  previous_staged=unit('cx-tta-cross-task100-v5.service')
  if previous_staged.get('ActiveState')in('active','activating','deactivating'):event('previous_staged_still_live_no_duplicate',service=previous_staged);return
  previous_four=unit('cx-tta-cross-task100-v4.service')
  if previous_four.get('ActiveState')in('active','activating','deactivating'):event('previous_four_still_live_no_duplicate',service=previous_four);return
  previous_cross=unit('cx-tta-cross-task100-v3.service')
  if previous_cross.get('ActiveState')in('active','activating','deactivating'):event('previous_cross_task_still_live_no_duplicate',service=previous_cross);return
  parent_cross=unit('cx-tta-cross-task100-v2.service')
  if parent_cross.get('ActiveState')in('active','activating','deactivating'):event('parent_cross_task_still_live_no_duplicate',service=parent_cross);return
  old_cross=unit('cx-tta-cross-task100-v1.service')
  if old_cross.get('ActiveState')in('active','activating','deactivating'):event('old_cross_task_still_live_no_duplicate',service=old_cross);return
  old=unit('cx-tta-best100-v1.service');parent=unit('cx-tta-best100-resume-v2.service');previous_repair=unit('cx-tta-best100-resume-v3.service');state=unit(UNIT)
  previous_completed=unit('cx-tta-best100-resume-v7.service')
  if previous_completed.get('ActiveState')in('active','activating','deactivating'):event('original_study_still_live_no_duplicate',service=previous_completed);return
  previous_sampling=unit('cx-tta-best100-resume-v6.service')
  if previous_sampling.get('ActiveState')in('active','activating','deactivating'):event('previous_sampling_still_live_no_duplicate',service=previous_sampling);return
  previous_throughput=unit('cx-tta-best100-resume-v5.service')
  if previous_throughput.get('ActiveState')in('active','activating','deactivating'):event('previous_throughput_still_live_no_duplicate',service=previous_throughput);return
  previous_endpoint=unit('cx-tta-best100-resume-v4.service')
  if previous_endpoint.get('ActiveState')in('active','activating','deactivating'):event('previous_endpoint_still_live_no_duplicate',service=previous_endpoint);return
  if previous_repair.get('ActiveState')in('active','activating','deactivating'):event('previous_repair_still_live_no_duplicate',service=previous_repair);return
  if parent.get('ActiveState')in('active','activating','deactivating'):event('parent_recovery_still_live_no_duplicate',service=parent);return
  if old.get('ActiveState')in('active','activating','deactivating'):event('original_service_still_live_no_duplicate',service=old);return
  status=snapshot(state)
  if status['benchmark_complete']:
   event('benchmark_complete')
   subprocess.run(['systemctl','--user','disable','--now','cx-tta-cross-task100-watch-v22.timer'],env=ENV,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
   return
  control=read(O/'control.json')if(O/'control.json').exists()else dict(recoveries={},handled_invocations=[])
  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip();confirmed_reboot=bool(control.get('boot_id')and control['boot_id']!=boot_id and control.get('was_active'))
  if state.get('ActiveState')in('active','activating','deactivating'):
   from .admission_reassign import run as reassign_CPU_only
   if reassign_CPU_only(status,T=T,O=O,UNIT=UNIT,ENV=ENV,read=read,save=save,event=event,unit=unit):return
   control.update(boot_id=boot_id,was_active=True);save(O/'control.json',control)

   if status.get('telemetry_age_seconds',0)>90 and state.get('ActiveState')=='active':save(O/'warning.json',dict(time=stamp(),reason='Telemetry is stale; live service retained; inspect CPU-stage/log activity before termination',status=status))
   return
  invocation=state.get('InvocationID','');previous=newest_attempt()
  if invocation and invocation in control['handled_invocations']:return
  if previous:
   ok,reason=interruption(previous,confirmed_reboot=confirmed_reboot);batch=previous.parents[1].name
   if not ok:
    from .preparation_recovery import run as preparation_recovery
    if preparation_recovery(previous,state,control,T=T,O=O,UNIT=UNIT,ENV=ENV,read=read,save=save,event=event,unit=unit,live_identity=live_identity):return
    save(O/'blocked.json',dict(time=stamp(),attempt=str(previous),reason=reason,service=state));event('recovery_blocked',attempt=str(previous),reason=reason);return
   number=control['recoveries'].get(batch,0)
   if number>=3:
    save(O/'blocked.json',dict(time=stamp(),attempt=str(previous),reason='Three interruptions in one logical batch; needs diagnosis',service=state));return
   control['recoveries'][batch]=number+1
  else:reason='Initial separate cup/shoe transfer study after CPU compatibility checks; same frozen093/officialseed0 matched full; no original-task result reuse; all resource floors unchanged'
  # Verify frozen study and code before every start; acquire no GPU here.
  from native_cross_task100_v19.common import verify
  verify(read(R/'study.json'))
  control.update(boot_id=boot_id,was_active=False);control['handled_invocations'].append(invocation);save(O/'control.json',control)
  result=subprocess.run(['systemctl','--user','start','--no-block',UNIT],env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
  if result.returncode:raise RuntimeError(result.stderr)
  event('evaluation_service_started',reason=reason,previous_attempt=str(previous)if previous else None)
  save(O/'last_action.json',dict(time=stamp(),action='started_evaluation_service',reason=reason,qualified_before_start=status['qualified']))
if __name__=='__main__':
 try:main()
 except BaseException:
  O.mkdir(parents=True,exist_ok=True);save(O/'monitor_error.json',dict(time=stamp(),traceback=traceback.format_exc()));raise

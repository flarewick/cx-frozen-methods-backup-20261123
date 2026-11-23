"""Persistent 25-batch official full/TTA evaluation; stop on any failed actual job."""
from pathlib import Path
import os,sys,subprocess,time,uuid,fcntl,traceback,json,argparse
from .common import C,T,P,O,TASKS,CANON,read,write,replace,sha,identity,mapped,verify,batch
sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T/'resource_guard_v1'));sys.path.insert(0,str(T/'native_nextchunk_v1'))
import user_container as container
import gpu_doctor
from native_resource_policy_v4 import admission
from native_process_sampler_v2 import guard
from native_fast_inventory_v1.inventory import inventory
from native_visible_expansion_v1.placement import plan,apply

def canonical(p):return str(CANON/Path(p).resolve().relative_to(P))
def script(mode,module,*args):return container.command(['bash',canonical(C/'scripts'/('cpu.sh'if mode=='cpu'else'gpu.sh')),*(['-m',module]if mode=='cpu'else[]),*args],mount_project=True)
def freeze():
 dest=O/'study.json'
 if dest.exists():s=read(dest);verify(s);return s
 original=T/'outputs/native_cross_task100_v2';parent=read(original/'study.json');verify(parent)
 from native_cross_task100_v2.host import qualified_roots as old_roots
 reused=[]
 for index,root in sorted(old_roots(parent).items()):
  audit=read(root/'terminal_audit.json');assert audit['passed']
  for name,digest in audit['input_sha256'].items():assert sha(mapped(name))==digest
  reused.append(dict(batch_index=index,root=str(root),terminal_audit_sha256=sha(root/'terminal_audit.json')))
 assert [r['batch_index']for r in reused]==[0,1]
 sources=dict(parent['source_sha256'])
 for directory in(C,T/'native_process_sampler_v2'):
  for f in directory.rglob('*'):
   if f.is_file()and f.suffix in('.py','.sh','.json'):sources[str(f.relative_to(P))]=sha(f)
 inputs=dict(parent['input_sha256'])
 for f in(original/'study.json',O/'retained_candidate.json',O/'CPU_entry_contract_checks.json',O/'CPU_actual_exit.json',T/'outputs/native_process_sampler_v2/CPU_direct_NVML_checks.json',T/'outputs/native_process_sampler_v2/CPU_actual_exit.json'):
  inputs[str(f)]=sha(f)
 s={k:v for k,v in parent.items()if k not in('identity','source_sha256','input_sha256')}
 s.update(version='native_cross_task100_v3',simulation_lanes=8,previous_batch002_accepted_root=canonical(T/'outputs/native_cross_task100_v2/batches/002/runs/102e195a92104227852e7925b448b033/accepted'),all_expert_selection_before_policy=True,one_independent_case_per_lane=True,continuous_GPU_guard_clock_includes_all_screening=True,source_sha256=sources,input_sha256=inputs,reused_qualified_batches=reused,resume_at_batch_index=2,parent_study_identity=parent['identity'],resource_acquisition_repair_only=True,process_metric_sampler='native_process_sampler_v2 direct explicit numeric NVML v1 final-interval records',failed_parent_batch002_preserved_and_excluded=True,native_run_case_AST_unchanged=True,resource_scheduling_only=True,resource_guard_decisions_and_thresholds_unchanged=True)
 
 for f in (T/'outputs/native_cross_task100_v2/batches/002/runs/102e195a92104227852e7925b448b033/accepted').glob('*.json'):s['input_sha256'][str(f)]=sha(f)
 s['identity']=identity(s);verify(s);write(dest,s);return s

def wait_file(path,process,seconds):
 start=time.monotonic()
 while not path.exists():
  if process.poll()is not None:raise RuntimeError('Prepared process exited before '+str(path)+'; inspect log')
  if time.monotonic()-start>seconds:raise TimeoutError(str(path))
  time.sleep(.25)
 return read(path)

def choose():
 start=time.monotonic()
 while True:
  fleet=inventory()
  if fleet['own_gpu_count']<3:
   for gpu in (5,1,0,6,3,2,4):
    if gpu not in fleet['available_indices']:continue
    info=gpu_doctor.gpu_snapshot(gpu);cert=container.WORK/('gpu-doctor-'+str(gpu)+'.json')
    if not cert.exists():continue
    d=read(cert)
    if d['uuid']!=info['uuid']or d['build_mtime_ns']!=(container.WORK/'stage-bootstrap.ok').stat().st_mtime_ns:continue
    admission.require_admission(fleet,info['uuid']);return gpu,info,plan(gpu),fleet
  if time.monotonic()-start>36000:raise TimeoutError('No free certified GPU below account3 limit')
  replace(O/'live_status.json',dict(phase='CPU_only_wait_for_empty_GPU',fleet=fleet));time.sleep(10)

def terminal(root,spec):
 rows=[json.loads(s)for s in(root/'attempt/resources.jsonl').read_text().splitlines()];child=read(root/'attempt/barrier/normal_child_exit.json');result=read(root/'result.json')
 assert child['exit_code']==0 and child['normal_exit']and child['proc_absent_after_wait']and not child['cpu_only_fixture']and result['complete']
 assert child['protocol_identity']==spec['identity']and not any('stopped'in r for r in rows)
 released=[r for r in rows if'own_gpu_released'in r];assert len(released)==1 and released[0]['own_gpu_released']and not released[0]['remaining_own_gpu_pids']
 steady=[r for r in rows if r.get('seconds',0)>=280 and r.get('gpu_pids')];assert steady,'Actual scientific batch must qualify sustained resource metrics'
 for r in steady:
  assert min(r['mean_utilization'],r['time_weighted_mean_utilization'],r['rolling_300s_utilization'])>=34 and 4096<=r['memory_mib']<=24563 and r['account_gpu_count']<=3
  for pid in r['gpu_pids']:
   k=str(pid);assert r['process_utilization'][k]is not None and min(r['process_mean_utilization'][k],r['process_rolling_300s_utilization'][k])>=34 and r['process_memory_mib'][k]>=4096 and 40<=r['cpu'][k]<=1600 and r['rss_bytes'][k]<=128*2**30
 assert read(root/'semantic/result.json')['complete']and read(root/'semantic_exit.json')['exit_code']==0
 verify(spec);out=dict(passed=True,actual_host_exit_code=0,actual_child_exit_code=0,own_GPU_released=True,steady_resource_samples=len(steady),batch_protocol_identity=spec['identity'],input_sha256={str(f):sha(f)for f in [root/'result.json',root/'CPU_episode_audit.json',root/'attempt/resources.jsonl',root/'attempt/barrier/normal_child_exit.json',root/'semantic/result.json',root/'semantic_exit.json']})
 write(root/'terminal_audit.json',out);return out

def execute_batch(root,spec,gpu,info):
 assert not(root/'attempt').exists(),'Existing GPU attempt must be inspected, never retried unchanged'
 semlog=(root/'semantic_host.log').open('x');glog=(root/'GPU_host.log').open('x');sem=None;process=None;start=time.monotonic()
 try:
  semcmd=script('cpu','native_cross_task100_v3.semantic','--batch',canonical(root));sem=subprocess.Popen(semcmd,stdout=semlog,stderr=subprocess.STDOUT,start_new_session=True)
  write(root/'semantic_launch.json',dict(pid=sem.pid,started_ticks=int(Path('/proc/%d/stat'%sem.pid).read_text().split(') ',1)[1].split()[19]),command=semcmd));ready=wait_file(root/'semantic/ready.json',sem,240);assert ready['passed']and not ready['CUDA_initialized']and ready['source_identity']==spec['identity']
  attempt=root/'attempt';attempt.mkdir();barrier=attempt/'barrier';nonce=uuid.uuid4().hex
  cmd=script('gpu',None,'--batch',canonical(root),'--barrier',canonical(barrier),'--nonce',nonce);split=cmd.index('--');cmd[split:split]=gpu_doctor.driver_options(info)
  process=subprocess.Popen(cmd,stdout=glog,stderr=subprocess.STDOUT,start_new_session=True);ticks=int(Path('/proc/%d/stat'%process.pid).read_text().split(') ',1)[1].split()[19]);write(attempt/'launch.json',dict(pid=process.pid,started_ticks=ticks,command=cmd,gpu=info,nonce=nonce))
  proof=wait_file(barrier/'ready.json',process,240);assert proof['passed']and proof['cpu_fork_child_passed']and proof['os_threads']==1 and not proof['cuda_initialized']and proof['source_sha256']==spec['source_sha256'];write(root/'source_bound_CPU_preload_proof.json',proof)
  lockdir=container.WORK/'gpu_locks';lockdir.mkdir(exist_ok=True);waiting=time.monotonic()
  with(lockdir/'account_admission.lock').open('a')as account,(lockdir/('gpu_'+str(gpu)+'.lock')).open('a')as card:
   while True:
    try:
     fcntl.flock(account,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB);admission.check_admission(gpu_doctor.gpu_snapshot(gpu));break
    except(BlockingIOError,RuntimeError):
     fcntl.flock(account,fcntl.LOCK_UN);fcntl.flock(card,fcntl.LOCK_UN)
     if time.monotonic()-waiting>1500:raise TimeoutError('CPU admission wait expired without GPU')
     time.sleep(2)
   began=time.monotonic();started_ns=time.monotonic_ns();released=False;permitted=False
   def permit():
    admission.check_admission(gpu_doctor.gpu_snapshot(gpu));write(barrier/'permit.json',dict(gpu_uuid=info['uuid'],attempt_nonce=nonce,protocol_identity=spec['identity']))
   def sample(row):
    nonlocal released,permitted
    if len(row['gpu_pids'])>1:return 'Exactly one GPU owner required'
    if row['gpu_pids']and not released:fcntl.flock(account,fcntl.LOCK_UN);released=True
    if row['gpu_pids']and not permitted and(barrier/'cuda_ready.json').exists():write(barrier/'owned_pid_permit.json',dict(gpu_uuid=info['uuid'],host_GPU_PID=row['gpu_pids'][0]));permitted=True
    replace(O/'live_status.json',dict(phase='paired_GPU_evaluation',batch=spec['batch_index'],gpu=gpu,protocol_identity=spec['identity'],resources=row))
   guard.guarded_command(cmd,info,attempt/'resources.jsonl',gpu=gpu,deadline=7200,on_sample=sample,min_cumulative_utilization=34,max_memory_mib=24563,prepared_process=process,before_monitor=permit,child_exit_proof=barrier/'normal_child_exit.json',child_exit_identity=dict(attempt_nonce=nonce,gpu_uuid=info['uuid'],protocol_identity=spec['identity'],cpu_only_fixture=False,attempt_started_ns=started_ns))
  code=process.wait();assert code==0;write(root/'actual_GPU_parent_exit.json',dict(exit_code=code,actual_Popen_wait_return=True,pid=process.pid,started_ticks=ticks))
  write(root/'semantic/stop.json',dict(GPU_parent_actual_exit0=True));semcode=sem.wait(timeout=90);write(root/'semantic_exit.json',dict(exit_code=semcode,actual_Popen_wait_return=True,pid=sem.pid));assert semcode==0
  terminal(root,spec)
 except BaseException:
  write(root/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
  if process is not None and process.poll()is None:
   known=guard.stop_job(process);guard.record_release(process,info,start,root/'failed_release.jsonl',known)
  if sem is not None and sem.poll()is None:
   if not(root/'semantic/stop.json').exists():write(root/'semantic/stop.json',dict(reason='GPU_or_CPU_attempt_failed_no_retry'))
   try:code=sem.wait(timeout=30)
   except subprocess.TimeoutExpired:guard.stop_job(sem);code=sem.returncode
   if not(root/'semantic_exit.json').exists():write(root/'semantic_exit.json',dict(exit_code=code,actual_Popen_wait_return=True,pid=sem.pid))
  raise
 finally:semlog.close();glog.close()

def qualified_roots(s):
 result={r['batch_index']:mapped(r['root'])for r in s['reused_qualified_batches']}
 for n in range(25):
  found=[r for r in(O/'batches'/('%03d'%n)/'runs').glob('*')if(r/'terminal_audit.json').exists()]
  assert len(found)<=1,'Multiple qualified attempts must never be combined'
  if found:result[n]=found[0]
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');a=p.parse_args();s=freeze()
 if a.freeze_only:print('BEST100_RECOVERY_STUDY_FROZEN',s['identity']);return
 os.environ['CX_BEST100_RUN_ID']=uuid.uuid4().hex
 write(O/'host_runs'/os.environ['CX_BEST100_RUN_ID']/'launch.json',dict(pid=os.getpid(),started_ticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),run_id=os.environ['CX_BEST100_RUN_ID']))
 with(O/'campaign.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state={t:dict(next_seed=100000,accepted=0)for t in TASKS};existing=qualified_roots(s)
  for n in range(25):
   if n in existing:
    previous=existing[n];verify(read(previous/'protocol.json'));assert read(previous/'terminal_audit.json')['passed'];state=read(previous/'screening_complete.json')['state'];continue
   root=batch(n);assert not root.exists(),'Never overwrite a prior interrupted attempt'
   gpu,info,placement,fleet=choose();root.mkdir(parents=True);write(root/'cpu_placement_plan.json',placement);write(root/'device_plan.json',dict(gpu=gpu,info=info,fresh_fleet=fleet));spec=dict(s,batch_index=n,initial_screen_state=state,CPU_placement=canonical(root/'cpu_placement_plan.json'),required_gpu_uuid=info['uuid'],study_identity=s['identity']);spec.pop('identity');spec['input_sha256']=dict(spec['input_sha256'],**{canonical(O/'study.json'):sha(O/'study.json'),canonical(root/'cpu_placement_plan.json'):sha(root/'cpu_placement_plan.json'),canonical(root/'device_plan.json'):sha(root/'device_plan.json')});spec['identity']=identity(spec);write(root/'protocol.json',spec)
   allowed=os.sched_getaffinity(0)
   try:apply(placement);execute_batch(root,spec,gpu,info)
   finally:os.sched_setaffinity(0,allowed)
   state=read(root/'screening_complete.json')['state']
   from .report import summarize
   summarize();print('BEST100_RECOVERED_BATCH_COMPLETE',n,state,flush=True)
  assert all(state[t]['accepted']==100 for t in TASKS)
  from .report import summarize
  summarize(final=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  folder=O/'host_runs'/os.environ.get('CX_BEST100_RUN_ID','before_launch');write(folder/'failure.json',dict(traceback=traceback.format_exc(),automatic_GPU_retry=False));replace(O/'campaign_failure.json',dict(traceback=traceback.format_exc(),run_id=os.environ.get('CX_BEST100_RUN_ID'),automatic_GPU_retry=False));raise

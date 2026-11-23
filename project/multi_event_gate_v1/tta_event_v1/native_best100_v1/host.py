"""Persistent 25-batch official full/TTA evaluation; stop on any failed actual job."""
from pathlib import Path
import os,sys,subprocess,time,uuid,fcntl,traceback,json,argparse
from .common import C,T,P,O,TASKS,CANON,read,write,replace,sha,identity,mapped,verify,batch
sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T/'resource_guard_v1'));sys.path.insert(0,str(T/'native_nextchunk_v1'))
import user_container as container
import gpu_doctor
from native_resource_policy_v4 import guard,admission
from native_fast_inventory_v1.inventory import inventory
from native_visible_expansion_v1.placement import plan,apply

def canonical(p):return str(CANON/Path(p).resolve().relative_to(P))
def script(mode,module,*args):return container.command(['bash',canonical(C/'scripts'/('cpu.sh'if mode=='cpu'else'gpu.sh')),*(['-m',module]if mode=='cpu'else[]),*args],mount_project=True)
def freeze():
 dest=O/'study.json'
 if dest.exists():s=read(dest);verify(s);return s
 oldfile=T/'outputs/native_visible_expand_c_v1/protocol.json';old=read(oldfile);sources=dict(old['source_sha256'])
 for directory in [C,T/'native_resource_policy_v4',T/'native_fast_inventory_v1']:
  for f in directory.rglob('*'):
   if f.is_file()and f.suffix in('.py','.sh','.json'):sources[str(f.relative_to(P))]=sha(f)
 for n in ('collection_capture_gate.py','collection_gc_schedule.py','shared6_planner_pool.py','model_service.py','concurrent_rng.py','scene_gate.py'):
  f=P/'deployment'/n;sources[str(f.relative_to(P))]=sha(f)
 for n in ('target_shared6_planner_pool_cpu_20261024.json','target_collection_capture_gate_cpu_20261024.json','target_collection_gc_schedule_cpu_20261024.json'):
  f=P/'deployment/reports'/n;sources[str(f.relative_to(P))]=sha(f)
 q=mapped(old['trained_qualification']);qualified=read(q);assert qualified['training_checkpoint_sha256']=='0937511affea00771ed7f595d07220ae390ba85e06e398fce678446e7078a1d5'
 inputs={str(f):sha(f)for f in [oldfile,q,mapped(qualified['training_checkpoint']),O/'retained_candidate.json',O/'historical_references.json',T/'outputs/native_arrived_mask_v3/cpu_models.json']}
 s=dict(version='native_best100_v1',scope='frozen_best_TTA_official_seed0_paired100',tasks=list(TASKS),official_seed_group=0,actual_seed_start=100000,max_screen_attempts_per_task=1000,episodes_per_task=100,batches=25,cases_per_task_per_batch=4,cases_per_batch=8,simulation_lanes=4,conditions=['M_event_full','TTA'],first_batch_whole_controls='first accepted episode each task also zero_update and shadow_update',candidate_checkpoint_sha256=qualified['training_checkpoint_sha256'],trained_qualification=old['trained_qualification'],training=old['training'],runtime_source=old['runtime_source'],projection=old['projection'],event_config=old['event_config'],source_sha256=sources,input_sha256=inputs,offline_expert_screening_only=True,formal_outcomes_never_training_or_hyperparameter_inputs=True,actor_sampler_seed=17,actor_sampler_seed_reason='Unchanged original093 candidate recipe; same for newly paired full; official seed0 selects scene pool only',new_training_steps=0,exact_initial_physics_RNG_RGB_commands_and_common_prefix_required=True,resource_policy='native_resource_policy_v4',min_cumulative_and_rolling300_device_and_process_utilization=34,GPU_memory_MiB=[4096,24563],CPU_percent=[40,1600],max_logical_CPUs_per_card=16,max_RSS_GiB=128,max_account_GPUs=3,GPU_deadline_seconds=7200,no_outcome_early_stop=True,no_automatic_retry_of_failed_GPU_workload=True,user_authority='2026-10-29 explicitly approved official seed0 and same-scene frozen full controls for both100',reserved_independent_windows_unopened=True)
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
   for gpu in (5,1,0,6,3):
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
  semcmd=script('cpu','native_best100_v1.semantic','--batch',canonical(root));sem=subprocess.Popen(semcmd,stdout=semlog,stderr=subprocess.STDOUT,start_new_session=True)
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

def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');a=p.parse_args();s=freeze()
 if a.freeze_only:print('BEST100_STUDY_FROZEN',s['identity']);return
 O.mkdir(exist_ok=True)
 with(O/'campaign.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state={t:dict(next_seed=100000,accepted=0)for t in TASKS}
  for n in range(25):
   root=batch(n)
   if(root/'terminal_audit.json').exists():
    assert read(root/'terminal_audit.json')['passed'];state=read(root/'screening_complete.json')['state'];continue
   assert not(root/'protocol.json').exists(),'Prior prepared/failed batch exists; inspect explicit state before continuation'
   gpu,info,placement,fleet=choose();root.mkdir(parents=True);write(root/'cpu_placement_plan.json',placement);write(root/'device_plan.json',dict(gpu=gpu,info=info,fresh_fleet=fleet));spec=dict(s,batch_index=n,initial_screen_state=state,CPU_placement=canonical(root/'cpu_placement_plan.json'),required_gpu_uuid=info['uuid'],study_identity=s['identity']);spec.pop('identity');spec['input_sha256']=dict(spec['input_sha256'],**{canonical(O/'study.json'):sha(O/'study.json'),canonical(root/'cpu_placement_plan.json'):sha(root/'cpu_placement_plan.json'),canonical(root/'device_plan.json'):sha(root/'device_plan.json')});spec['identity']=identity(spec);write(root/'protocol.json',spec)
   # Restrict this batch and its CPU semantic/model children to the same16 CPUs.
   allowed=os.sched_getaffinity(0)
   try:
    apply(placement);execute_batch(root,spec,gpu,info)
   finally:os.sched_setaffinity(0,allowed)
   state=read(root/'screening_complete.json')['state']
   from .report import summarize
   summarize();print('BEST100_PAIRED_BATCH_COMPLETE',n,state,flush=True)
  assert all(state[t]['accepted']==100 for t in TASKS)
  from .report import summarize
  summarize(final=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  replace(O/'campaign_failure.json',dict(traceback=traceback.format_exc(),automatic_GPU_retry=False));raise

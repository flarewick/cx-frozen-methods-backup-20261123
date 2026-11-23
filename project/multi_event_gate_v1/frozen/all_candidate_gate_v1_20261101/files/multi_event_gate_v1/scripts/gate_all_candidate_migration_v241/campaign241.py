"""Trace the next startup after diagnosed SIGINT; continue only normal progressing drains."""
import os,sys,time,json,signal,subprocess,fcntl
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent;W=HERE.parents[1];D=W.parent/'deployment';H=W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4';PREV=HERE.parent/'gate_all_candidate_v237';STATE=H/'all_candidate_v241_state.json'
sys.path.insert(0,str(D));import benefit_inventory_nvml_v60 as inv;inv.install()
from gpu_policy import require_admission
sys.path.insert(0,str(HERE.parent/'gate_transfer_affinity_v225'));from affinity225 import choose
sys.path.insert(0,str(HERE.parent/'gate_transfer_topp_pool_v218'));from pool_host218 import Pool
UUID='GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return __import__('hashlib').sha256(Path(f).read_bytes()).hexdigest()
def atomic(f,v):t=f.with_suffix('.tmp');t.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');t.replace(f)
def run():
 lock=(H/'all_candidate237.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 child=None;pool=None;st=dict(pid=os.getpid(),phase='CPU_diagnosed_signal_recovery',started_unix=time.time(),goal_achieved=False);frozen={str(f):sha(f)for f in [*PREV.glob('*.py'),*HERE.glob('*.py'),HERE/'audit242.py']}
 def update(**kw):st.update(updated_unix=time.time(),**kw);atomic(STATE,st)
 def check():
  assert all(sha(f)==h for f,h in frozen.items())
  assert sha('/home/qiufw/服务器使用注意事项.md')=='ba46e7a5a8ca832941742b3b2a764036e83236d95b4ee0226a73ce916dc5b4db'
 def call(script,*args,original=False):
  nonlocal child
  check();log=root/('migration241_'+Path(script).stem+'_'+str(time.time_ns())+'.log');env=dict(os.environ,CX_GATE_TRANSFER_COHORT=str(cohort),CX_ALL237_ORIGINAL_CPU='1'if original else'0',CX_BENEFIT_GROWTH_BLOCK='dev0',PYTHONDONTWRITEBYTECODE='1',CX_TOPP218_CONFIG=pool.canonical_config if pool else'')
  with log.open('w')as out:child=subprocess.Popen([sys.executable,'-u','-B',str(script),*map(str,args)],env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
  update(phase=Path(script).stem,cohort=cohort,child_pid=child.pid,log=str(log))
  while child.poll()is None:
   check()
   if pool:
    assert pool.healthy()
    cores=set(st['CPU_cores'])
    for proc in [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]:
     try:assert set(proc.cpu_affinity())<=cores and proc.memory_info().rss<=128*2**30
     except psutil.NoSuchProcess:pass
   time.sleep(3)
  code=child.returncode;child=None
  if code:raise RuntimeError('Child failed; require diagnosis, no GPU retry: '+str(log))
 def cpu(script,*args,original=False):call(PREV/'cpu237.py' if original else HERE/'cpu241.py',script,*args,original=original)
 def whole():
  f=root/'validate_plan.json'
  if not f.exists():return 0,False
  count=0;done=True
  for j in read(f)['jobs']:
   r=Path(str(j['run']).replace('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA',str(W.parent)));q=r/'concurrency_queue.json';res=r/'result.json';count+=len(read(q)['completed'])if q.exists()else 0;done=done and res.exists()and read(res)['complete']
  return count,done
 def interrupted(sig,frame):raise KeyboardInterrupt('Explicit stop '+str(sig))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  update(source_sha256=frozen)
  # Preserve every old failure and exact known SIGINT episode; source was not observable then.
  old=read(H/'all_candidate_v237_state.json');assert old['phase']=='stopped_requires_diagnosis'and old.get('CPU_TOPP_workers_stopped')
  update(explicit_GPU0_to_original_GPU3_migration=True,original_failure_preserved=old['error'],next_start_has_signal_sender_instrumentation=True,no_threshold_or_policy_change=True)
  for cohort in (0,1):
   root=H/('candidate_veto_all_events_v237_'+str(cohort))/'dev0'
   if(root.parent/'final.json').exists():assert read(root.parent/'final.json')['passed'];continue
   if not(root/'expert_membership_import237.json').exists():
    cpu(PREV/'bootstrap237.py',original=True)
    for name,target in [('verify_benefit_full_source_CPU_v177.py','mixed_runtime_cpu_proof.json'),('benefit_followup_mixed_contract_v5.py','growth_cpu_contract.json'),('verify_benefit_full_deferred_CPU_v179.py','deferred_finalize_cpu_proof.json')]:
     if not(root/target).exists():cpu(name,original=True)
    cpu(PREV/'verify237.py',original=True);cpu(PREV/'seal237.py')
   call(HERE/'host241.py','--gpu','3','--stage','validate','--cpu-proof')
   cpu(HERE/'verify241.py')
   for _ in range(5760):
    check();fleet=inv.inventory()
    try:require_admission(fleet,UUID);break
    except RuntimeError as e:update(phase='CPU_waiting_GPU0_admission',reason=str(e));time.sleep(15)
   else:raise RuntimeError('Admission wait expired')
   # Restore CPU choices for this idle cohort before selecting a fresh nonoverlapping mask.
   os.sched_setaffinity(0,initial_affinity)
   cores,node=choose(fleet);os.sched_setaffinity(0,cores);update(CPU_cores=cores,CPU_NUMA_node=node)
   pool=Pool(cores,'all241_'+str(cohort));update(CPU_TOPP_worker_host_pids=[p.pid for p in pool.children],CPU_TOPP_workers_stopped=False)
   for batch in range(20):
    before,done=whole()
    if done:break
    update(G_whole=before,batch=batch)
    call(HERE/'host241.py','--gpu','3','--stage','validate')
    after,done=whole();update(G_whole=after)
    if done:break
    hh=read(root/'host_state.json');assert hh['phase']=='maintenance_batch_complete'and hh.get('normal_drained_exit')and after>before
   assert whole()==(40,True)
   pool.close();pool=None;update(CPU_TOPP_workers_stopped=True)
   cpu(HERE/'audit242.py')
  # This final original-controller invocation only aggregates already audited finals.
  assert all(read(H/('candidate_veto_all_events_v237_'+str(i))/'final.json')['passed']for i in(0,1))
  lock.close();root=H/'candidate_veto_all_events_v237_1/dev0';call(PREV/'controller237.py')
  assert read(H/'all_candidate_v237_final.json')['passed'];update(phase='complete',goal_achieved=True,final_sha256=sha(H/'all_candidate_v237_final.json'))
 except BaseException as e:
  update(phase='stopped_requires_diagnosis',error=repr(e),whole_results_preserved=True)
  if child and child.poll()is None:__import__('gpu_guard_bounded_metrics').stop_job(child)
  raise
 finally:
  if pool:pool.close();update(CPU_TOPP_workers_stopped=True)
initial_affinity=os.sched_getaffinity(0)
if __name__=='__main__':run()

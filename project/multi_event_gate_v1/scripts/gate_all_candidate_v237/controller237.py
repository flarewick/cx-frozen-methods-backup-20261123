"""Bounded two-cohort all-event diagnostic; original guards and no blind failure retries."""
import fcntl,json,os,sys,time,subprocess,signal
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent;W=HERE.parents[1];D=W.parent/'deployment';H=W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4';STATE=H/'all_candidate_v237_state.json'
sys.path.insert(0,str(D));import benefit_inventory_nvml_v60 as inv;inv.install()
from gpu_policy import require_admission
sys.path.insert(0,str(HERE.parent/'gate_transfer_affinity_v225'));from affinity225 import choose
sys.path.insert(0,str(HERE.parent/'gate_transfer_topp_pool_v218'));from pool_host218 import Pool
UUID='GPU-fa87bc41-1919-efe1-73f9-11f38008763e';RULES=Path('/home/qiufw/服务器使用注意事项.md')
def read(f):return json.loads(Path(f).read_text())
def sha(f):return __import__('hashlib').sha256(Path(f).read_bytes()).hexdigest()
def atomic(f,v):tmp=Path(f).with_suffix('.tmp');tmp.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');tmp.replace(f)
def main():
 lock=(H/'all_candidate237.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(pid=os.getpid(),started_unix=time.time(),phase='CPU_preparation',goal_achieved=False,GPU_index=0,GPU_UUID=UUID,no_training=True);child=None;pool=None;frozen={str(f):sha(f)for f in HERE.glob('*.py')}
 def update(**kw):state.update(updated_unix=time.time(),**kw);atomic(STATE,state)
 def check():
  assert all(sha(f)==h for f,h in frozen.items()),'237 source changed'
  assert sha(RULES)=='ba46e7a5a8ca832941742b3b2a764036e83236d95b4ee0226a73ce916dc5b4db'
 def stop():
  if child and child.poll()is None:
   import gpu_guard_bounded_metrics as guard;guard.stop_job(child)
 def run(script,*args,original=False):
  nonlocal child
  check();phase=Path(script).stem;root=H/('candidate_veto_all_events_v237_'+str(cohort))/'dev0';log=root/('controller_'+phase+'_'+str(time.time_ns())+'.log')
  env=dict(os.environ,CX_GATE_TRANSFER_COHORT=str(cohort),CX_BENEFIT_GROWTH_BLOCK='dev0',CX_ALL237_ORIGINAL_CPU='1'if original else'0',PYTHONDONTWRITEBYTECODE='1',CX_TOPP218_CONFIG=pool.canonical_config if pool else'')
  cmd=[sys.executable,'-u','-B',str(script),*map(str,args)]
  with log.open('w')as out:child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
  update(phase=phase,cohort=cohort,child_pid=child.pid,log=str(log))
  while child.poll()is None:
   check()
   if pool and not pool.healthy():raise RuntimeError('CPU TOPP worker unhealthy')
   # All children inherit this reserved <=16-core mask. Check individual RSS.
   if pool:
    cores=set(state['CPU_cores'])
    for proc in [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]:
     try:assert set(proc.cpu_affinity())<=cores and proc.memory_info().rss<=128*2**30
     except psutil.NoSuchProcess:pass
   time.sleep(3)
  code=child.returncode;child=None
  if code:raise RuntimeError('Child failed; diagnose before retry: '+phase+' code='+str(code)+' log='+str(log))
 def cpu(script,*args,original=False):run(HERE/'cpu237.py',script,*args,original=original)
 def status(root):
  f=root/'validate_plan.json'
  if not f.exists():return(0,False)
  jobs=read(f)['jobs'];n=0;complete=True
  for j in jobs:
   run=Path(j['run'])
   if not run.exists():run=W.parent/run.relative_to('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')
   q=run/'concurrency_queue.json';r=run/'result.json';n+=len(read(q)['completed'])if q.exists()else 0;complete=complete and r.exists()and read(r).get('complete',False)
  return n,complete
 def interrupted(sig,frame):raise KeyboardInterrupt('Stop '+str(sig))
 signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
 try:
  update(source_sha256=frozen)
  for cohort in (0,1):
   root=H/('candidate_veto_all_events_v237_'+str(cohort))/'dev0'
   if(root.parent/'final.json').exists():assert read(root.parent/'final.json')['passed'];continue
   if not(root/'expert_membership_import237.json').exists():
    cpu(HERE/'bootstrap237.py',original=True)
    for name,target in [('verify_benefit_full_source_CPU_v177.py','mixed_runtime_cpu_proof.json'),('benefit_followup_mixed_contract_v5.py','growth_cpu_contract.json'),('verify_benefit_full_deferred_CPU_v179.py','deferred_finalize_cpu_proof.json')]:
     if not(root/target).exists():cpu(name,original=True)
    cpu(HERE/'verify237.py',original=True)
    cpu(HERE/'seal237.py')
   assert read(root/'all_candidate_CPU237.json')['passed']
   run(HERE/'host237.py','--gpu','0','--stage','validate','--cpu-proof')
   for _ in range(5760):
    check();fleet=inv.inventory()
    try:require_admission(fleet,UUID);break
    except RuntimeError as e:update(phase='CPU_waiting_GPU0_admission',reason=str(e));time.sleep(15)
   else:raise RuntimeError('24h admission wait exhausted')
   cores,node=choose(fleet);os.sched_setaffinity(0,cores);update(CPU_cores=cores,CPU_NUMA_node=node)
   pool=Pool(cores,'all237_'+str(cohort));update(CPU_TOPP_worker_host_pids=[p.pid for p in pool.children])
   for batch in range(20):
    before,done=status(root)
    if done:break
    update(G_whole=before,batch=batch)
    run(HERE/'host237.py','--gpu','0','--stage','validate')
    after,done=status(root);update(G_whole=after)
    if done:break
    host=read(root/'host_state.json');assert host['phase']=='maintenance_batch_complete'and host.get('normal_drained_exit')and after>before,'Only normal maintenance with progress may automatically resume'
   assert status(root)==(40,True)
   pool.close();pool=None;update(CPU_TOPP_workers_stopped=True)
   cpu(HERE/'audit237.py')
  finals=[read(H/('candidate_veto_all_events_v237_'+str(i))/'final.json')for i in (0,1)];cases=sum([f['cases']for f in finals],[]);assert len(cases)==80
  metrics={}
  for t in ['place_empty_cup','stack_bowls_two']:
   rows=[c for c in cases if c['task']==t];assert len(rows)==40
   metrics[t]={key:sum(r[key]for r in rows)for key in ['baseline_success','event_full_success','first_gate_success','all_gate_success','B_chunks','first_G_chunks','all_G_chunks','vetoed_candidates','later_vetoes']}
   metrics[t].update(scenes=40,harms_avoided=sum(r['baseline_success']and not r['event_full_success']and r['all_gate_success']for r in rows),rescues_lost=sum(not r['baseline_success']and r['event_full_success']and not r['all_gate_success']for r in rows))
  final=dict(passed=True,complete=True,metrics=metrics,cases=cases,no_training=True,not_independent_confirmation_or_100_episode=True);atomic(H/'all_candidate_v237_final.json',final)
  report=H/'全事件门控v118对照结果.md';lines=['# 冻结v118全事件调用对照','','|任务|baseline|event_full|首次门控|全事件门控|避免B损害|损失B救回|','|---|---:|---:|---:|---:|---:|---:|']
  for t,m in metrics.items():lines.append(f"|{t}|{m['baseline_success']}/40|{m['event_full_success']}/40|{m['first_gate_success']}/40|{m['all_gate_success']}/40|{m['harms_avoided']}|{m['rescues_lost']}|")
  lines+=['','沿用已查看的固定EVAL场景；仅新增全事件G，旧控制/首次G保留明确来源。冻结adapter/门控/阈值/预算；后续事件未经校准，属于迁移诊断，不作训练标签、独立确认或100局总体结论。'];report.write_text('\n'.join(lines)+'\n');update(phase='complete',goal_achieved=True,metrics=metrics,report=str(report))
 except BaseException as e:
  update(phase='stopped_requires_diagnosis',error=repr(e),whole_results_preserved=True);stop();raise
 finally:
  if pool:pool.close();update(CPU_TOPP_workers_stopped=True)
if __name__=='__main__':main()

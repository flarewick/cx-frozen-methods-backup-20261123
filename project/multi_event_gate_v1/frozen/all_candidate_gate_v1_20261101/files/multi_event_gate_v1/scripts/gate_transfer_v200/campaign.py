"""Two prospective 20/task cohorts, frozen gate, original guards, no blind retries."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[2];D=R/'deployment';sys.path.insert(0,str(D))
H=R/'multi_event_gate_v1/outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4'
STATE=H/'cup_bowls_transfer_v200_state.json'
RULES=Path('/home/qiufw/服务器使用注意事项.md')
from benefit_pilot_protocol import sha

def atomic(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,sort_keys=True)+'\n');q.replace(p)
def main():
 lock=(H/'cup_bowls_transfer_v200.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import benefit_inventory_nvml_v60 as inv;inv.install()
 from gpu_policy import require_admission
 def physical(cpu):
  p=Path('/sys/devices/system/cpu',f'cpu{cpu}','topology');return(int((p/'physical_package_id').read_text()),int((p/'core_id').read_text()))
 busy=set()
 for card in inv.inventory()['cards']:
  for pid in card['own_pids']:
   try:busy.update(physical(c)for c in os.sched_getaffinity(pid))
   except ProcessLookupError:pass
 cores=[];seen=set()
 for cpu in sorted(os.sched_getaffinity(0)):
  key=physical(cpu)
  if Path('/sys/devices/system/node/node0',f'cpu{cpu}').exists()and key not in busy|seen:cores.append(cpu);seen.add(key)
 if len(cores)<12:raise RuntimeError('Need 12 available local physical cores; no GPU launched')
 cores=cores[:16];os.sched_setaffinity(0,cores)
 frozen={str(f):sha(f)for f in HERE.glob('*.py')};state=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],CPU_cores=cores,gpu=3,cohorts_complete=[],models_and_thresholds_unchanged=True,source_sha256=frozen,goal_achieved=False)
 log=H/('cup_bowls_transfer_v200_'+str(time.time_ns())+'.log');state['log']=str(log)
 def update(**kw):state.update(updated_unix=time.time(),**kw);atomic(STATE,state)
 def check():assert all(sha(Path(f))==h for f,h in frozen.items())
 def admission(out):
  assert sha(RULES)=='ba46e7a5a8ca832941742b3b2a764036e83236d95b4ee0226a73ce916dc5b4db';out.write(RULES.read_text()+'\n');out.flush()
  for _ in range(360):
   try:require_admission(inv.inventory(),'GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a');return
   except RuntimeError as e:update(phase='CPU_waiting_fresh_admission',reason=str(e));time.sleep(5)
  raise RuntimeError('Admission wait expired without launching')
 with log.open('w')as out:
  try:
   for cohort in range(2):
    v=H/('candidate_veto_cup_bowls_v200_'+str(cohort));root=v/'dev0';env=dict(os.environ,CX_GATE_TRANSFER_COHORT=str(cohort),CX_BENEFIT_GROWTH_BLOCK='dev0',PYTHONDONTWRITEBYTECODE='1')
    def run(args,phase,gpu=False,extra=None):
     check();update(phase=phase,cohort=cohort)
     if gpu:admission(out)
     child=subprocess.Popen([sys.executable,'-B',*map(str,args)],env=dict(env,**(extra or{})),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
     update(child_pid=child.pid,child_start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21])
     watch=None
     if phase=='guarded_controls88':
      watch=subprocess.Popen([sys.executable,'-B',str(HERE/'cpu_host.py'),'benefit_canonical_AA_watch_v96.py'],env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
      update(AA_watcher_pid=watch.pid)
     try:
      while child.poll()is None:
       check()
       if watch is not None and ((root/'canonical_integrity_failure_v96.json').exists()or(watch.poll()is not None and watch.returncode!=0)):
        f=root/'host_state.json'
        if f.exists():
         hh=json.loads(f.read_text());proc=Path('/proc',str(hh['pid']))
         if proc.exists()and proc.stat().st_uid==os.getuid():
          cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();ee=(proc/'environ').read_bytes().split(b'\0')
          if 'gate_transfer_v200/entry.py'in cmd and 'benefit_cache_preference_host_v96.py'in cmd and ('CX_GATE_TRANSFER_COHORT='+str(cohort)).encode()in ee:os.kill(hh['pid'],signal.SIGINT)
        raise RuntimeError('AA integrity observer failed; stop own verified host, preserve results')
       time.sleep(2)
      code=child.returncode;check()
      if watch is not None and code==0:
       assert watch.wait(timeout=60)==0
       aa=json.loads((root/'canonical_AA_watch_v96.json').read_text());assert aa['passed']and not aa['partial_only']and len(aa['AA_pairs'])==8
     finally:
      if watch is not None and watch.poll()is None:
       import gpu_guard_bounded_metrics as guard;guard.stop_job(watch)
     if code:raise RuntimeError('Child failed, no blind retry: '+phase+' exit '+str(code))
    def cpu(script,*args,original=False):run([HERE/'cpu_host.py',script,*args],'CPU_'+Path(script).stem,extra={'CX_TRANSFER_ORIGINAL_PLANNER_CHECK':'1'}if original else None)
    if (v/'final.json').exists():
     f=json.loads((v/'final.json').read_text());assert f['passed']and f['complete'];state['cohorts_complete'].append(cohort);continue
    cpu(HERE/'preflight.py');cpu('benefit_event_full_scope_audit_v175.py','verify')
    cpu('verify_benefit_event_full_audit_CPU_v135.py')
    if not(root/'finite_bootstrap_host_receipt_v6.json').exists():
     run([HERE/'entry.py',D/'benefit_finite_signal_trace_host_v187.py','--gpu',3,'--cpu-proof'],'CPU_finite_safe_fork')
     run([HERE/'entry.py',D/'benefit_finite_signal_trace_host_v187.py','--gpu',3],'guarded_finite_real_experts',True)
    cpu('verify_benefit_full_source_CPU_v177.py',original=True)
    cpu('benefit_followup_mixed_contract_v5.py',original=True)
    cpu('verify_benefit_event_full_productive_CPU_v178.py')
    cpu('verify_benefit_full_deferred_CPU_v179.py',original=True)
    run([HERE/'collector.py','collect','--gpu',3],'guarded_controls88',True)
    for n in ['collection_audit.json','mixed_membership_final_audit.json','full_event_control_extension_v134.json']:
     a=json.loads((root/n).read_text());assert a['passed']and a['complete']
    run([HERE/'collector.py','validate','--gpu',3],'guarded_actual_G40',True)
    f=json.loads((v/'final.json').read_text());assert f['passed']and f['complete']and f['actual_G_episodes']==40;state['cohorts_complete'].append(cohort);update(phase='cohort_complete_all_science_and_resource_audits')
   finals=[json.loads((H/('candidate_veto_cup_bowls_v200_'+str(i))/'final.json').read_text())for i in range(2)]
   cases=sum([f['cases']for f in finals],[]);assert len(cases)==80 and len({(c['task'],c['seed'])for c in cases})==80
   metrics={}
   for task in ['place_empty_cup','stack_bowls_two']:
    rows=[c for c in cases if c['task']==task];assert len(rows)==40
    metrics[task]=dict(scenes=40,baseline=sum(c['baseline_success']for c in rows),event_full=sum(c['event_full_success']for c in rows),gate=sum(c['gate_success']for c in rows),harms_avoided=sum(c['baseline_success']and not c['event_full_success']and c['gate_success']for c in rows),rescues_lost=sum(not c['baseline_success']and c['event_full_success']and not c['gate_success']for c in rows),B_chunks=sum(c['event_full_active_chunks']for c in rows),G_chunks=sum(c['gate_active_chunks']for c in rows))
   atomic(H/'cup_bowls_transfer_v200_final.json',dict(passed=True,complete=True,metrics=metrics,cases=cases,not_100_episode_claim=True,no_fit_or_threshold_tuning=True,cohort_final_sha256={str(i):sha(H/('candidate_veto_cup_bowls_v200_'+str(i))/'final.json')for i in range(2)}));update(phase='complete',goal_achieved=True,metrics=metrics)
  except BaseException as e:update(phase='stopped_requires_diagnosis',error=repr(e),whole_episodes_preserved=True);raise
if __name__=='__main__':main()

"""Actual frozen TEST G40; same original host, guards, and maintenance-only resume."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import benefit_followup_protocol_v2 as p
from benefit_followup_stream_v3 import atomic
from evaluate import host
import benefit_growth_runtime as runtime
STATE=p.ROOT/'mixed_validation_state_v17.json'

def whole():
 file=p.ROOT/'validate_plan.json'
 if not file.exists():return 0
 return sum(len(json.loads((host(j['run'])/'concurrency_queue.json').read_text())['completed'])for j in json.loads(file.read_text())['jobs']if(host(j['run'])/'concurrency_queue.json').exists())
def complete():
 file=p.ROOT/'validate_plan.json'
 if not file.exists():return False
 return all((host(j['run'])/'result.json').exists()and json.loads((host(j['run'])/'result.json').read_text()).get('complete')for j in json.loads(file.read_text())['jobs'])
def main():
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,required=True);args=a.parse_args()
 if p.BLOCK!='dev0':raise RuntimeError('Explicit fresh full-event validation scope required')
 folder=p.HOME/'candidate_veto_event_full_extension_GPU3_v174';plan=json.loads((folder/'prospective_plan.json').read_text())
 model=p.HOME/'fixed_ensemble_TRAIN472_v118/diagnostic_gate.json';target=p.ROOT/'gate.json'
 assert p.sha(model)==plan['model_sha256']and target.read_bytes()==model.read_bytes()
 extension=json.loads((p.ROOT/'full_event_control_extension_v134.json').read_text());assert extension['passed']and extension['complete']
 if not json.loads((p.ROOT/'mixed_membership_final_audit.json').read_text())['passed']:raise RuntimeError('Full TEST controls audit required')
 receipt=json.loads((p.ROOT/'finite_bootstrap_receipt_v6.json').read_text())
 if not receipt['passed']or receipt['complete40']or receipt['helper_sha256']!=p.sha(p.D/'benefit_followup_bootstrap_prefix_v6.py'):raise RuntimeError('Finite actual expert bootstrap missing or stale')
 p.prepare();runtime.install_sources()
 from shared6_eval_backend import sources
 from gpu_policy import inventory,require_admission
 validation_files=[p.D/n for n in ('benefit_event_full_entry_v174.py','benefit_event_full_resources_v174.py','benefit_event_full_policy_v134.py','benefit_event_full_G_collect_v174.py','benefit_event_full_audit_v134.py','benefit_event_full_veto_v132.py','benefit_fixed_ensemble_gate_v119.py','benefit_raw_features_TRAIN472_v108.py','benefit_cache_preference_host_v96.py')]+[folder/n for n in ('prospective_plan.json','implementation_v174.json')]+[p.ROOT/n for n in ('gate.json','protocol.json','cache_preference_v96.json','cache_provenance_CPU_v96.json','full_event_control_extension_v134.json')];validation_files.extend([Path(__file__),p.D/'benefit_event_full_policy_v142.py',p.ROOT.parent/'cache_selection_CPU_v156.json']);validation_files.extend([Path(__file__),p.D/'benefit_event_full_scope_audit_v175.py',p.ROOT/'full_event_resource_scope_CPU_proof_v175.json']);validation_closure={str(f):p.sha(f)for f in validation_files}
 source=sources();state=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],gpu=args.gpu,block=p.BLOCK,source_sha256=source)
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:16])
 lock=(p.ROOT/'validation_owner_v6.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 log=p.ROOT/('mixed_validation_pipeline_v17_'+str(time.time_ns())+'.log');state['log']=str(log)
 def update(**kw):state.update(updated_unix=time.time(),**kw);atomic(STATE,state)
 def check():
  if sources()!=source:raise RuntimeError('Mixed runtime changed; refuse new GPU work')
  if any(p.sha(Path(f))!=h for f,h in validation_closure.items()):raise RuntimeError('Frozen validation identity changed')
 def external(argv,stream):
  check()
  if Path(argv[0]).name=='benefit_growth_host_v3.py':argv=[p.D/'benefit_cache_preference_host_v96.py',*argv[1:]]
  proc=subprocess.Popen([sys.executable,'-B',str(p.D/'benefit_event_full_resources_v174.py'),*map(str,argv)],stdout=stream,stderr=subprocess.STDOUT)
  update(child_pid=proc.pid,child_start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21])
  if proc.wait():raise RuntimeError('Mixed child failed; diagnose before retry: '+str(argv[0]))
 try:
  update(phase='checking_actual_idle_GPU')
  fleet=inventory();card=next(c for c in fleet['cards']if c['index']==args.gpu);require_admission(fleet,card['uuid'])
  pin=p.ROOT/'collect_gpu.json'
  if pin.exists():
   if json.loads(pin.read_text())!=dict(index=args.gpu,uuid=card['uuid']):raise RuntimeError('Existing actual-card identity cannot be moved')
  else:atomic(pin,dict(index=args.gpu,uuid=card['uuid']))
  amendment=p.HOME/'inactive_alias_validation_amendment.json'
  if not amendment.exists():amendment.write_bytes((p.PRIOR/'inactive_alias_validation_amendment.json').read_bytes())
  with log.open('a')as stream:
   proof=json.loads((p.ROOT/'mixed_runtime_cpu_proof.json').read_text());contract=json.loads((p.ROOT/'growth_cpu_contract.json').read_text());deferred=json.loads((p.HOME/'deferred_finalize_cpu_proof.json').read_text())
   if any(not x['passed']or x['source_sha256']!=source for x in(proof,contract,deferred)):raise RuntimeError('Mixed CPU proof missing or stale')
   if proof['checker_sha256']!=p.sha(p.D/'verify_benefit_followup_mixed_cpu_v5.py')or contract['checker_sha256']!=p.sha(p.D/'benefit_followup_mixed_contract_v5.py')or deferred['checker_sha256']!=p.sha(p.D/'verify_benefit_followup_mixed_deferred_v4.py'):raise RuntimeError('Mixed CPU checker changed')
   command=[p.D/'benefit_growth_host_v3.py','--gpu',str(args.gpu),'--stage','validate']
   for batch in range(12):
    if complete():break
    before=whole();update(phase='CPU_preflight',memory_batch=batch,whole_before=before)
    cpu_file=p.ROOT/'full_cpu_proof.json';cpu=json.loads(cpu_file.read_text())if cpu_file.exists()else{}
    if cpu.get('task')=='benefit40_collect':
     archive=p.ROOT/'full_cpu_proof_collect_before_G_v6.json'
     if not archive.exists():archive.write_bytes(cpu_file.read_bytes())
    if not(cpu.get('passed')and cpu.get('source_sha256')==source and cpu.get('gpu_uuid')==card['uuid']and cpu.get('task')=='benefit40_validate'and cpu.get('host_sha256')==p.sha(p.D/'benefit_growth_host_v3.py')):external([*command,'--cpu-proof'],stream)
    update(phase='actual_guarded_real_G40')
    external(command,stream)
    if complete():break
    h=json.loads((p.ROOT/'host_state.json').read_text());after=whole()
    if h['phase']!='maintenance_batch_complete'or not h.get('normal_drained_exit')or after<=before:raise RuntimeError('Incomplete non-maintenance exit or no genuine progress; diagnose')
    update(phase='normal_memory_drain_same_identity_resume',whole_after=after)
   if not complete():raise RuntimeError('Bounded maintenance batches exhausted')
   update(phase='GPU_released_full_CPU_G40_audit',whole=whole())
   external([p.D/'benefit_followup_cpu_host_v6.py','--block','dev0','--gpu-uuid',card['uuid'],'benefit_event_full_scope_audit_v175.py','validate'],stream)
   update(phase='actual_G40_complete_and_audited',whole=whole(),qualification_complete=True)
 except BaseException as e:update(phase='stopped_requires_diagnosis',error=repr(e),whole_results_preserved=True);raise
if __name__=='__main__':main()

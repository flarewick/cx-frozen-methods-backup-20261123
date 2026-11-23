"""Prospective DEV/TEST controls; original guarded execution and strict audit."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import benefit_followup_protocol_v2 as p
from benefit_followup_stream_v3 import atomic
from evaluate import host
import benefit_growth_runtime as runtime
STATE=p.ROOT/'mixed_collector_state.json'

def whole():
 file=p.ROOT/'collect_plan.json'
 if not file.exists():return 0
 return sum(len(json.loads((host(j['run'])/'concurrency_queue.json').read_text())['completed'])for j in json.loads(file.read_text())['jobs']if(host(j['run'])/'concurrency_queue.json').exists())
def complete():
 file=p.ROOT/'collect_plan.json'
 if not file.exists():return False
 return all((host(j['run'])/'result.json').exists()and json.loads((host(j['run'])/'result.json').read_text()).get('complete')for j in json.loads(file.read_text())['jobs'])
def main():
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,required=True);a.add_argument('--cpu-proof',action='store_true');args=a.parse_args()
 if p.BLOCK=='train0':raise RuntimeError('Preserve current v4 collector for original train0; v5 is future cohorts only')
 receipt=json.loads((p.ROOT/'finite_bootstrap_receipt_v6.json').read_text())
 if not receipt['passed']or receipt['complete40']or receipt['helper_sha256']!=p.sha(p.D/'benefit_followup_bootstrap_prefix_v6.py'):raise RuntimeError('Finite actual expert bootstrap missing or stale')
 p.prepare();runtime.install_sources()
 slot_proof=json.loads((p.HOME/'expert_affinity_CPU_proof_v10.json').read_text());wire=json.loads((p.HOME/'policy_capacity_entry_host_CPU_proof_v12.json').read_text())
 if not slot_proof['passed']or slot_proof['helper_sha256']!=p.sha(p.D/'benefit_expert_affinity_v10.py')or not wire['passed']or wire['entry_sha256']!=p.sha(p.D/'benefit_followup_capacity_entry_v12.py'):raise RuntimeError('SLOTS CPU or child-route proof stale')
 from shared6_eval_backend import sources
 from gpu_policy import inventory,require_admission
 resource_files=[p.D/n for n in ('benefit_followup_expert_visual_entry_v16.py','benefit_followup_mixed_collect_v17.py','benefit_policy_capacity_v12.py','benefit_sampler_gc_v11.py','benefit_expert_affinity_v10.py')]+[p.HOME/n for n in ('policy_capacity_CPU_proof_v12.json','policy_capacity_resource_amendment_v12.json','sampler_GC_CPU_proof_v11.json','sampler_GC_resource_amendment_v11.json','expert_affinity_resource_amendment_v10.json')]
 resource_files.extend([Path(__file__),p.D/'benefit_supplement_entry_v24.py',p.HOME/'supplement_v24/experiment.json'])
 resource_closure={str(f):p.sha(f)for f in resource_files}
 from benefit_followup_expert_visual_entry_v16 import evidence
 resource_files.extend(p.R/f for f in evidence(p))
 resource_files.extend([p.D/n for n in ('benefit_followup_policy_first_entry_v17.py','benefit_owner_policy_first_v17.py','verify_benefit_owner_policy_first_cpu_v17.py')]+[p.HOME/n for n in ('owner_policy_first_CPU_proof_v17.json','owner_policy_first_resource_amendment_v17.json')])
 resource_files.extend([Path(__file__),p.D/'benefit_supplement_entry_v24.py',p.HOME/'supplement_v24/experiment.json'])
 resource_closure={str(f):p.sha(f)for f in resource_files}
 audit_files=[p.D/n for n in('benefit_supplement_alias_audit_v27.py','benefit_supplement_full_audit_v27.py','verify_benefit_supplement_alias_cpu_v27.py')]+[p.HOME/'supplement_v24'/n for n in('inactive_alias_amendment_v27.json','inactive_alias_CPU_proof_v27.json')]
 resource_files.extend(audit_files)
 resource_closure={str(f):p.sha(f)for f in resource_files}
 validation_files=[p.D/'benefit_event_full_resources_v174.py',p.HOME/'candidate_veto_event_full_extension_GPU3_v174/implementation_v174.json',p.HOME/'candidate_veto_event_full_extension_GPU3_v174/prospective_plan.json',p.HOME/'fixed_ensemble_TRAIN472_v118/TRAIN_selected_candidate.json',p.HOME/'fixed_ensemble_TRAIN472_v118/training_result.json']
 resource_files.extend(validation_files+[p.D/'benefit_event_full_policy_v142.py',p.ROOT.parent/'cache_selection_CPU_v156.json',p.D/'benefit_event_full_audit_v134.py',p.D/'benefit_event_full_policy_v134.py',p.D/'benefit_event_full_veto_v132.py'])
 resource_files.extend([p.D/n for n in ('benefit_policy_capacity_v45.py','benefit_event_full_resources_v174.py','verify_benefit_policy_capacity_cpu_v45.py')]+[p.HOME/'cohort_ensemble_v44/resource_capacity_v45'/n for n in ('CPU_proof.json','resource_amendment.json','retained14_before_resume.json')])
 resource_closure={str(f):p.sha(f)for f in resource_files}
 resource_files.extend([p.D/'benefit_fixed_ensemble_gate_v119.py',p.D/'benefit_raw_features_TRAIN472_v108.py',p.D/'benefit_cache_preference_host_v96.py',p.ROOT/'cache_preference_v96.json',p.ROOT/'cache_provenance_CPU_v96.json']);resource_closure={str(f):p.sha(f)for f in resource_files}
 source=sources();state=dict(resource_source_sha256=resource_closure,expert_affinity_helper_sha256=p.sha(p.D/'benefit_expert_affinity_v10.py'),expert_affinity_amendment_sha256=p.sha(p.HOME/'expert_affinity_resource_amendment_v10.json'),pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],gpu=args.gpu,block=p.BLOCK,source_sha256=source)
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:16])
 lock=(p.ROOT/'collection_owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 log=p.ROOT/('mixed_pipeline_'+str(time.time_ns())+'.log');state['log']=str(log)
 def update(**kw):state.update(updated_unix=time.time(),**kw);atomic(STATE,state)
 def check():
  if sources()!=source:raise RuntimeError('Mixed runtime changed; refuse new GPU work')
  if any(p.sha(Path(f))!=v for f,v in resource_closure.items()):raise RuntimeError('Resource workflow changed; diagnose before new GPU work')
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
   command=[p.D/'benefit_growth_host_v3.py','--gpu',str(args.gpu),'--stage','collect']
   receipt_file=p.ROOT/'SEVEN_POLICY_actual_full_CPU_receipt_v45.json'
   cache=json.loads(receipt_file.read_text())if receipt_file.exists()else{}
   valid=cache.get('policy_first_helper_sha256')==p.sha(p.D/'benefit_owner_policy_first_v17.py')and cache.get('policy_first_CPU_proof_sha256')==p.sha(p.HOME/'owner_policy_first_CPU_proof_v17.json')and cache.get('policy_first_amendment_sha256')==p.sha(p.HOME/'owner_policy_first_resource_amendment_v17.json')and cache.get('capacity_helper_sha256')==p.sha(p.D/'benefit_policy_capacity_v12.py')and cache.get('capacity_CPU_proof_sha256')==p.sha(p.HOME/'policy_capacity_CPU_proof_v12.json')and cache.get('capacity_amendment_sha256')==p.sha(p.HOME/'policy_capacity_resource_amendment_v12.json')and cache.get('sampler_GC_helper_sha256')==p.sha(p.D/'benefit_sampler_gc_v11.py')and cache.get('sampler_GC_CPU_proof_sha256')==p.sha(p.HOME/'sampler_GC_CPU_proof_v11.json')and cache.get('passed')and cache.get('full_CPU_proof_sha256')==p.sha(p.ROOT/'full_cpu_proof.json')and cache.get('entry_sha256')==p.sha(p.D/'benefit_followup_policy_first_entry_v17.py')and cache.get('helper_sha256')==p.sha(p.D/'benefit_expert_affinity_v10.py')and cache.get('amendment_sha256')==p.sha(p.HOME/'expert_affinity_resource_amendment_v10.json')
   if not valid:
    before_policy={str(f)for f in(p.ROOT/'policy_first_entry_receipts_v17').glob('*.json')}
    before_visual={str(f)for f in(p.ROOT/'expert_visual_entry_receipts_v16').glob('*.json')}
    before_receipts={str(f)for f in(p.ROOT/'capacity_entry_receipts_v12').glob('*.json')}
    external([*command,'--cpu-proof'],stream)
    receipts=[f for f in(p.ROOT/'capacity_entry_receipts_v12').glob('*.json')if str(f)not in before_receipts]
    if not receipts:raise RuntimeError('Actual SLOTS no-CUDA prelude receipt missing')
    visual_receipts=[f for f in(p.ROOT/'expert_visual_entry_receipts_v16').glob('*.json')if str(f)not in before_visual]
    assert visual_receipts
    policy_receipts=[f for f in(p.ROOT/'policy_first_entry_receipts_v17').glob('*.json')if str(f)not in before_policy]
    assert policy_receipts
    for f in policy_receipts:
     value=json.loads(f.read_text());assert value['entry_sha256']==p.sha(p.D/'benefit_followup_policy_first_entry_v17.py')and value['installed_initializer_module']=='benefit_owner_policy_first_v17'and value['actual_execute_wrapper_module']=='benefit_sampler_gc_v11'
    for f in visual_receipts:
     value=json.loads(f.read_text());assert value['entry_sha256']==p.sha(p.D/'benefit_followup_expert_visual_entry_v16.py')and value['evidence_sha256']==evidence(p)
    for file in receipts:
     value=json.loads(file.read_text())
     if value['entry_sha256']!=p.sha(p.D/'benefit_followup_capacity_entry_v12.py')or value['actual_execute_wrapper_module']!='benefit_sampler_gc_v11':raise RuntimeError('Slot child did not retain the original model owner')
    atomic(receipt_file,dict(passed=True,policy_first_helper_sha256=p.sha(p.D/'benefit_owner_policy_first_v17.py'),policy_first_CPU_proof_sha256=p.sha(p.HOME/'owner_policy_first_CPU_proof_v17.json'),policy_first_amendment_sha256=p.sha(p.HOME/'owner_policy_first_resource_amendment_v17.json'),capacity_helper_sha256=p.sha(p.D/'benefit_policy_capacity_v12.py'),capacity_CPU_proof_sha256=p.sha(p.HOME/'policy_capacity_CPU_proof_v12.json'),capacity_amendment_sha256=p.sha(p.HOME/'policy_capacity_resource_amendment_v12.json'),sampler_GC_helper_sha256=p.sha(p.D/'benefit_sampler_gc_v11.py'),sampler_GC_CPU_proof_sha256=p.sha(p.HOME/'sampler_GC_CPU_proof_v11.json'),full_CPU_proof_sha256=p.sha(p.ROOT/'full_cpu_proof.json'),entry_receipt_sha256={str(f):p.sha(f)for f in receipts},entry_sha256=p.sha(p.D/'benefit_followup_policy_first_entry_v17.py'),amendment_sha256=p.sha(p.HOME/'expert_affinity_resource_amendment_v10.json'),helper_sha256=p.sha(p.D/'benefit_expert_affinity_v10.py')))
   if args.cpu_proof:
    update(phase='CPU_prepared_without_GPU',no_GPU_launched=True);return
   for batch in range(12):
    if complete():break
    before=whole();update(phase='CPU_preflight',memory_batch=batch,whole_before=before)
    cpu_file=p.ROOT/'full_cpu_proof.json';cpu=json.loads(cpu_file.read_text())if cpu_file.exists()else{}
    if not(cpu.get('passed')and cpu.get('source_sha256')==source and cpu.get('gpu_uuid')==card['uuid']and cpu.get('task')=='benefit40_collect'and cpu.get('host_sha256')==p.sha(p.D/'benefit_growth_host_v3.py')):external([*command,'--cpu-proof'],stream)
    update(phase='actual_guarded_mixed_collection')
    external(command,stream)
    host_record=json.loads((p.ROOT/'host_state.json').read_text());barrier=Path(host_record['barrier'])
    visual=json.loads((barrier/'expert_visual_host_schedule_v16.json').read_text());assert visual['evidence_sha256']==evidence(p)
    policy=json.loads((barrier/'policy_first_host_schedule_v17.json').read_text());assert policy['helper_sha256']==p.sha(p.D/'benefit_owner_policy_first_v17.py')and policy['amendment_sha256']==p.sha(p.HOME/'owner_policy_first_resource_amendment_v17.json')
    schedule=json.loads((barrier/'policy_capacity_host_schedule_v12.json').read_text())
    if schedule['helper_sha256']!=p.sha(p.D/'benefit_policy_capacity_v12.py')or schedule['amendment_sha256']!=p.sha(p.HOME/'policy_capacity_resource_amendment_v12.json'):raise RuntimeError('Actual GPU scheduling receipt missing or changed')
    if complete():break
    h=json.loads((p.ROOT/'host_state.json').read_text());after=whole()
    if h['phase']!='maintenance_batch_complete'or not h.get('normal_drained_exit')or after<=before:raise RuntimeError('Incomplete non-maintenance exit or no genuine progress; diagnose')
    update(phase='normal_memory_drain_same_identity_resume',whole_after=after)
   if not complete():raise RuntimeError('Bounded maintenance batches exhausted')
   update(phase='GPU_released_full_CPU_collection_audit',whole=whole())
   external([p.D/'benefit_followup_cpu_host_v6.py','--block','dev0','--gpu-uuid',card['uuid'],'benefit_event_full_scope_audit_v175.py','controls'],stream)
   update(phase='cohort88_complete_and_audited',whole=whole(),qualification_complete=True)
 except BaseException as e:update(phase='stopped_requires_diagnosis',error=repr(e),whole_results_preserved=True);raise
if __name__=='__main__':main()

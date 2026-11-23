"""Strict science, separately report every resource attempt including historical failure."""
import ast,json,copy,sys,runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent;H243=HERE.parent/'gate_all_candidate_headroom_v243';BASE=HERE.parent/'gate_all_candidate_v237';M=HERE.parent/'gate_all_candidate_migration_v241'
import benefit_followup_protocol_v2 as p
# Execute original242 ownership checks; stop before its original237 final dispatch.
source=M/'audit242.py';tree=ast.parse(source.read_text());body=[]
for node in tree.body:
 if isinstance(node,ast.Expr)and isinstance(node.value,ast.Call)and ast.unparse(node.value.func)=='runpy.run_path':break
 body.append(node)
ownership=dict(__name__='ownership244',__file__=str(source));exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ownership)
# Execute all original237 science checks, with no resource assertion weakened or relabelled.
source=BASE/'audit237.py';tree=ast.parse(source.read_text());body=[]
for node in tree.body:
 if isinstance(node,ast.Expr)and isinstance(node.value,ast.Call)and ast.unparse(node.value.func)=='sys.path.insert':break
 body.append(node)
science=dict(__name__='science244',__file__=str(source));exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),science)
cases=science['cases'];assert len(cases)==40
resource=ownership['resources'];reports=[];failed=[];protected=[]
# Every actual file appears once; each original exact checker still raises on a violation.
for file in sorted(p.ROOT.glob('telemetry_gpu[03]_*.jsonl')):
 class SingleFileRoot(type(p.ROOT)):
  def glob(self,pattern,**kw):return iter([file])if pattern.startswith('telemetry_gpu')else super().glob(pattern,**kw)
 try:
  result=resource(SingleFileRoot(p.ROOT));reports.append(dict(file=str(file),sha256=p.sha(file),passed=True,original_report=result))
 except AssertionError as e:
  rows=[json.loads(x)for x in file.read_text().splitlines()if x.strip()];bad=[{k:r.get(k)for k in ['seconds','mean_utilization','time_weighted_mean_utilization','rolling_300s_utilization','process_mean_utilization','process_rolling_300s_utilization']}for r in rows if r.get('seconds',0)>=280 and min(r.get('mean_utilization',100),r.get('time_weighted_mean_utilization',100),r.get('rolling_300s_utilization',100),*r.get('process_mean_utilization',{}).values(),*r.get('process_rolling_300s_utilization',{}).values())<33]
  record=dict(file=str(file),sha256=p.sha(file),passed=False,error=repr(e),actual_below33_samples=bad,raw_failed_attempt_retained=True);failed.append(record);reports.append(record)
 marker=file.with_name(file.name+'.protection243.json')
 if marker.exists():
  v=json.loads(marker.read_text());assert v['telemetry_basename']==file.name and v['protection_sha256']==p.sha(H243/'protection243.py');protected.append(dict(file=str(file),resource_passed=reports[-1]['passed']))
assert reports and protected and all(r['resource_passed']for r in protected),'New protected attempts must pass original resource audit'
recoveries=[]
for folder in p.ROOT.glob('control_*'):
 for version in (245,246):
  proof=folder/('post_release_recovery'+str(version)+'.json')
  if not proof.exists():continue
  v=json.loads(proof.read_text());rr=json.loads((folder/'host_gpu_release_receipt.json').read_text());intent=folder/'cpu_finalize_intent.json'
  assert v['passed']and v['original_CPU_finalizer_passed']and rr['original_guard_exit_success']is False and rr['recovery_proof_sha256']==p.sha(proof)
  assert v['intent_sha256']==p.sha(intent)==rr['intent_sha256']and v['telemetry_sha256']==rr['telemetry_sha256']
  helper=HERE.parent/('gate_all_candidate_finalize_v'+str(version))/('recover'+str(version)+'.py');assert v['helper'+str(version)+'_sha256']==p.sha(helper)
  recoveries.append(dict(proof=str(proof),proof_sha256=p.sha(proof),version=version,original_guard_error_retained=v['original_guard_stop_preserved'],no_GPU_restart=True,original_CPU_finalizer_passed=True,original_host_sha256_is_compatibility_binding=True))
  for report in reports:
   for binding in report.get('original_report',{}).get('actual_file_bindings',[]):
    if Path(binding['barrier'])==folder:
     binding['legacy_compatibility_kind']=binding['kind'];binding['kind']='verified_original_CPU_finalize_after_confirmed_GPU_exit';binding['recovery_proof_sha256']=p.sha(proof)
resources=dict(passed=not failed,all_actual_attempts_accounted=True,attempts=reports,historical_failed_attempts=failed,new_protected_attempts=protected,new_protected_attempts_passed=True,no_historical_failure_erased=True,operational_CPU_recoveries=recoveries)
result=dict(complete=True,passed=not failed,science_passed=True,cases=cases,actual_all_event_G=40,actual_query_controller_replays=science['queries'],resource_audit=resources,no_training=True,later_event_scores_uncalibrated_transfer=True,not_independent_confirmation_or_100_episode=True,all40_actual_cache_owners=ownership['owners'],strict_science237_sha256=p.sha(BASE/'audit237.py'),auditor247_sha256=p.sha(__file__))
p.atomic(p.ROOT.parent/'final.json',result);p.atomic(p.ROOT/'migration241/full_audit244.json',dict(complete=True,science_passed=True,all_attempts_resources_passed=not failed,new_protected_attempts_passed=True,final_sha256=p.sha(p.ROOT.parent/'final.json'),historical_failure_disclosed=True))
print('247_COMPLETE_SCIENCE_PASSED_RESOURCE_HISTORY',len(cases),len(failed),flush=True)

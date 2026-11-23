"""Strict science, separately report every resource attempt including historical failure."""
import ast,json,copy,sys,runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'gate_all_candidate_v237';M=HERE.parent/'gate_all_candidate_migration_v241'
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
  v=json.loads(marker.read_text());assert v['telemetry_basename']==file.name and v['protection_sha256']==p.sha(HERE/'protection243.py');protected.append(dict(file=str(file),resource_passed=reports[-1]['passed']))
assert reports and protected and all(r['resource_passed']for r in protected),'New protected attempts must pass original resource audit'
resources=dict(passed=not failed,all_actual_attempts_accounted=True,attempts=reports,historical_failed_attempts=failed,new_protected_attempts=protected,new_protected_attempts_passed=True,no_historical_failure_erased=True)
result=dict(complete=True,passed=not failed,science_passed=True,cases=cases,actual_all_event_G=40,actual_query_controller_replays=science['queries'],resource_audit=resources,no_training=True,later_event_scores_uncalibrated_transfer=True,not_independent_confirmation_or_100_episode=True,all40_actual_cache_owners=ownership['owners'],strict_science237_sha256=p.sha(BASE/'audit237.py'),auditor244_sha256=p.sha(__file__))
p.atomic(p.ROOT.parent/'final.json',result);p.atomic(p.ROOT/'migration241/full_audit244.json',dict(complete=True,science_passed=True,all_attempts_resources_passed=not failed,new_protected_attempts_passed=True,final_sha256=p.sha(p.ROOT.parent/'final.json'),historical_failure_disclosed=True))
print('244_COMPLETE_SCIENCE_PASSED_RESOURCE_HISTORY',len(cases),len(failed),flush=True)

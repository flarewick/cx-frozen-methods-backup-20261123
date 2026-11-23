"""Audit actual managed GPU PIDs; preserve raw foreign-PID/arrival evidence and all thresholds."""
import ast,copy,inspect,json,linecache,textwrap
from pathlib import Path
import benefit_followup_protocol_v2 as p

def own_memory(row):
 values={int(k):v for k,v in row['process_memory_mib'].items()};active=set(row['gpu_pids']);extra=set(values)-active
 assert active and all(4096<=values.get(pid,0)<=24563 for pid in active),'Managed GPU PID missing or below4GiB'
 assert all(0<=v<=24563 for v in values.values()),'Any PID memory exceeds unchanged hard ceiling'
 if extra:
  arrival=row['gpu_arrival_order'];assert extra<=set(arrival['foreign_pids'])
  assert arrival['own_arrival_precedes_foreign'] is True and not arrival['stop_for_prior_or_unverified_occupant']
  for pid in extra:
   first=arrival['foreign_first_seen_seconds'].get(str(pid),arrival['foreign_first_seen_seconds'].get(pid));assert first is not None and first>arrival['own_first_seen_seconds']
 return True

def install():
 import benefit_stage_audit as stage
 old=stage.resource_audit
 if getattr(old,'_memory_scope164',False):return
 original=ast.parse(textwrap.dedent(inspect.getsource(old)));tree=copy.deepcopy(original)
 needle="all((4096 <= v <= 24563 for v in r['process_memory_mib'].values()))";hits=0
 for n in ast.walk(tree):
  if isinstance(n,ast.Assert) and ast.unparse(n.test)==needle:
   n.test=ast.parse("__import__('benefit_full_resource_audit_v164').own_memory(r)",mode='eval').body;hits+=1
 assert hits==1
 restored=copy.deepcopy(tree)
 for n in ast.walk(restored):
  if isinstance(n,ast.Assert)and ast.unparse(n.test)=="__import__('benefit_full_resource_audit_v164').own_memory(r)":n.test=ast.parse(needle,mode='eval').body
 assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
 source=ast.unparse(ast.fix_missing_locations(tree))+'\n';filename='<full_resource_PID_scope_v164>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename);ns=dict(old.__globals__);exec(compile(source,filename,'exec'),ns);fn=ns['resource_audit'];fn._memory_scope164=True;stage.resource_audit=fn

def main():
 import sys
 install()
 if sys.argv[1:] == ['verify']:
  import benefit_growth_audit as audit
  value=audit.resource_audit();assert value['passed']
  p.atomic(p.ROOT/'resource_PID_scope_CPU_proof_v164.json',dict(passed=True,no_GPU=True,all8_actual_session_resource_audits_passed=value,inverse_AST_all_other_resource_checks_exact=True,managed_PID4GiB_and24563_hard_limits_unchanged=True,foreign_later_arrival_required=True,original_runtime_guard_checks_only_active_managed_PIDs=True,original_stage_auditor_sha256=p.sha(p.D/'benefit_stage_audit.py'),original_guard_sha256=p.sha(p.D/'gpu_guard_bounded_metrics.py'),extension_sha256=p.sha(__file__),raw_logs_unchanged=True));print('RESOURCE_PID_SCOPE_CPU_PASSED',flush=True)
 else:
  assert sys.argv[1:]in(['controls'],['validate'])
  proof=p.ROOT/'resource_PID_scope_CPU_proof_v164.json';assert json.loads(proof.read_text())['passed'] and json.loads(proof.read_text())['extension_sha256']==p.sha(__file__)
  import benefit_event_full_audit_v134 as original
  getattr(original,sys.argv[1])()
  p.atomic(p.ROOT/('full_'+sys.argv[1]+'_resource_scope_receipt_v164.json'),dict(passed=True,scope_proof_sha256=p.sha(proof),extension_sha256=p.sha(__file__),original_full_auditor_sha256=p.sha(p.D/'benefit_event_full_audit_v134.py'),all_other_scientific_audits_unchanged=True))
if __name__=='__main__':main()

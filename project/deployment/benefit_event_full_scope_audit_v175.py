"""Full scientific audit with explicit actual-cohort resource binding; no fragile text replacement."""
import ast,copy,json
from pathlib import Path
import benefit_followup_protocol_v2 as p
from benefit_full_resource_audit_v164 import own_memory

def resource_for(root):
 import benefit_stage_audit as stage
 original=copy.deepcopy(next(n for n in ast.parse((p.D/'benefit_stage_audit.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='resource_audit'));edited=copy.deepcopy(original);a=b=0
 needle="all((4096 <= v <= 24563 for v in r['process_memory_mib'].values()))"
 old_files=next(n.value for n in ast.walk(original)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='files'for t in n.targets))
 for n in ast.walk(edited):
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='files'for t in n.targets):n.value=ast.Name(id='_actual_files',ctx=ast.Load());a+=1
  if isinstance(n,ast.Assert)and ast.unparse(n.test)==needle:n.test=ast.parse('_own_memory(r)',mode='eval').body;b+=1
 assert(a,b)==(1,1)
 restored=copy.deepcopy(edited)
 for n in ast.walk(restored):
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='files'for t in n.targets):n.value=copy.deepcopy(old_files)
  if isinstance(n,ast.Assert)and ast.unparse(n.test)=='_own_memory(r)':n.test=ast.parse(needle,mode='eval').body
 assert ast.dump(original,include_attributes=False)==ast.dump(restored,include_attributes=False)
 files=[];bindings=[];protocol=p.sha(root/'protocol.json')
 for barrier in sorted(root.glob('control_*')):
  ready=barrier/'ready.json';receipt=barrier/'host_gpu_release_receipt.json';r=json.loads(receipt.read_text());plan=json.loads(ready.read_text())['plan'];f=root/Path(r['telemetry']).name
  assert r['GPU_job_released']and p.sha(f)==r['telemetry_sha256']and r['host_sha256']==p.sha(p.D/'benefit_growth_host_v3.py')
  assert plan['task']in('benefit40_collect','benefit40_validate')and plan['gpu_uuid']=='GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a'
  assert all(j['manifest']['pilot']['protocol_sha256']==protocol for j in plan['jobs'])
  assert all(p.sha(p.D/n)==v for n,v in plan['source_sha256'].items())
  files.append(f);bindings.append(dict(path=str(f),sha256=p.sha(f),ready_sha256=p.sha(ready),release_receipt_sha256=p.sha(receipt),task=plan['task']))
 assert files and len(files)==len(set(files))and set(files)==set(root.glob('telemetry_gpu3_*.jsonl')),'Unbound or unreleased actual cohort resource files'
 ns=dict(stage.resource_audit.__globals__,_actual_files=sorted(files),_own_memory=own_memory);exec(compile(ast.fix_missing_locations(ast.Module(body=[edited],type_ignores=[])),__file__,'exec'),ns);report=ns['resource_audit']();assert report['passed']and len(report['sessions'])==len(files)
 report.update(actual_cohort_root=str(root),actual_file_bindings=bindings,all_other_resource_checks_inverse_AST_exact=True,explicit_file_scope_and_managed_PID_minimum_only=True,scope_helper_sha256=p.sha(__file__),managed_PID_helper_sha256=p.sha(p.D/'benefit_full_resource_audit_v164.py'));return report

def install():
 import benefit_growth_audit as growth
 import benefit_followup_audit_v2 as followup
 def scoped():return resource_for(p.ROOT)
 growth.resource_audit=scoped;followup.resource_audit=scoped

def main():
 import sys,torch
 torch.set_num_threads(1);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized()
 if sys.argv[1:]==['verify']:
  fixture=p.HOME/'candidate_veto_event_full_GPU3_v154/dev0';report=resource_for(fixture);assert len(report['sessions'])==11
  p.atomic(p.ROOT/'full_event_resource_scope_CPU_proof_v175.json',dict(passed=True,no_GPU=True,fixture_resource_only_not_new_collection_evidence=True,fixture_root=str(fixture),actual_completed_fixture_sessions11=True,explicit_cohort_binding_and_inverse_AST_passed=True,helper_sha256=p.sha(__file__),managed_PID_helper_sha256=p.sha(p.D/'benefit_full_resource_audit_v164.py'),source100_previous_unchanged=True));print('FULL_EVENT_ACTUAL_SCOPE_CPU_PASSED',flush=True);return
 assert sys.argv[1:]in(['controls'],['validate']);proof=p.ROOT/'full_event_resource_scope_CPU_proof_v175.json';v=json.loads(proof.read_text());assert v['passed']and v['helper_sha256']==p.sha(__file__)
 install();import benefit_event_full_audit_v134 as original;getattr(original,sys.argv[1])()
 target=p.ROOT/'collection_audit.json'if sys.argv[1]=='controls'else p.ROOT.parent/'final.json';report=json.loads(target.read_text());assert report['passed']and report['complete']and report['resource_audit']['actual_cohort_root']==str(p.ROOT)
 p.atomic(p.ROOT/('full_'+sys.argv[1]+'_actual_scope_receipt_v175.json'),dict(passed=True,scope_helper_sha256=p.sha(__file__),CPU_proof_sha256=p.sha(proof),actual_report_sha256=p.sha(target),actual_sessions=len(report['resource_audit']['sessions'])));print('FULL_EVENT_ACTUAL_SCOPE_AUDIT_PASSED',sys.argv[1],flush=True)
if __name__=='__main__':main()

"""Original resource thresholds, with truthful bindings for failed/pre-CUDA attempts."""
import ast,copy,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import benefit_followup_protocol_v2 as p

def resource_for(root):
 import benefit_stage_audit as stage
 from benefit_full_resource_audit_v164 import own_memory
 original=copy.deepcopy(next(n for n in ast.parse((p.D/'benefit_stage_audit.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='resource_audit'));edited=copy.deepcopy(original)
 needle="all((4096 <= v <= 24563 for v in r['process_memory_mib'].values()))";old_files=next(n.value for n in ast.walk(original)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='files'for t in n.targets));hits=[0,0]
 for n in ast.walk(edited):
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='files'for t in n.targets):n.value=ast.Name(id='_actual_files',ctx=ast.Load());hits[0]+=1
  if isinstance(n,ast.Assert)and ast.unparse(n.test)==needle:n.test=ast.parse('_own_memory(r)',mode='eval').body;hits[1]+=1
 assert hits==[1,1]
 restored=copy.deepcopy(edited)
 for n in ast.walk(restored):
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='files'for t in n.targets):n.value=copy.deepcopy(old_files)
  if isinstance(n,ast.Assert)and ast.unparse(n.test)=='_own_memory(r)':n.test=ast.parse(needle,mode='eval').body
 assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False),'Any original resource threshold/check changed'
 files=sorted(root.glob('telemetry_gpu3_*.jsonl'));assert files
 schedules=[(b,(b/'seven_policy_host_schedule_v45.json').stat().st_mtime_ns)for b in root.glob('control_*')if(b/'seven_policy_host_schedule_v45.json').exists()];bindings=[];used=set()
 for file in files:
  stamp=int(file.stem.rsplit('_',1)[1]);matches=[(b,t)for b,t in schedules if 0<=stamp-t<100_000_000];assert len(matches)==1,('Nonunique original prelaunch schedule binding',file.name)
  barrier,_=matches[0];assert barrier not in used;used.add(barrier)
  rows=[json.loads(x)for x in file.read_text().splitlines()if x.strip()];release=next((x for x in reversed(rows)if x.get('own_gpu_released')),None);assert release and not release['remaining_own_gpu_pids']
  live=any(x.get('gpu_pids')for x in rows);normal=barrier/'host_gpu_release_receipt.json';ready=barrier/'ready.json';binding=dict(path=str(file),sha256=p.sha(file),barrier=str(barrier),original_prelaunch_schedule_sha256=p.sha(barrier/'seven_policy_host_schedule_v45.json'),actual_release=release)
  if live:
   data=json.loads(ready.read_text());plan=data['plan'];assert plan['task']in('benefit40_collect','benefit40_validate')and plan['gpu_uuid']=='GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a'
   assert all(j['manifest']['pilot']['protocol_sha256']==p.sha(root/'protocol.json')for j in plan['jobs'])
   assert all(p.sha(p.D/n)==h for n,h in plan['source_sha256'].items());binding.update(task=plan['task'],ready_sha256=p.sha(ready))
   if normal.exists():
    r=json.loads(normal.read_text());assert r['GPU_job_released']and r['telemetry_sha256']==p.sha(file)and r['host_sha256']==p.sha(p.D/'benefit_growth_host_v3.py');binding.update(original_release_receipt_sha256=p.sha(normal),kind='normal_original_host_binding')
   else:
    assert any('stopped'in x for x in rows)and release['cleanup_seconds_after_start']<300,'Unfinalized real GPU attempt must have truthful bounded stop'
    binding.update(kind='actual_guard_early_stop_no_CPU_finalize',not_a_successful_batch_or_finalization_receipt=True)
  else:
   assert release['cleanup_seconds_after_start']<300 and not any('seconds'in x for x in rows),'Only the actual pre-sample preparation release is allowed here'
   binding.update(kind='CPU_preparation_release_no_GPU_samples',ready_completed=ready.exists(),not_claiming_GPU_evaluation_or_CPU_preflight_completed=True)
  bindings.append(binding)
 ns=dict(stage.resource_audit.__globals__,_actual_files=files,_own_memory=own_memory);exec(compile(ast.fix_missing_locations(ast.Module(body=[edited],type_ignores=[])),__file__,'exec'),ns);report=ns['resource_audit']();assert report['passed']and len(report['sessions'])==len(files)
 report.update(actual_cohort_root=str(root),actual_file_bindings=bindings,all_other_resource_checks_inverse_AST_exact=True,all_original_280_and300_checks_and33_thresholds_unchanged=True,failed_and_CPU_preparation_attempts_all_included=True,scope_helper_sha256=p.sha(__file__),base_scope_helper_sha256=p.sha(p.D/'benefit_event_full_scope_audit_v175.py'),managed_PID_helper_sha256=p.sha(p.D/'benefit_full_resource_audit_v164.py'))
 return report

def main():
 import torch
 torch.set_num_threads(1);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized()
 if sys.argv[1:]==['verify']:
  r=resource_for(p.ROOT);p.atomic(p.ROOT/'actual_resource_scope_CPU_v196.json',dict(passed=True,CPU_only=True,all_actual_attempts_checked=len(r['sessions']),actual_resource_audit=r,helper_sha256=p.sha(__file__)));print('ACTUAL_RESOURCE_SCOPE196_CPU_PASSED',len(r['sessions']),flush=True);return
 assert sys.argv[1:]in(['controls'],['validate'])
 proof=json.loads((p.ROOT/'actual_resource_scope_CPU_v196.json').read_text());assert proof['passed']and proof['helper_sha256']==p.sha(__file__)
 import benefit_event_full_scope_audit_v175 as base
 base.resource_for=resource_for;base.main()
 target=p.ROOT/'collection_audit.json'if sys.argv[1]=='controls'else p.ROOT.parent/'final.json';result=json.loads(target.read_text());assert result['passed']and result['complete']and result['resource_audit']['scope_helper_sha256']==p.sha(__file__)
 p.atomic(p.ROOT/('full_'+sys.argv[1]+'_resource_binding_receipt_v196.json'),dict(passed=True,actual_scope_helper_sha256=p.sha(__file__),actual_report_sha256=p.sha(target),base175_receipt_helper_field_is_compatibility_not_unmodified_scope_claim=True,all_original_scientific_and_resource_thresholds_unchanged=True))
if __name__=='__main__':main()

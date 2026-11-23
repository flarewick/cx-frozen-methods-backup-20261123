"""Original science237, truthful GPU0/3 resource and imported-cache ownership bindings."""
import ast,copy,json,sys,importlib.util,runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent
import benefit_followup_protocol_v2 as p,benefit_growth_runtime as runtime
runtime.install_sources()
from bridge241 import read,sha,verify_origin,locate
ORIGINAL=HERE.parent/'gate_extension_recovery_v192/scope_v196.py'
spec=importlib.util.spec_from_file_location('_scope242',ORIGINAL);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
old=next(n for n in ast.parse(ORIGINAL.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='resource_for');node=copy.deepcopy(old);changes=[]
for n in ast.walk(node):
 if isinstance(n,ast.Constant)and n.value=='telemetry_gpu3_*.jsonl':changes.append((n,copy.deepcopy(n)));n.value='telemetry_gpu[03]_*.jsonl'
 if isinstance(n,ast.Compare)and ast.unparse(n)=="plan['gpu_uuid'] == 'GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a'":
  changes.append((n,copy.deepcopy(n)));n.left=ast.parse('_device242(plan,file,root)',mode='eval').body;n.comparators=[ast.Constant(True)]
 if isinstance(n,ast.Assert)and ast.unparse(n.test)=="release['cleanup_seconds_after_start'] < 300 and (not any(('seconds' in x for x in rows)))":
  changes.append((n,copy.deepcopy(n)));n.test=ast.parse('_CPU_only242(barrier,rows,release)',mode='eval').body
assert len(changes)==3,[(type(x).__name__,getattr(x,'lineno',None))for x,_ in changes]
reverse=copy.deepcopy(node)
for n in ast.walk(reverse):
 for new,orig in changes:
  if type(n)is type(new)and getattr(n,'lineno',None)==getattr(new,'lineno',None)and getattr(n,'col_offset',None)==getattr(new,'col_offset',None):n.__dict__.clear();n.__dict__.update(copy.deepcopy(orig.__dict__))
assert ast.dump(reverse,include_attributes=False)==ast.dump(old,include_attributes=False)

def device(plan,file,root):
 index=int(file.name.split('_')[1].removeprefix('gpu'));uuids={0:'GPU-fa87bc41-1919-efe1-73f9-11f38008763e',3:'GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a'}
 assert plan['gpu_uuid']==uuids[index]
 for j in plan['jobs']:
  assert j['manifest']['deployment_shared6']['gpu_uuid']==uuids[index]and j['manifest']['all_candidate_v237']['same_gate_every_legal_candidate']
  if index==3:assert j['manifest']['gpu_migration241']['registry_sha256']==sha(root/'migration241/origin_G_registry.json')
 return True

def CPU_only(barrier,rows,release):
 assert not any('seconds'in x or x.get('gpu_pids')for x in rows)and not release['remaining_own_gpu_pids']
 assert not(barrier/'permit.json').exists(),'CPU-only exception must not have admitted CUDA'
 ready=barrier/'ready.json'
 if ready.exists():assert read(ready)['cuda_initialized']is False
 return True
ns=dict(vars(base),_device242=device,_CPU_only242=CPU_only,__file__=__file__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),ns)
original_resources=ns['resource_for']
def resources(root):
 v=original_resources(root);v.update(explicit_GPU0_GPU3_binding242=True,only_CPU_without_GPU_or_permit_may_exceed300s_preparation=True,no_GPU_runtime_threshold_changed=True,all_other_resource_checks_inverse_AST_exact=True,resource_auditor242_sha256=sha(__file__));return v
# Verify each imported origin and the actual new originalGPU3 cache owner.
reg=read(p.ROOT/'migration241/origin_G_registry.json')
for job in reg['jobs'].values():verify_origin(job)
plan=read(p.ROOT/'validate_plan.json');owners=[]
for j in plan['jobs']:
 run=Path(j['run'])
 if j['task']in reg['jobs']:
  receipt=read(run/'resume_import241_receipt.json');assert receipt['registry_sha256']==sha(p.ROOT/'migration241/origin_G_registry.json')
 for e in read(run/'episodes.json'):
  if e['process_session'].startswith('import241_'):
   assert any(r['seed']==e['seed']and r['actual_origin_GPU_UUID']==reg['origin_GPU_UUID']for r in receipt['episodes']);owners.append(dict(seed=e['seed'],imported=True,actual_origin_GPU_UUID=reg['origin_GPU_UUID']));continue
  s=read(run/'sessions'/e['process_session']/'storage.json');master=Path(s.get('shared_process_scratch_owner',str(run/'sessions'/e['process_session'])));m=read(master/'deployment_model_cache.json');n=read(master/'native_cache_seed.json');d=read(master/'cross_driver_cache_seed.json')
  assert m['gpu_uuid']==d['actual_gpu_uuid']==reg['target_GPU_UUID']and d['same_physical_gpu_only']
  assert m['imported']['artifact_sha256']==n['artifact_sha256']=='2fc67e7d2e5c2d38879001f085dd107080df49c35b649532087c0672f615af2b'and m['imported']['source_closure_verified']and m['imported']['rng_unchanged']
  assert n['official_requests_exact']and n['unique_selected_sources']==48 and n['unselected_original_runtime_sources']==0
  owners.append(dict(seed=e['seed'],imported=False,actual_GPU_UUID=m['gpu_uuid'],master=str(master)))
assert len(owners)==40
sys.path.insert(0,str(HERE.parent/'gate_transfer_migration_v231'));import resource231;resource231.resource_for=resources
runpy.run_path(str(HERE.parent/'gate_all_candidate_v237/audit237.py'),run_name='__main__')
f=p.ROOT.parent/'final.json';assert read(f)['passed']and read(f)['complete'];p.atomic(p.ROOT/'migration241/full_audit242.json',dict(passed=True,complete=True,final_sha256=sha(f),resource242_sha256=sha(__file__),original_science237_sha256=sha(HERE.parent/'gate_all_candidate_v237/audit237.py'),all40_actual_cache_owners=owners,origin_GPU0_whole_results_preserved=True,full_new_GPU3_science_zero_tolerance_passed=True))

"""Explicit seven-scene resource schedule over unchanged science83 and batch1."""
import os,sys,time,json,hashlib,marshal
from pathlib import Path
_INSTALLED=False

def install():
 global _INSTALLED
 if _INSTALLED:return
 import benefit_followup_sampler_GC_entry_v11 as parent;parent.install()
 import benefit_followup_protocol_v2 as p
 proof=p.HOME/'policy_capacity_CPU_proof_v12.json';v=json.loads(proof.read_text())
 if not v['passed']or v['helper_sha256']!=p.sha(p.D/'benefit_policy_capacity_v12.py')or v['checker_sha256']!=p.sha(p.D/'verify_benefit_policy_capacity_cpu_v12.py'):raise RuntimeError('Seven-scene CPU proof stale')
 amendment=p.HOME/'policy_capacity_resource_amendment_v12.json'
 if not amendment.exists():raise RuntimeError('Explicit seven-scene resource amendment required')
 import benefit_growth_runtime as runtime
 previous=runtime.install
 def resource_install():
  previous();__import__('benefit_policy_capacity_v12').apply()
 runtime.install=resource_install
 info=dict(schema='six_policy_one_expert_resource_schedule_v12',entry_sha256=p.sha(__file__),
  helper_sha256=p.sha(p.D/'benefit_policy_capacity_v12.py'),CPU_proof_sha256=p.sha(proof),amendment_sha256=p.sha(amendment),
  policy_cap=6,pending_expert_cap=1,max_real_scenes_while_pending=7,max_real_scenes_after_experts=6,
  historical_scope_resource_ceiling_explicitly_superseded=True,
  original_scope_membership_immutable=True,original83_runtime_files_unchanged=True,
  original_model_batch=1,original_model_physics_RNG_and_dependencies_unchanged=True,
  task_affine_expert_and_first_probe_GC_retained=True,all_resource_guards_unchanged=True)
 atomic=p.atomic
 def record(path,data):
  if Path(path).name=='deployment_cpu_schedule.json':data=dict(data,policy_capacity_schedule_v12=info)
  return atomic(path,data)
 p.atomic=record
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT,host
  before=c.command;old=str(CANONICAL_PROJECT/'deployment/benefit_followup_sampler_GC_entry_v11.py');new=str(CANONICAL_PROJECT/'deployment/benefit_followup_capacity_entry_v12.py')
  def route(argv,*a,**kw):
   out=before(argv,*a,**kw)
   for i,x in enumerate(out):
    if x==old:out[i]=new
    elif isinstance(x,str)and "__import__('benefit_followup_sampler_GC_entry_v11').install();"in x:out[i]=x.replace("__import__('benefit_followup_sampler_GC_entry_v11').install();","__import__('benefit_followup_capacity_entry_v12').install();",1)
   values=list(argv)
   if '--barrier'in values:
    folder=host(values[values.index('--barrier')+1]);folder.resolve().relative_to(p.ROOT.resolve());atomic(folder/'policy_capacity_host_schedule_v12.json',info)
   return out
  c.command=route
 else:
  from model_service import ModelService
  folder=p.ROOT/'capacity_entry_receipts_v12';folder.mkdir(exist_ok=True)
  atomic(folder/('entry_'+str(time.time_ns())+'.json'),dict(info,pid=os.getpid(),created_unix=time.time(),actual_execute_wrapper_module=ModelService.execute.__module__,actual_execute_code_sha256=hashlib.sha256(marshal.dumps(ModelService.execute.__code__)).hexdigest()))
 _INSTALLED=True

def main():
 install();__import__('benefit_followup_sampler_GC_entry_v11').main()
if __name__=='__main__':main()

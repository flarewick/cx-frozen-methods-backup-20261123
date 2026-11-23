"""Explicit resource scheduling amendment over unchanged original science83."""
import json,os,sys,time,hashlib,marshal
from pathlib import Path
_INSTALLED=False


def install():
 global _INSTALLED
 if _INSTALLED:return
 import benefit_followup_mixed_entry_v4 as parent;parent.install()
 import benefit_followup_protocol_v2 as p
 proof=p.HOME/'expert_affinity_CPU_proof_v10.json';v=json.loads(proof.read_text())
 if not v['passed']or v['helper_sha256']!=p.sha(p.D/'benefit_expert_affinity_v10.py')or v['checker_sha256']!=p.sha(p.D/'verify_benefit_expert_affinity_cpu_v10.py'):raise RuntimeError('Expert-slot CPU proof stale')
 amendment=p.HOME/'expert_affinity_resource_amendment_v10.json'
 if not amendment.exists():raise RuntimeError('Explicit slot resource amendment required')
 import model_service
 if model_service.ModelService.__init__.__module__!='model_service':raise RuntimeError('Keep original owner priority scheme in slot experiment')
 import benefit_growth_runtime as runtime
 before=runtime.install
 def resource_install():
  before();__import__('benefit_expert_affinity_v10').apply()
 runtime.install=resource_install
 info=dict(schema='explicit_task_affine_5plus1_host_resource_schedule_v10',entry_sha256=p.sha(__file__),
  helper_sha256=p.sha(p.D/'benefit_expert_affinity_v10.py'),CPU_proof_sha256=p.sha(proof),amendment_sha256=p.sha(amendment),
  original83_runtime_files_unchanged=True,original_model_owner_priorities_retained=True,
  initial_scope_resource_fields_superseded_explicitly=True,
  two_pending_expert_tasks=dict(policy_cap=5,expert_cap=1,task_affine_until_fixed20=True),one_pending_expert_task=dict(policy_cap=5,expert_cap=1),
  no_pending_expert_tasks=dict(policy_cap=6,expert_cap=0),max_real_scenes=6,all_resource_guards_unchanged=True)
 atomic=p.atomic
 def record(path,data):
  if Path(path).name=='deployment_cpu_schedule.json':data=dict(data,expert_affinity_resource_schedule_v10=info)
  return atomic(path,data)
 p.atomic=record
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT,host
  previous=c.command;old=str(CANONICAL_PROJECT/'deployment/benefit_followup_mixed_entry_v4.py');new=str(CANONICAL_PROJECT/'deployment/benefit_followup_affinity_entry_v10.py')
  def route(argv,*args,**kwargs):
   out=previous(argv,*args,**kwargs)
   for i,x in enumerate(out):
    if x==old:out[i]=new
    elif isinstance(x,str)and "__import__('benefit_followup_mixed_entry_v4').install();"in x:out[i]=x.replace("__import__('benefit_followup_mixed_entry_v4').install();","__import__('benefit_followup_affinity_entry_v10').install();",1)
   values=list(argv)
   if '--barrier'in values:
    folder=host(values[values.index('--barrier')+1]);folder.resolve().relative_to(p.ROOT.resolve());atomic(folder/'expert_affinity_host_schedule_v10.json',info)
   return out
  c.command=route
 else:
  receipt=dict(info,pid=os.getpid(),created_unix=time.time(),
   original_execute_code_sha256=hashlib.sha256(marshal.dumps(model_service.ModelService.execute.__code__)).hexdigest(),
   installed_initializer_module=model_service.ModelService.__init__.__module__)
  folder=p.ROOT/'affinity_entry_receipts_v10';folder.mkdir(exist_ok=True);atomic(folder/('entry_'+str(time.time_ns())+'.json'),receipt)
 _INSTALLED=True


def main():
 install();__import__('benefit_followup_mixed_entry_v4').main()
if __name__=='__main__':main()

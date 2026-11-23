"""Explicit host-only POLICY_FIRST scheduling amendment; original runtime83 stays frozen."""
import json,os,sys,time
from pathlib import Path
_INSTALLED=False


def install():
 global _INSTALLED
 if _INSTALLED:return
 import benefit_followup_expert_visual_entry_v16 as parent;parent.install()
 import benefit_followup_protocol_v2 as p
 proof=p.HOME/'owner_policy_first_CPU_proof_v17.json';value=json.loads(proof.read_text())
 if not value['passed']or value['helper_sha256']!=p.sha(p.D/'benefit_owner_policy_first_v17.py')or value['checker_sha256']!=p.sha(p.D/'verify_benefit_owner_policy_first_cpu_v17.py'):raise RuntimeError('POLICY_FIRST scheduling CPU proof stale')
 amendment=p.HOME/'owner_policy_first_resource_amendment_v17.json'
 if not amendment.exists():raise RuntimeError('Explicit resource-only scheduling amendment required')
 info=dict(schema='owner_request_policy_first_host_schedule_v17',entry_sha256=p.sha(__file__),
  helper_sha256=p.sha(p.D/'benefit_owner_policy_first_v17.py'),CPU_proof_sha256=p.sha(proof),
  amendment_sha256=p.sha(amendment),original83_runtime_unchanged=True,
  policy_slots=6,active_expert_slots=1,max_real_scenes=7,all_resource_guards_unchanged=True)
 import benefit_owner_policy_first_v17 as scheduler;scheduler.install()
 atomic=p.atomic
 def schedule_record(path,record):
  if Path(path).name=='deployment_cpu_schedule.json':record=dict(record,owner_policy_first_schedule_v17=info)
  return atomic(path,record)
 p.atomic=schedule_record
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT,host
  previous=c.command;old=str(CANONICAL_PROJECT/'deployment/benefit_followup_expert_visual_entry_v16.py');new=str(CANONICAL_PROJECT/'deployment/benefit_followup_policy_first_entry_v17.py')
  def route(argv,*args,**kwargs):
   out=previous(argv,*args,**kwargs)
   for i,v in enumerate(out):
    if v==old:out[i]=new
    elif isinstance(v,str)and "__import__('benefit_followup_expert_visual_entry_v16').install();"in v:out[i]=v.replace("__import__('benefit_followup_expert_visual_entry_v16').install();","__import__('benefit_followup_policy_first_entry_v17').install();",1)
   values=list(argv)
   if '--barrier'in values:
    folder=host(values[values.index('--barrier')+1]);folder.resolve().relative_to(p.ROOT.resolve())
    atomic(folder/'policy_first_host_schedule_v17.json',info)
   return out
  c.command=route
 else:
  # This stamp is written in the real no-CUDA prelude. Its fork inherits the
  # exact installed class; the original prelude still verifies one OS thread.
  import model_service
  receipt=dict(info,pid=os.getpid(),created_unix=time.time(),
   original_execute_code_sha256=scheduler.RECEIPT['original_code_sha256']['execute'],
   installed_initializer_module=model_service.ModelService.__init__.__module__,actual_execute_wrapper_module=model_service.ModelService.execute.__module__)
  folder=p.ROOT/'policy_first_entry_receipts_v17';folder.mkdir(exist_ok=True)
  atomic(folder/('entry_'+str(time.time_ns())+'.json'),receipt)
 _INSTALLED=True


def main():
 install();__import__('benefit_followup_expert_visual_entry_v16').main()
if __name__=='__main__':main()

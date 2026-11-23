"""Verified resource-only seven-policy overlay on new TRAIN cohorts."""
import json,os,sys,time
from pathlib import Path
_INSTALLED=False

def install():
 global _INSTALLED
 if _INSTALLED:return
 import benefit_event_full_entry_v174 as parent;parent.install()
 import benefit_followup_protocol_v2 as p
 out=p.HOME/'cohort_ensemble_v44/resource_capacity_v45';proof=out/'CPU_proof.json';amend=out/'resource_amendment.json';v=json.loads(proof.read_text());a=json.loads(amend.read_text())
 assert v['passed']and v['helper_sha256']==p.sha(p.D/'benefit_policy_capacity_v45.py')and v['checker_sha256']==p.sha(p.D/'verify_benefit_policy_capacity_cpu_v45.py')and a['CPU_proof_sha256']==p.sha(proof)
 schedule=p.HOME/'sampler_followup_v53/resource_schedule_v54';sp=json.loads((schedule/'CPU_proof.json').read_text());sa=json.loads((schedule/'resource_amendment.json').read_text())
 assert sp['passed']and sp['helper_sha256']==p.sha(p.D/'benefit_owner_setup_priority_v54.py')and sa['CPU_proof_sha256']==p.sha(schedule/'CPU_proof.json')
 __import__('benefit_owner_setup_priority_v54').install()
 cap=p.HOME/'resource_capacity_v89/CPU_proof.json';cv=json.loads(cap.read_text());assert cv['passed']and cv['helper_sha256']==p.sha(p.D/'benefit_policy_capacity_v89.py')and cv['checker_sha256']==p.sha(p.D/'verify_benefit_policy_capacity_cpu_v89.py')
 import benefit_growth_runtime as runtime
 previous=runtime.install
 def resource_install():previous();__import__('benefit_policy_capacity_v89').apply()
 runtime.install=resource_install
 info=dict(eight_policy_helper_sha256=cv['helper_sha256'],eight_policy_CPU_proof_sha256=p.sha(cap),capacity_semantics='8pending;6onfinalseal;8onnewqualifiedhost; inherited45descriptor superseded',schema='explicit_seven_policy_resource_overlay_v45',entry_sha256=p.sha(__file__),helper_sha256=p.sha(p.D/'benefit_policy_capacity_v45.py'),CPU_proof_sha256=p.sha(proof),amendment_sha256=p.sha(amend),policy_pending=7,policy_after_experts=6,expert=1,max_real_scenes=8,old_v12_capacity_explicitly_superseded=True,original83_and_all_resource_guards_unchanged=True)
 original=p.atomic
 def record(path,value):
  if Path(path).name=='deployment_cpu_schedule.json':value=dict(value,seven_policy_capacity_v45=info)
  return original(path,value)
 p.atomic=record
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT,host
  before=c.command;old=str(CANONICAL_PROJECT/'deployment/benefit_event_full_entry_v174.py');new=str(CANONICAL_PROJECT/'deployment/benefit_event_full_resources_v174.py')
  def route(argv,*args,**kwargs):
   cmd=before(argv,*args,**kwargs)
   for i,s in enumerate(cmd):
    if s==old:cmd[i]=new
    elif isinstance(s,str)and "__import__('benefit_event_full_entry_v174').install();"in s:cmd[i]=s.replace("__import__('benefit_event_full_entry_v174').install();","__import__('benefit_event_full_resources_v174').install();",1)
   values=list(argv)
   if '--barrier'in values:
    folder=host(values[values.index('--barrier')+1]);folder.resolve().relative_to(p.ROOT.resolve());original(folder/'seven_policy_host_schedule_v45.json',info)
   return cmd
  c.command=route
 else:
  folder=p.ROOT/'seven_policy_entry_receipts_v45';folder.mkdir(exist_ok=True);original(folder/('entry_'+str(time.time_ns())+'.json'),dict(info,pid=os.getpid()))
 _INSTALLED=True

def main():install();__import__('benefit_followup_mixed_entry_v3').main()
if __name__=='__main__':main()

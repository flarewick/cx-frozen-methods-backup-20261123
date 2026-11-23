"""Verified no-save expert visual synchronization over unchanged science83."""
import json,os,time
from pathlib import Path
_INSTALLED=False

def evidence(p):
 files=[p.D/'benefit_expert_visual_sync_v13.py',p.D/'benefit_expert_visual_replay_worker_v15.py',p.HOME/'expert_visual_sync_CPU_proof_v13.json']
 cpu=json.loads(files[-1].read_text())
 assert cpu['passed']and cpu['helper_sha256']==p.sha(files[0])
 for task in ('lift_pot','dump_bin_bigbin'):
  root=p.HOME/('diagnostics/expert_visual_replay_v15_'+task+'_v1')
  result=root/'expert_visual_replay_GPU_result_v15.json';host=root/'expert_visual_replay_host_result_v15.json'
  v=json.loads(result.read_text());h=json.loads(host.read_text())
  assert v['passed']and h['passed']and v['policy_episodes']==0 and v['formal_eligibility_decisions_added']==0
  assert v['helper_sha256']==p.sha(files[0])and v['worker_sha256']==p.sha(files[1])
  assert all(c['task']==task and c['all_physics_steps_covered']and c['original_record_vs_replay_physics_exact']for c in v['cases'])
  files.extend((result,host))
 return {str(f.relative_to(p.R)):p.sha(f)for f in files}

def install():
 global _INSTALLED
 if _INSTALLED:return
 import benefit_followup_capacity_entry_v12 as parent;parent.install()
 import benefit_followup_protocol_v2 as p
 sources=evidence(p)
 import benefit_growth_runtime as runtime
 previous=runtime.install
 def resource_install():
  previous();__import__('benefit_expert_visual_sync_v13').install_mixed_hook()
 runtime.install=resource_install
 info=dict(schema='no_save_expert_visual_sync_v16',entry_sha256=p.sha(__file__),evidence_sha256=sources,
  original83_files_unchanged=True,only_unobserved_expert_visual_tail_deferred=True,all_observations_original_full_sync=True,
  original_planning_dense_control_physics_RNG_eligibility_unchanged=True,full_planner_repeatability_not_claimed=True,
  all_resource_guards_unchanged=True)
 atomic=p.atomic
 def record(path,data):
  if Path(path).name=='deployment_cpu_schedule.json':data=dict(data,expert_visual_sync_schedule_v16=info)
  return atomic(path,data)
 p.atomic=record
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT,host
  before=c.command;old=str(CANONICAL_PROJECT/'deployment/benefit_followup_capacity_entry_v12.py');new=str(CANONICAL_PROJECT/'deployment/benefit_followup_expert_visual_entry_v16.py')
  def route(argv,*a,**kw):
   out=before(argv,*a,**kw)
   for i,x in enumerate(out):
    if x==old:out[i]=new
    elif isinstance(x,str)and "__import__('benefit_followup_capacity_entry_v12').install();"in x:out[i]=x.replace("__import__('benefit_followup_capacity_entry_v12').install();","__import__('benefit_followup_expert_visual_entry_v16').install();",1)
   values=list(argv)
   if '--barrier'in values:
    folder=host(values[values.index('--barrier')+1]);folder.resolve().relative_to(p.ROOT.resolve());atomic(folder/'expert_visual_host_schedule_v16.json',info)
   return out
  c.command=route
 else:
  folder=p.ROOT/'expert_visual_entry_receipts_v16';folder.mkdir(exist_ok=True)
  atomic(folder/('entry_'+str(time.time_ns())+'.json'),dict(info,pid=os.getpid()))
 _INSTALLED=True

def main():
 install();__import__('benefit_followup_capacity_entry_v12').main()
if __name__=='__main__':main()

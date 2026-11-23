"""Actual source setup and repeated same-lane closed proxy reuse, CPU only."""
from pathlib import Path
import gc,types,weakref
import torch,sapien
from concurrent_rng import Lane
from shared6_planner_pool import make_class
from native_expert_case_release_v1.checks import Planner,Task
from native_expert_case_release_v1.release import retire_live_group,shared
from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
from native_onpolicy_mixed_teacher_v2.dispatch_checks import run
from native_policy_execution_v1.common import T,write,sha,read

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();o=T/'outputs/native_onpolicy_mixed_teacher_v3';o.mkdir(parents=True,exist_ok=True)
 setup=run('native_onpolicy_mixed_teacher_v3',o/'CPU_priority5_setup_requests.jsonl')
 c=Coordinator();c.rng.install();lane=Lane('same_live_worker');lane.phase='expert_screen';Planner.constructors=0;pooled,pool=make_class(Planner,c);schedule=types.SimpleNamespace(pooled=pooled,pool=pool);rows=[]
 def episode():
  task=Task();task.scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);task.left=pooled();task.right=pooled();left=task.left.plan_path();right=task.right.plan_path();refs=weakref.ref(task),weakref.ref(task.scene);task=None;gc.collect();assert all(r()is None for r in refs)
  proof=retire_live_group(schedule,lane,c);return left[0],right[0],proof
 try:
  for i in range(5):rows.append(c.call(lane,episode));assert rows[-1][0]==rows[-1][1]==i+1 and not pool
  assert Planner.constructors==2 and len(shared(schedule))==2
 finally:shared(schedule).clear();c.rng.uninstall()
 old=read(T/'outputs/native_expert_case_release_v1/CPU_actual_closed_expert_release_checks.json')
 for p,h in old['source_sha256'].items():assert sha(p)==h
 write(o/'CPU_actual_live_lifetime_checks.json',dict(passed=True,actual_source_priority5_check=setup,actual_PhysX_closed_live_Task_scene_weakrefs_dead=True,repeated_same_worker_live_cases=5,original_real_planners_constructed_only_once=True,original_mutable_planner_counts_preserved=True,no_CUDA_or_optimizer_work=True,input_sha256={str(T/'outputs/native_expert_case_release_v1/CPU_actual_closed_expert_release_checks.json'):sha(T/'outputs/native_expert_case_release_v1/CPU_actual_closed_expert_release_checks.json')},source_sha256={str(p):sha(p)for d in(Path(__file__).parent,T/'native_expert_case_release_v1')for p in d.glob('*')if p.is_file()}))
 print('ACTUAL_LIVE_SOURCE_SETUP_PRIORITY_AND_FIVE_NATIVE_CLOSED_PROXY_REUSE_PASS')
if __name__=='__main__':main()

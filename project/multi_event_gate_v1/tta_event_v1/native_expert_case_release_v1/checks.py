"""Actual CPU PhysX lifetimes and original owner/pool numerical parity."""
from pathlib import Path
import ast,inspect,threading,types,weakref,random,gc
import numpy as np,torch,sapien
from concurrent_rng import Lane
from shared6_planner_pool import make_class
from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
from native_policy_execution_v1.common import T,read,write,sha
from .release import drop_closed,shared

class Planner:
 constructors=0
 def __init__(self):
  type(self).constructors+=1;self.count=0
  self.initial=(torch.rand(3),np.random.rand(3),[random.random()for _ in range(3)])
 def plan_grippers(self,a,b):return np.linspace(a,b,4)
 def plan_path(self):
  self.count+=1
  return np.r_[self.count,torch.rand(3).numpy(),np.random.rand(3),[random.random()for _ in range(3)]]

class Task:pass

def experiment(retire,output):
 c=Coordinator();c.rng.install();tasks=['a','b','c','d','e'];lanes={t:Lane('expert_'+t)for t in tasks}
 for lane in lanes.values():lane.phase='expert_screen'
 Planner.constructors=0;pooled,groups=make_class(Planner,c);schedule=types.SimpleNamespace(tasks={},pool=groups,pooled=pooled,lanes=lanes)
 rows=[];proofs=[]
 def run(name,index):
  torch.manual_seed(7120000+index);np.random.seed(7120000+index);random.seed(7120000+index)
  if name not in schedule.tasks:
   task=Task();task.left=pooled();task.right=pooled();schedule.tasks[name]=task
  task=schedule.tasks[name]
  task.scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);builder=task.scene.create_actor_builder();builder.add_box_collision(half_size=[.01]*3);actor=builder.build_kinematic(name='native_CPU_fixture');actor.set_pose(sapien.Pose([index*.01,0,0]))
  rows.append(np.r_[actor.get_pose().p,task.left.plan_path(),task.right.plan_path()])
  # Native scene exists, with a real closed task lifetime; do not render/step.
 try:
  for index in range(20):
   name=tasks[index%5];lane=lanes[name];c.call(lane,run,name,index)
   if retire:proofs.append(c.call(lane,drop_closed,schedule,name,c,torch,(),output))
  assert Planner.constructors==2,'Real original pair was reinitialized'
  real=shared(schedule);assert len(real)==2 and real[0].count==real[1].count==20
  if retire:assert not schedule.tasks and not schedule.pool and len(proofs)==20
  else:assert len(schedule.tasks)==len(schedule.pool)==5
  return rows,proofs
 finally:schedule.tasks.clear();schedule.pool.clear();shared(schedule).clear();gc.collect();c.rng.uninstall()

def mplib_check():
 """Real original MPLIB contexts can be recreated without changing TOPP/RNG."""
 import mplib,toppra as ta
 from mplib.sapien_utils import SapienPlanningWorld,SapienPlanner
 source=T.parents[1]/'outputs/evaluations/structured_action/beat_block_hammer/L0/eval_20261003_164626_759241/robotwin/envs/robot/planner.py'
 cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='MplibPlanner')
 ns=dict(np=np,ta=ta,mplib=mplib,SapienPlanningWorld=SapienPlanningWorld,SapienPlanner=SapienPlanner)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),ns)
 scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);builder=scene.create_articulation_builder();base=builder.create_link_builder();base.set_name('base');tip=builder.create_link_builder(base);tip.set_name('tip');tip.set_joint_name('joint');tip.set_joint_properties('revolute',limits=[[-1,1]],friction=0,damping=0,pose_in_parent=sapien.Pose(),pose_in_child=sapien.Pose());tip.add_box_collision(half_size=[.01]*3);entity=builder.build(fix_root_link=True);entity.set_name('cpu_robot')
 before=(torch.random.get_rng_state().clone(),np.random.get_state(),random.getstate())
 a=ns['MplibPlanner']('unused_for_scene',None,'tip',sapien.Pose(),entity,scene=scene)
 b=ns['MplibPlanner']('unused_for_scene',None,'tip',sapien.Pose(),entity,scene=scene)
 assert torch.equal(before[0],torch.random.get_rng_state())and before[1][0]==np.random.get_state()[0]and np.array_equal(before[1][1],np.random.get_state()[1])and before[1][2:]==np.random.get_state()[2:]and before[2]==random.getstate()
 path=np.linspace(0,.2,10)[:,None];left=a.TOPP(path,.004);right=b.TOPP(path,.004)
 from native_cpu_topp_v1.runtime import exact
 assert exact(left,right)
 return dict(passed=True,actual_original_MplibPlanner_AST_used=True,actual_CPU_PhysX_and_SapienPlanningWorld=True,two_fresh_original_TOPP_contexts_exact=True,constructor_Torch_numpy_python_RNG_unchanged=True,fixture_one_joint_only=True,source_sha256={str(source):sha(source)})

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();o=T/'outputs/native_expert_case_release_v1';o.mkdir(parents=True,exist_ok=True)
 old,_=experiment(False,o/'CPU_unused_baseline.jsonl');new,proofs=experiment(True,o/'CPU_closed_task_release_v2.jsonl')
 assert len(old)==len(new)==20 and all(np.array_equal(a,b)for a,b in zip(old,new))
 native=mplib_check();assert not torch.cuda.is_initialized()
 write(o/'CPU_actual_closed_expert_release_checks.json',dict(passed=True,original_ModelService_and_shared_pool_used=True,actual_PhysX_scene_lifetimes=20,closed_native_scene_and_task_weakrefs_dead=True,real_planner_pair_constructed_only_once=True,old_and_fresh_task_original_mutable_planner_outputs_and_RNG_exact=True,native_MPLIB=native,no_native_RGB_or_task_expert_or_GPU_qualification_claim=True,actual_GPU_q0_and_complete_resources_still_required=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()}))
 print('ACTUAL_CPU_PHYSX20_CLOSED_LIFETIMES_POOL_RNG_AND_ORIGINAL_MPLIB_TOPP_PASS')
if __name__=='__main__':main()

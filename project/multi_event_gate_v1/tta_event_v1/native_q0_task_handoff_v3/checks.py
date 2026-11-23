"""Real CPU PhysX dynamics and original pool/router; no GPU Robot/RGB claim."""
from pathlib import Path
import ast,random,threading,types,weakref,gc,os
import mplib,toppra as ta
from mplib.sapien_utils import SapienPlanner,SapienPlanningWorld
import numpy as np
import torch,sapien
from concurrent_rng import Lane
from shared6_planner_pool import make_class
from cx_tta_feedback_confirm5.lanes import gripper_only
from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
from native_policy_execution_v1.common import T,write,sha
from .handoff import capture,detach,take

class Planner:
 constructors=0
 def __init__(self):type(self).constructors+=1;self.count=0
 def plan_grippers(self,a,b):return np.linspace(a,b,4)
 def plan_path(self):self.count+=1;return self.count
class Task:pass

def old_world_planners():
 source=T.parents[1]/'outputs/evaluations/structured_action/beat_block_hammer/L0/eval_20261003_164626_759241/robotwin/envs/robot/planner.py'
 cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='MplibPlanner');namespace=dict(np=np,ta=ta,mplib=mplib,SapienPlanningWorld=SapienPlanningWorld,SapienPlanner=SapienPlanner)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),namespace)
 scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);builder=scene.create_articulation_builder();base=builder.create_link_builder();base.set_name('base');tip=builder.create_link_builder(base);tip.set_name('tip');tip.set_joint_name('joint');tip.set_joint_properties('revolute',limits=[[-1,1]],friction=0,damping=0,pose_in_parent=sapien.Pose(),pose_in_child=sapien.Pose());tip.add_box_collision(half_size=[.01]*3);entity=builder.build(fix_root_link=True);entity.set_name('old_CPU_robot')
 return [namespace['MplibPlanner']('unused',None,'tip',sapien.Pose(),entity,scene=scene)for _ in range(2)],weakref.ref(scene)

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 affinity=os.sched_getaffinity(0);os.sched_setaffinity(0,sorted(affinity)[:2])
 from native_cpu_topp_v1.runtime import prepare,close
 actual_pool=prepare(T/'outputs/native_q0_task_handoff_v3/actual_wrapper',write)
 c=Coordinator();c.rng.install();expert=Lane('expert_fixture');expert.phase='expert_screen';full=Lane('actual_full');reference=Lane('fresh_reference')
 pooled,groups=make_class(Planner,c);replay=gripper_only(Planner)
 out=T/'outputs/native_q0_task_handoff_v3';out.mkdir(parents=True,exist_ok=True)
 schedule=types.SimpleNamespace(c=c,tasks={},pool=groups,pooled=pooled,lanes={'fixture':expert},replay=replay,root=out)
 model=torch.nn.Linear(3,2);completed=0
 def seed(n):torch.manual_seed(n);np.random.seed(n);random.seed(n)
 def setup(planners):
  task=Task();task.take_action_cnt=0;task.robot=types.SimpleNamespace(left_planner=planners[0](),right_planner=planners[1]())
  worlds,task.old_scene_ref=old_world_planners();task.robot.left_mplib_planner,task.robot.right_mplib_planner=worlds;worlds=None
  task.scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);task.scene.set_timestep(.004)
  b=task.scene.create_actor_builder();b.add_box_collision(half_size=[.01,.01,.01]);task.actor=b.build(name='CPU_dynamic_fixture');task.actor.set_pose(sapien.Pose([0.,0.,1.]))
  return task
 def observe(task):return dict(native_action_count=task.take_action_cnt,rgb=(np.zeros((2,2,3),np.uint8),),command_state=np.zeros(14,np.float32))
 def dynamics(task):
  values=[]
  for _ in range(30):
   task.scene.step();task.take_action_cnt+=1
   values.append(np.r_[task.actor.get_pose().p,torch.rand(2).numpy(),np.random.rand(2),random.random(),task.robot.left_planner.plan_grippers(0,1)])
  return np.stack(values)
 try:
  for index in range(10):
   case='fixture:'+str(index)
   def prepare():
    task=setup((pooled,pooled));seed(7120000+index);initial=observe(task);capture(schedule,task,initial,case);schedule.tasks['fixture']=task
    # Official language draws happen after captured q0 lane state.
    np.random.rand(17);random.random();torch.rand(3)
    return id(task.scene)
   scene_id=c.call(expert,prepare)
   c.call(expert,detach,schedule,'fixture',dict(case_id=case),torch,(model,))
   def consume():
    task,initial=take(schedule,case,full);assert id(task.scene)==scene_id and initial['native_action_count']==0 and task.old_scene_ref()is None
    assert np.array_equal(task.robot.right_planner.plan_grippers(0,1),np.linspace(0,1,4))
    return task
   task=c.call(full,consume);actual=c.call(full,dynamics,task)
   def fresh():
    other=setup((replay,replay));seed(7120000+index);return dynamics(other)
   expected=c.call(reference,fresh);assert np.array_equal(actual,expected)
   taskref,sceneref=weakref.ref(task),weakref.ref(task.scene);task=None;gc.collect();assert taskref()is None and sceneref()is None
   assert not schedule.q0_pending and not schedule.tasks and not schedule.pool;completed+=1
  assert Planner.constructors==2
 finally:c.rng.uninstall()
 close(actual_pool,T/'outputs/native_q0_task_handoff_v3/actual_wrapper',write);os.sched_setaffinity(0,affinity)
 assert not torch.cuda.is_initialized()
 write(out/'CPU_actual_checks.json',dict(passed=True,actual_CPU_PhysX_transfers=completed,actual_original_ModelService_RngRouter_and_planner_pool=True,actual_native_dynamics_and_Torch_numpy_python_outputs_exact=True,pre_language_RNG_restored_in_new_lane=True,actual_obsolete_MPLIB_scene_released_in_combined_handoff=True,actual_existing_CPU_TOPP_wrapper_and_pool=True,actual_CPU_pool_workers_exit0=True,expert_proxy_weakrefs_dead_and_actual_task_scene_released=True,original_real_pair_constructed_only_once=True,no_native_GPU_Robot_RGB_or_TTA_utility_qualification=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()}))
 print('ACTUAL_CPU_PHYSX_Q0_HANDOFF10_DYNAMICS_POOL_RNG_EXACT_PASS',flush=True)
if __name__=='__main__':main()

"""Actual CPU SapienPlanningWorld retention and exact original TOPP arrays/RNG."""
from pathlib import Path
import ast,gc,random,types,weakref,os
import numpy as np,torch,sapien,mplib,toppra as ta
from mplib.sapien_utils import SapienPlanner,SapienPlanningWorld
from native_policy_execution_v1.common import T,write,sha
from native_cpu_topp_v1.runtime import exact
from .solver import detach

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 affinity=os.sched_getaffinity(0);os.sched_setaffinity(0,sorted(affinity)[:2])
 from native_cpu_topp_v1.runtime import prepare,close
 out=T/'outputs/native_scene_free_topp_v2';out.mkdir(parents=True,exist_ok=True)
 pool=prepare(out/'actual_wrapper',write)
 assert getattr(mplib.Planner.TOPP,'_offloaded_TOPP_v52',False)
 source=T.parents[1]/'outputs/evaluations/structured_action/beat_block_hammer/L0/eval_20261003_164626_759241/robotwin/envs/robot/planner.py'
 cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='MplibPlanner');namespace=dict(np=np,ta=ta,mplib=mplib,SapienPlanningWorld=SapienPlanningWorld,SapienPlanner=SapienPlanner)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),namespace)
 def old_scene_robot():
  scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);builder=scene.create_articulation_builder();base=builder.create_link_builder();base.set_name('base');tip=builder.create_link_builder(base);tip.set_name('tip');tip.set_joint_name('joint');tip.set_joint_properties('revolute',limits=[[-1,1]],friction=0,damping=0,pose_in_parent=sapien.Pose(),pose_in_child=sapien.Pose());tip.add_box_collision(half_size=[.01]*3);entity=builder.build(fix_root_link=True);entity.set_name('cpu_old_robot')
  robot=types.SimpleNamespace()
  for name in('left_mplib_planner','right_mplib_planner'):setattr(robot,name,namespace['MplibPlanner']('unused_in_scene',None,'tip',sapien.Pose(),entity,scene=scene))
  return robot,weakref.ref(scene)
 class Task:pass
 rows=[]
 for cycle in range(5):
  robot,oldref=old_scene_robot();gc.collect();assert oldref()is not None,'Original planner should reproduce stale Scene retention'
  task=Task();task.robot=robot;task.scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);current=weakref.ref(task.scene)
  paths=[np.linspace(0,.2+.01*k,10+k)[:,None]for k in range(4)]
  before=(torch.get_rng_state().clone(),np.random.get_state(),random.getstate());expected=[robot.left_mplib_planner.TOPP(p,.004)for p in paths];method=robot.left_mplib_planner.TOPP.__func__
  proof=detach(task);assert oldref()is None and current()is task.scene
  actual=[robot.left_mplib_planner.TOPP(p,.004)for p in paths];assert all(exact(a,b)for a,b in zip(actual,expected));assert robot.left_mplib_planner.TOPP.__func__ is method
  assert torch.equal(before[0],torch.get_rng_state())and np.array_equal(before[1][1],np.random.get_state()[1])and before[1][2:]==np.random.get_state()[2:]and before[2]==random.getstate()
  for obj in(robot.left_mplib_planner,robot.right_mplib_planner):assert not hasattr(obj,'planner')and not hasattr(obj,'planning_world')
  robot=None;task=None;gc.collect();assert current()is None;rows.append(proof)
 close(pool,out/'actual_wrapper',write);os.sched_setaffinity(0,affinity)
 assert not torch.cuda.is_initialized()
 write(T/'outputs/native_scene_free_topp_v2/CPU_actual_MPLIB_scene_release_checks.json',dict(passed=True,actual_CPU_PhysX_old_scene_retention_and_release_cycles=5,actual_original_MplibPlanner_and_SapienPlanningWorld=True,original_TOPP_function_object_preserved=True,actual_existing_four_worker_TOPP_wrapper_preserved=True,actual_CPU_pool_workers_exit0=True,exact_original_TOPP_results=20,original_Torch_numpy_python_RNG_preserved=True,all_obsolete_scene_weakrefs_dead=True,current_task_scene_released_after_case=True,no_GPU_Robot_or_TTA_utility_claim=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()},input_sha256={str(source):sha(source)}))
 print('ACTUAL_CPU_MPLIB_OLD_SCENE_RELEASE5_ORIGINAL_TOPP20_RNG_EXACT_PASS',flush=True)
if __name__=='__main__':main()

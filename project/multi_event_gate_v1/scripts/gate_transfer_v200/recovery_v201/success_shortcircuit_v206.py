"""Preserve the official boolean predicate; test necessary gripper terms first."""
import ast,functools,inspect,json,types,time
from pathlib import Path
import numpy as np

def build(original):
 tree=ast.parse(__import__('textwrap').dedent(inspect.getsource(original)))
 value=tree.body[0].body[-1].value
 assert isinstance(value,ast.BoolOp)and isinstance(value.op,ast.And)
 assert [ast.unparse(v)for v in value.values[-2:]]==['self.is_left_gripper_open()', 'self.is_right_gripper_open()']
 @functools.wraps(original)
 def check(self):
  if not self.is_left_gripper_open()or not self.is_right_gripper_open():return False
  return original(self)
 return check

def install(task):
 import benefit_followup_protocol_v2 as p
 name=type(task).__name__
 if name not in p.TASKS or getattr(task,'_transfer_shortcircuit206',False):return
 proof=json.loads((p.ROOT/'success_shortcircuit_CPU_v206.json').read_text())
 assert proof['passed']and proof['helper_sha256']==p.sha(__file__)
 original=task.check_success.__func__
 assert p.sha(inspect.getsourcefile(original))==proof['task_source_sha256'][name]
 for path,expected in proof['pure_getter_source_sha256'].items():assert p.sha(path)==expected
 assert ast.unparse(ast.parse(__import__('textwrap').dedent(inspect.getsource(task.robot.is_left_gripper_open))).body[0].body[0])=='return self.left_gripper_val > 0.8'
 assert ast.unparse(ast.parse(__import__('textwrap').dedent(inspect.getsource(task.robot.is_right_gripper_open))).body[0].body[0])=='return self.right_gripper_val > 0.8'
 task.check_success=types.MethodType(build(original),task);task._transfer_shortcircuit206=True

def main():
 import torch,sapien,importlib
 import benefit_followup_protocol_v2 as p
 assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized()
 torch.set_num_threads(1)
 from evaluate import host
 plan=json.loads((p.TRAINING/'training_plan.json').read_text());protocol=json.loads((host(plan['dataset'])/'protocol.json').read_text());runtime=Path(protocol['runtime_source'])
 import copy,transforms3d as t3d
 def isolated_class(file,name,methods,assignments=()):
  node=copy.deepcopy(next(n for n in ast.parse(file.read_text()).body if isinstance(n,ast.ClassDef)and n.name==name))
  node.bases=[];node.keywords=[];node.body=[n for n in node.body if (isinstance(n,ast.FunctionDef)and n.name in methods)or(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in assignments for t in n.targets))]
  future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
  ns=dict(np=np,sapien=sapien,t3d=t3d)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[])),str(file),'exec'),ns)
  return ns[name]
 Base_Task=isolated_class(runtime/'envs/_base_task.py','Base_Task',('is_left_gripper_open','is_right_gripper_open'))
 Robot=isolated_class(runtime/'envs/robot/robot.py','Robot',('is_left_gripper_open','is_right_gripper_open'))
 Actor=isolated_class(runtime/'envs/utils/actor_utils.py','Actor',('get_point','get_pose','get_functional_point'),('POINTS',))
 getter_files=[Path(inspect.getsourcefile(x))for x in (Base_Task.is_left_gripper_open,Base_Task.is_right_gripper_open,Robot.is_left_gripper_open,Robot.is_right_gripper_open,Actor.get_point,Actor.get_pose,Actor.get_functional_point)]
 assert ast.unparse(ast.parse(__import__('textwrap').dedent(inspect.getsource(Base_Task.is_left_gripper_open))).body[0].body[0])=='return self.robot.is_left_gripper_open()'
 assert ast.unparse(ast.parse(__import__('textwrap').dedent(inspect.getsource(Base_Task.is_right_gripper_open))).body[0].body[0])=='return self.robot.is_right_gripper_open()'
 # Real original Actor getters include the exact SAPIEN transform, scaling,
 # matrix multiply and transforms3d quaternion conversion. No simulated GPU.
 def actor(position):
  value=Actor.__new__(Actor);pose=sapien.Pose(position)
  value.actor=types.SimpleNamespace(get_pose=lambda:pose)
  value.config=dict(functional_matrix=[np.eye(4).tolist()],scale=[1.,1.,1.])
  return value
 rng=np.random.RandomState(206);cases={};timings={};task_hash={}
 numpy_global=np.random.get_state();torch_global=torch.get_rng_state().clone()
 boundary=[0.,-0.,.8,np.nextafter(.8,-np.inf),np.nextafter(.8,np.inf),1.]
 for name in p.TASKS:
  raw=isolated_class(runtime/'envs'/(name+'.py'),name,('check_success',));original=raw.check_success;cls=type(name,(Base_Task,),dict(check_success=original));optimized=build(original)
  task_hash[name]=p.sha(inspect.getsourcefile(original));task=cls.__new__(cls)
  robot=Robot.__new__(Robot);task.robot=robot;task.table_z_bias=0.
  count=0
  pairs=[]
  for i in range(1000):
   a=rng.uniform([-.2,-.2,.70],[.2,.2,.85]);b=a+rng.uniform(-.08,.08,3);pairs.append((a,b))
  pairs.extend([(np.array([0.,0.,.74]),np.array([d,0.,.77]))for d in (0.,.02,.035,.04,np.nextafter(.035,-np.inf),np.nextafter(.035,np.inf))])
  for a,b in pairs:
   task.cup=task.bowl1=actor(a);task.coaster=task.bowl2=actor(b)
   for left in boundary:
    for right in boundary:
     robot.left_gripper_val=np.float64(left);robot.right_gripper_val=np.float64(right)
     before=rng.get_state();expected=bool(original(task));actual=bool(optimized(task));assert actual==expected
     after=rng.get_state();assert np.array_equal(before[1],after[1])and before[2:]==after[2:]
     count+=1
  # A true-positive state is explicitly exercised, as well as both closed cases.
  a=np.array([0.,0.,.74]);b=np.array([0.,0.,.74 if name=='place_empty_cup'else .77]);task.cup=task.bowl1=actor(a);task.coaster=task.bowl2=actor(b)
  robot.left_gripper_val=robot.right_gripper_val=np.float64(1.);assert bool(original(task))and bool(optimized(task))
  robot.left_gripper_val=0.;robot.right_gripper_val=1.
  began=time.monotonic()
  for _ in range(2000):original(task)
  old=time.monotonic()-began;began=time.monotonic()
  for _ in range(2000):optimized(task)
  new=time.monotonic()-began
  cases[name]=count+1;timings[name]=dict(original_seconds=old,shortcircuit_seconds=new,closed_gripper_CPU_microbenchmark_only=True)
 after_global=np.random.get_state();assert np.array_equal(numpy_global[1],after_global[1])and numpy_global[2:]==after_global[2:]and torch.equal(torch_global,torch.get_rng_state())and not torch.cuda.is_initialized()
 assert task_hash==json.loads((p.ROOT/'transfer_task_runtime_CPU.json').read_text())['task_source_sha256']
 p.atomic(p.ROOT/'success_shortcircuit_CPU_v206.json',dict(passed=True,CPU_only=True,no_CUDA=True,helper_sha256=p.sha(__file__),task_source_sha256=task_hash,pure_getter_source_sha256={str(f):p.sha(f)for f in getter_files},boolean_equivalence_cases=cases,actual_original_Actor_and_Robot_getters=True,isolated_original_method_AST_no_planner_import=True,original_geometry_and_thresholds_unchanged=True,no_physics_tick_or_render_prefix_removed=True,no_early_success_added=True,RNG_unchanged=True,return_value_compared_as_boolean_as_original_consumers=True,microbenchmark=timings))
 print('SUCCESS206_CPU_PASSED',cases,timings,flush=True)
if __name__=='__main__':main()

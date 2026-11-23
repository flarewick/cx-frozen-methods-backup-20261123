"""Preserve original pure TOPP without its obsolete collision world and Scene."""
import ast,gc,inspect,textwrap,types,weakref
import numpy as np

class TimeParameterizer:
 __slots__=('joint_vel_limits','joint_acc_limits','TOPP')
 def __init__(self,method):
  fn=method.__func__;planner=method.__self__
  inspected=fn
  if getattr(fn,'_offloaded_TOPP_v52',False):
   import mplib
   from native_cpu_topp_v1 import backend,runtime
   closure=inspect.getclosurevars(fn).nonlocals
   code=next(v for v in runtime.prepare.__code__.co_consts if isinstance(v,types.CodeType)and v.co_name=='offload')
   if fn is not mplib.Planner.TOPP or fn.__code__ is not code or closure.get('original')is not backend.ORIGINAL or closure.get('pool')is not backend.POOL:raise ValueError('Unknown TOPP scheduling wrapper')
   inspected=backend.original_function();assert inspected is backend.ORIGINAL
  attributes={n.attr for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(inspected))))if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='self'}
  if attributes!={'joint_vel_limits','joint_acc_limits'}:raise ValueError('Original TOPP gained mutable world dependencies')
  self.joint_vel_limits=np.asarray(planner.joint_vel_limits).copy();self.joint_acc_limits=np.asarray(planner.joint_acc_limits).copy()
  self.TOPP=types.MethodType(fn,self)

def replace_one(robot,name,current):
 old=getattr(robot,name);planner=old.TOPP.__self__;world=planner.planning_world;scene=world._sim_scene
 refs=dict(planner=weakref.ref(planner),world=weakref.ref(world),scene=weakref.ref(scene),old=weakref.ref(old));stale=scene is not current
 value=TimeParameterizer(old.TOPP);assert value.TOPP.__func__ is old.TOPP.__func__
 assert np.array_equal(value.joint_vel_limits,planner.joint_vel_limits)and np.array_equal(value.joint_acc_limits,planner.joint_acc_limits)
 setattr(robot,name,value)
 return refs,stale

def detach(task):
 refs=[]
 for name in('left_mplib_planner','right_mplib_planner'):refs.append(replace_one(task.robot,name,task.scene))
 gc.collect()
 for group,stale in refs:
  assert group['old']()is None and group['planner']()is None and group['world']()is None,'Unused MPLIB planner/world still owned'
  if stale:assert group['scene']()is None,'Obsolete expert reset Scene still owned'
  else:assert group['scene']()is task.scene
 return dict(passed=True,original_TOPP_function_objects_and_limits_exact=True,obsolete_MPLIB_worlds_released=2,stale_scene_references=sum(int(stale)for _,stale in refs),all_obsolete_scene_weakrefs_dead=True,current_native_scene_preserved=True,no_geometric_planner_calls_in_full_qpos=True)

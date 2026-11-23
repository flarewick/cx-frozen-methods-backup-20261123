"""Actual original ModelService/Torch RNG equality under unrelated policy work."""
from pathlib import Path
import threading,time,random,inspect,ast
import numpy as np
import torch
from concurrent_rng import RngRouter,Lane
from model_service import ModelService
from shared6_planner_pool import make_class
from scene_gate import SceneGate
from native_policy_execution_v1.common import T,write,sha
from .dispatch import route

class Planner:
 def __init__(self):self.count=0
 def plan_grippers(self,now,target):return np.linspace(now,target,4)
 def plan_path(self,value):
  self.count+=1
  return np.asarray([float(torch.rand(())),np.random.rand(),random.random(),self.count,float(value)])
 def plan_batch(self,value):return self.plan_path(value)
 def update_point_cloud(self,value):return self.plan_path(float(np.sum(value)))

class Coordinator:
 def __init__(self):
  self.rng=RngRouter(False);self.scene_gate=SceneGate();self.service=ModelService(self);self.cancel=threading.Event()
 def check_cancel(self):assert not self.cancel.is_set()
 def note(self,*args,**kwargs):pass
 def call(self,*args,**kwargs):return self.service.call(*args,**kwargs)

def experiment(dispatched):
 c=Coordinator();c.rng.install();lane=Lane('expert');lane.phase='expert_screen';rows=[];errors=[];done=threading.Event()
 try:
  pooled,groups=make_class(Planner,c);init=pooled.__init__
  if dispatched:pooled=route(pooled,c)
  assert pooled.__init__ is init and 'shared'in inspect.getclosurevars(init).nonlocals
  def construct():
   torch.manual_seed(501);np.random.seed(501);random.seed(501);return pooled(),pooled()
  left,right=c.call(lane,construct)
  def body():
   for i in range(20):
    rows.append(left.plan_path(i));rows.append(right.plan_batch(i+1));rows.append(left.update_point_cloud(np.asarray([i,i+2])))
    rows.append(np.asarray([np.random.rand(),random.random()]))
    assert np.array_equal(left.plan_grippers(0.,1.),np.linspace(0.,1.,4))
   if dispatched:
    try:getattr(left,'count')
    except RuntimeError as e:assert 'cannot escape owner'in str(e)
    else:raise AssertionError('Mutable attribute escaped owner')
  if not dispatched:c.call(lane,body)
  else:
   def worker():
    try:
     with c.rng.bind(lane):body()
    except BaseException as e:errors.append(repr(e))
    finally:done.set()
   full=Lane('unrelated_full');full_done=threading.Event()
   def policy():
    try:
     for _ in range(40):c.call(full,lambda:torch.rand(19))
    except BaseException as e:errors.append(repr(e))
    finally:full_done.set()
   threads=[threading.Thread(target=worker),threading.Thread(target=policy)]
   for t in threads:t.start()
   deadline=time.monotonic()+15
   while not(done.is_set()and full_done.is_set()):
    assert time.monotonic()<deadline;c.service.process_one(.01)
   for t in threads:t.join()
   assert not errors and c.service.requests.empty()
  return rows,lane.numpy.get_state(),lane.python.getstate(),lane.torch_cpu.clone(),c.service.executed
 finally:c.rng.uninstall()

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 old,a,b,c,n=experiment(False);new,x,y,z,m=experiment(True)
 assert len(old)==len(new)==80 and all(np.array_equal(p,q)for p,q in zip(old,new))
 assert a[0]==x[0]and np.array_equal(a[1],x[1])and a[2:]==x[2:]and b==y and torch.equal(c,z)
 runtime=T.parents[1]/'outputs/evaluations/structured_action/beat_block_hammer/L0/eval_20261003_164626_759241/robotwin/envs'
 # The original expert CPU bodies cannot manipulate Torch; setup is owner-routed.
 source=(runtime/'_base_task.py').read_text();tree=ast.parse(source)
 torch_functions=[]
 for cls in tree.body:
  if isinstance(cls,ast.ClassDef):
   for f in cls.body:
    if isinstance(f,ast.FunctionDef)and any(isinstance(node,ast.Attribute)and isinstance(node.value,ast.Name)and node.value.id=='torch'for node in ast.walk(f)):torch_functions.append(f.name)
 assert torch_functions==['_init_task_env_'],torch_functions
 for name in('beat_block_hammer','turn_switch','adjust_bottle','grab_roller','open_laptop'):
  tree=ast.parse((runtime/(name+'.py')).read_text())
  assert not any(isinstance(node,ast.Attribute)and isinstance(node.value,ast.Name)and node.value.id=='torch'for node in ast.walk(tree))
 sources={str(p):sha(p)for p in Path(__file__).resolve().parent.glob('*')if p.is_file()}
 write(T/'outputs/native_expert_owner_dispatch_v1/CPU_actual_dispatch_checks.json',dict(passed=True,original_ModelService_and_RngRouter_used=True,actual_Torch_CPU=True,CUDA_initialized=torch.cuda.is_initialized(),original_vs_dispatched_rows=80,original_vs_dispatched_numpy_python_Torch_lane_state_exact=True,unrelated_full_owner_requests=40,mutable_attribute_escape_rejected=True,original_pooled_constructor_preserved=True,only_setup_direct_Torch_in_base_task=True,expert_task_direct_Torch_calls=0,source_sha256=sources,no_native_recovery_or_training_qualification=True))
 print('ACTUAL_MODEL_SERVICE_AND_TORCH_DISPATCH_80_ROWS_FINAL_RNG_EXACT_NO_CUDA')
if __name__=='__main__':main()

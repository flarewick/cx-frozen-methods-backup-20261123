"""Execute the exact extracted production callback through real owner/RNG dispatch."""
import ast,copy,contextlib,threading,time,random
import numpy as np
import torch
from concurrent_rng import Lane
from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
from .common import T,C

def extract(path):
 tree=ast.parse(path.read_text());nodes=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='tracker']
 assert len(nodes)==1
 return copy.deepcopy(nodes[0])
def experiment(path):
 c=Coordinator();c.rng.install();owner=threading.get_ident();events=[];rows=[];errors=[];done=threading.Event();lane=Lane('actual_tracker_fixture');old_precision=torch.get_float32_matmul_precision()
 def check_allocator():
  assert threading.get_ident()==owner,'Allocator escaped owner';events.append('check')
 @contextlib.contextmanager
 def sam2_profile():
  assert threading.get_ident()==owner,'SAM2 profile escaped owner'
  saved=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest');events.append('enter')
  try:yield
  finally:torch.set_float32_matmul_precision(saved);events.append('exit')
 def forward(index,*,scale):
  assert threading.get_ident()==owner,'SAM2 forward escaped owner'
  assert torch.get_float32_matmul_precision()=='highest'
  events.append('forward');return np.r_[torch.rand(5).numpy(),np.random.rand(5),[random.random()for _ in range(5)],index*scale]
 env=dict(c=c,lane=lane,check_allocator=check_allocator,sam2_profile=sam2_profile)
 module=ast.fix_missing_locations(ast.Module(body=[extract(path)],type_ignores=[]));exec(compile(module,str(path),'exec'),env);tracker=env['tracker']
 try:
  def seed():torch.manual_seed(417);np.random.seed(417);random.seed(417)
  c.call(lane,seed)
  def worker():
   try:
    with c.rng.bind(lane):
     for i in range(12):rows.append(tracker(forward,i,scale=3))
   except BaseException as exc:errors.append(repr(exc))
   finally:done.set()
  thread=threading.Thread(target=worker);thread.start();deadline=time.monotonic()+10
  while not done.is_set():
   assert time.monotonic()<deadline;c.service.process_one(.01)
  thread.join(1);assert not thread.is_alive();assert not errors,errors
  final=(lane.numpy.get_state(),lane.python.getstate(),lane.torch_cpu.clone())
  assert events==['check','enter','forward','exit']*12 and torch.get_float32_matmul_precision()==old_precision
  c.call(lane,seed)
  expected=[]
  for i in range(12):
   def reference(i=i):
    with sam2_profile():return forward(i,scale=3)
   expected.append(c.call(lane,reference))
  assert all(np.array_equal(a,b)for a,b in zip(rows,expected))
  a,b,d=final;x=lane.numpy.get_state();assert a[0]==x[0]and np.array_equal(a[1],x[1])and a[2:]==x[2:]and b==lane.python.getstate()and torch.equal(d,lane.torch_cpu)
  return True
 finally:torch.set_float32_matmul_precision(old_precision);c.rng.uninstall()
def run():
 assert not torch.cuda.is_initialized();assert experiment(C/'runtime.py')
 try:experiment(T/'native_onpolicy_collection_v22/runtime.py')
 except AssertionError as exc:assert 'escaped owner'in str(exc)
 else:raise AssertionError('Frozen C22 broken production callback unexpectedly accepted')
 return dict(passed=True,actual_extracted_production_tracker_executed=True,real_ModelService_and_RngRouter=True,worker_to_owner_dispatch_exact=True,CPU_Torch_numpy_python_rows_and_final_lane_RNG_exact=True,original_profile_precision_restored=True,callback_arguments_and_results_exact=True,original_C22_owner_escape_rejected=True,no_CUDA_or_native_or_training_work=True)

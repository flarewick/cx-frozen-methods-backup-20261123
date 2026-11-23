"""Actual original owner queue/RNG, nested requests and failures remain unchanged."""
from pathlib import Path
import threading,random,pickle,time
import numpy as np,torch
from concurrent_rng import Lane
from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
from native_owner_profile_v2.profile import OwnerProfile
from native_policy_execution_v1.common import T,write,sha
O=T/'outputs/native_owner_profile_v2/CPU_v1'
def trial(profiled):
 torch.manual_seed(71);np.random.seed(71);random.seed(71)
 c=Coordinator();c.rng.install();p=OwnerProfile(c.service)if profiled else None
 a,b=Lane('A'),Lane('B');result=[];events=[];threads=[]
 def draw():return np.r_[torch.rand(3).numpy(),np.random.rand(3),random.random()]
 def nested():
  if p:
   current=p.snapshot();assert not current['passed']and current['bookkeeping_only']and current['open_call_depth']==1
  first=draw();other=c.call(a,draw);last=draw();return first,other,last
 def error():raise ValueError('original retained failure')
 def worker():
  try:
   for _ in range(32):result.append(c.call(a,nested))
   try:c.call(a,error)
   except ValueError as e:events.append((type(e).__name__,str(e)))
  except BaseException as ex:events.append(('unexpected',str(ex)))
 try:
  for lane in[a,b]:c.call(lane,lambda:torch.manual_seed(17))
  t=threading.Thread(target=worker);threads.append(t);t.start();deadline=time.monotonic()+10
  while t.is_alive():assert time.monotonic()<deadline;c.service.process_one(.01)
  t.join();assert c.service.requests.empty()
  assert events==[('ValueError','original retained failure')],events
  states=pickle.dumps([(l.name,l.numpy.get_state(),l.python.getstate(),l.torch_cpu.numpy())for l in[a,b]])
  receipt=p.close()if p else None
  if receipt:
   assert sum(r['calls']for r in receipt['rows'])==c.service.executed==67
   assert sum(r['exceptions']for r in receipt['rows'])==1
   assert all(0<=r['exclusive_host_ns']<=r['inclusive_host_ns']for r in receipt['rows'])
   assert c.service.execute==p.original
  assert not torch.cuda.is_initialized()
  return result,states,events,receipt
 finally:
  c.cancel.set()
  for t in threads:t.join(1);assert not t.is_alive()
  if p and not p.closed:p.close()
  c.rng.uninstall()
def main():
 torch.set_num_threads(1);assert not O.exists();O.mkdir(parents=True)
 a=trial(False);b=trial(True)
 assert a[1:3]==b[1:3]and len(a[0])==len(b[0])==32
 assert all(np.array_equal(x,y)for aa,bb in zip(a[0],b[0],strict=True)for x,y in zip(aa,bb,strict=True))
 write(O/'CPU_contract_checks_v1.json',dict(passed=True,CPU_only=True,actual_original_ModelService_and_RngRouter=True,all32_nested_requests_arrays_and_final_RNG_exact=True,original_queued_callable_identity_not_replaced=True,original_error_propagated=True,all67_original_execute_calls_measured=True,nested_inclusive_and_exclusive_separated=True,partial_snapshot_is_not_qualification_and_preserves_open_stack=True,observer_restored_original_method=True,not_GPU_integrated_or_qualified=True,receipt=b[3],source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()}))
 print('OWNER_PROFILE_ORIGINAL_QUEUE_NESTED_RNG_EXCEPTIONS_CPU_PASS',flush=True)
if __name__=='__main__':main()

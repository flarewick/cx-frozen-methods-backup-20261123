"""CPU-only ownership, argument/result, RNG and cache-pressure negative checks."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import torch,tempfile,threading,json,weakref
from .completed_planners import release_completed
from .forward_cache import install
from .common import O,write

def main():
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 class Object:pass
 def pooled():
  shared=[Object(),Object()]
  class Proxy:
   def __init__(self):self.real=shared[0]
  return Proxy,shared
 Proxy,shared=pooled();refs=[weakref.ref(o)for o in shared]
 schedule=SimpleNamespace(pooled=Proxy,tasks={'task':Object()},pool={'a':[Proxy()]},state={'a':dict(rows=[{},{}])})
 class CUDA:
  def __init__(self):self.reserved=100*2**20;self.allocated=20*2**20;self.free=100*2**20;self.calls=0
  def mem_get_info(self):return self.free,2**30
  def memory_allocated(self):return self.allocated
  def memory_reserved(self):return self.reserved
  def synchronize(self):pass
  def get_rng_state(self):return torch.tensor([9,8,7],dtype=torch.uint8)
  def empty_cache(self):self.free+=self.reserved-self.allocated;self.reserved=self.allocated;self.calls+=1
 cuda=CUDA();fake=SimpleNamespace(cuda=cuda,random=torch.random,equal=torch.equal)
 c=SimpleNamespace(service=SimpleNamespace(owner=threading.get_ident()),scene_gate=SimpleNamespace(write=nullcontext));model=torch.nn.Linear(2,2)
 with tempfile.TemporaryDirectory(prefix='tta_completed_expert_cpu_',dir=O)as tmp:
  output=Path(tmp)/'release.json';items=[dict(case_id='a'),dict(case_id='b')]
  release_completed(schedule,c,fake,(model,),output,items,2);proof=json.loads(output.read_text());assert proof['real_planners_released']==2 and all(ref()is None for ref in refs)
  calls=[];value=object();base=SimpleNamespace(infer=lambda *a,**k:(calls.append(('infer',a,k)),value)[1]);cache=SimpleNamespace(encode=lambda *a,**k:(calls.append(('encode',a,k)),value)[1])
  cuda.reserved=100*2**20;cuda.free=100*2**20
  check=install(c,fake,(model,),base,cache,Path(tmp)/'forward.jsonl');before=cuda.calls
  assert base.infer(3,k='same')is value and cache.encode(4)is value and cuda.calls==before+1 and calls==[('infer',(3,),dict(k='same')),('encode',(4,),{})]
  cuda.free=2**30;cuda.reserved=100*2**20;check();assert cuda.calls==before+1
  errors=[]
  def other():
   try:check()
   except RuntimeError as e:errors.append(str(e))
  thread=threading.Thread(target=other);thread.start();thread.join();assert len(errors)==1
  # Still-owned experts must fail; never pretend a live object was destroyed.
  Proxy,shared=pooled();held=shared[0];schedule=SimpleNamespace(pooled=Proxy,tasks={},pool={},state={'a':dict(rows=[{},{}])})
  try:release_completed(schedule,c,fake,(model,),Path(tmp)/'bad.json',items,2)
  except AssertionError:pass
  else:raise AssertionError('Live completed planner was incorrectly accepted')
 assert not torch.cuda.is_initialized()
 write(O/'CPU_lifetime_checks.json',dict(passed=True,no_CUDA_context=True,completed_objects_proven_dead=True,still_live_expert_rejected=True,original_forward_args_and_result_identity_preserved=True,unused_only_cache_low_headroom_release=True,owner_and_high_headroom_negative_checks_passed=True))
 print('COMPLETED_EXPERT_AND_FORWARD_CACHE_CPU_PASS',flush=True)
if __name__=='__main__':main()

"""Original allocator callback, real CPU model tensors, explicit CUDA allocator fixture."""
from pathlib import Path
import threading,types
import torch
from scene_gate import SceneGate
from .common import T,O
from .forward_cache import install

def run():
 assert not torch.cuda.is_initialized()
 model=torch.nn.Linear(4,4).eval();x=torch.arange(8,dtype=torch.float32).reshape(2,4);expected=model(x).detach().clone()
 class CUDA:
  def __init__(self,free):self.free=free;self.allocated=9*2**30;self.reserved=self.allocated+323*2**20;self.empty_calls=0;self.rng=torch.arange(16,dtype=torch.uint8)
  def mem_get_info(self):return self.free,24*2**30
  def memory_allocated(self):return self.allocated
  def memory_reserved(self):return self.reserved
  def get_rng_state(self):return self.rng.clone()
  def synchronize(self):pass
  def empty_cache(self):self.empty_calls+=1;self.free+=self.reserved-self.allocated;self.reserved=self.allocated
 def experiment(free,expected_calls):
  cuda=CUDA(free);proxy=types.SimpleNamespace(cuda=cuda,random=torch.random,equal=torch.equal);c=types.SimpleNamespace(scene_gate=SceneGate());base=types.SimpleNamespace(infer=lambda v:model(v));cache=types.SimpleNamespace(encode=lambda fn,views,prompt:fn(views))
  before=[(v.data_ptr(),v._version,v.detach().clone())for v in model.parameters()];rng=torch.random.get_rng_state().clone()
  check=install(c,proxy,(model,),base,cache,O/'CPU_SAM2_unused_allocator_fixture.jsonl')
  check();assert torch.equal(base.infer(x),expected)and torch.equal(cache.encode(model,x,'fixture'),expected)
  assert cuda.empty_calls==expected_calls and all(ptr==v.data_ptr()and version==v._version and torch.equal(saved,v)for(ptr,version,saved),v in zip(before,model.parameters()))and torch.equal(rng,torch.random.get_rng_state())
  failures=[]
  def wrong_owner():
   try:check()
   except RuntimeError as error:failures.append(str(error))
  thread=threading.Thread(target=wrong_owner);thread.start();thread.join(2);assert not thread.is_alive()and len(failures)==1 and 'owner'in failures[0]
 experiment(2*2**30,0);experiment(59*2**20,1)
 a=T/'native_onpolicy_collection_v22';b=T/'native_onpolicy_collection_v24';assert(b/'forward_cache.py').read_text()==(a/'forward_cache.py').read_text()
 expected_runtime=(a/'runtime.py').read_text().replace('native_onpolicy_collection_v22','native_onpolicy_collection_v24').replace('       check_allocator()\n      with sam2_profile():return fn(*args,**kwargs)','       check_allocator()\n       with sam2_profile():return fn(*args,**kwargs)')
 assert expected_runtime==(b/'runtime.py').read_text()
 from .tracker_checks import run as owner_run
 owner_proof=owner_run();assert owner_proof['passed']
 assert not torch.cuda.is_initialized()
 return dict(passed=True,actual_production_tracker_owner_callback=owner_proof,actual_original_reclamation_function_used=True,actual_CPU_model_tensor_values_pointers_versions_and_RNG_exact=True,explicit_allocator_fixture_only=True,known_OOM59MiB_free323MiB_unused_exercises_one_reclaim=True,healthy_headroom_no_reclaim=True,wrong_owner_rejected=True,original_forward_cache_source_exact=True,runtime_only_adds_original_check_before_SAM2=True,no_actual_CUDA_native_or_efficacy_claim=True)

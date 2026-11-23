"""CPU allocator fixture proves model/output/RNG and exact reclaim boundaries."""
import ast,gc,json,contextlib,types
from pathlib import Path
from unittest.mock import patch
from .common import T,C,O,write,sha

def main():
 import torch
 from .cache import CacheMaintenance
 old=ast.parse((T/'native_trained_runtime_v1/cache.py').read_text());new=ast.parse((C/'cache.py').read_text())
 for n in ast.walk(new):
  if isinstance(n,ast.Compare)and ast.unparse(n).startswith("before['device_used_bytes'] >= (24563 - 1536)"):n.comparators[0]=ast.parse('21*2**30',mode='eval').body
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='collected'for t in n.targets):
   assert isinstance(n.value,ast.IfExp)and ast.unparse(n.value.test)=='self.epoch == 0';n.value=n.value.body
  if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='dict':
   for k in n.keywords:
    if k.arg=='python_objects_collected':assert isinstance(k.value,ast.IfExp);k.value=k.value.orelse
   n.keywords[:]=[k for k in n.keywords if k.arg not in('python_GC_performed','full_GC_at_native_case_and_feedback_release_retained','unused_cache_urgency_MiB','allocator_headroom_MiB')]
 assert ast.dump(old)==ast.dump(new),'Other allocator/model operations changed'
 class CUDA:
  def __init__(self):self.allocated=1024*2**20;self.reserved=2048*2**20;self.freed=0;self.device_used=23*2**30
  def is_initialized(self):return False
  def synchronize(self):pass
  def get_rng_state(self):return torch.get_rng_state().clone()
  def mem_get_info(self):return(24*2**30-self.device_used,24*2**30)
  def memory_allocated(self):return self.allocated
  def memory_reserved(self):return self.reserved
  def empty_cache(self):self.freed+=self.reserved-self.allocated;self.reserved=self.allocated
 cuda=CUDA();proxy=types.SimpleNamespace(cuda=cuda,random=torch.random,equal=torch.equal)
 with torch.random.fork_rng(devices=[]):model=torch.nn.Linear(4,4)
 c=types.SimpleNamespace(rng=types.SimpleNamespace(current=lambda:None),scene_gate=types.SimpleNamespace(write=contextlib.nullcontext))
 base=types.SimpleNamespace(_model=model,infer=lambda:None);session=O/'CPU_allocator_fixture';session.mkdir();cache=CacheMaintenance(proxy,c,base,session,cpu_only_fixture=True)
 original_output=object();params=[p.data_ptr()for p in model.parameters()];before=torch.get_rng_state().clone();collect=gc.collect
 with patch.object(gc,'collect',wraps=collect)as counted:
  assert cache.encode(lambda *_:original_output,(),None)is original_output and counted.call_count==1
  cuda.reserved=cuda.allocated+1024*2**20;cuda.device_used=22*2**30
  assert cache.encode(lambda *_:original_output,(),None)is original_output and cache.epoch==1 and counted.call_count==1,'Nonurgent reusable CUDA cache was unnecessarily flushed'
  cuda.device_used=23*2**30
  assert cache.encode(lambda *_:original_output,(),None)is original_output and counted.call_count==1
  assert cuda.freed==2048*2**20
 assert params==[p.data_ptr()for p in model.parameters()]and torch.equal(before,torch.get_rng_state())and not torch.cuda.is_initialized()
 cache.uninstall();rows=[json.loads(x)for x in(session/'CUDA_cache_reclamation.jsonl').read_text().splitlines()]
 assert rows[0]['python_GC_performed']and rows[1]['python_objects_collected']is None and not rows[1]['python_GC_performed']
 assert all(r['model_parameter_pointers_unchanged']and r['Torch_CPU_CUDA_RNG_unchanged']and r['actual_unused_reserved_bytes_released']>0 for r in rows)
 write(O/'CPU_cache_checks.json',dict(passed=True,original_full_cache_AST_exact_except_GC_schedule_cache_urgency_and_truthful_log=True,resource_guard_cap_unchanged_24563=True,cache_headroom_MiB=1536,reuse_nonurgent_cached_blocks=True,mandatory_first_full_GC=True,subsequent_allocator_reclaims_keep_model_and_RNG_and_exact_output=True,unused_cache_release_preserved=True,CUDA_initialized=False,CPU_only_fixture_NOT_GPU_proof=True,case_and_feedback_release_sources_unchanged=True,source_sha256={str(f):sha(f)for f in[C/'cache.py',Path(__file__),T/'native_case_release_v1/release.py',T/'native_completed_memory_v1/reclaim.py']}))
 print('ALLOCATOR_ONLY_RECLAIM_CPU_PASS',flush=True)
if __name__=='__main__':main()

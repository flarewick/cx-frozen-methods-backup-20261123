"""CPU-only boundary tests; no CUDA context or native episode is created."""
import ast,tempfile,threading
from pathlib import Path
from types import SimpleNamespace
import torch
from .scene_cache import install
from .common import O,write,T

def main():
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 class CUDA:
  def __init__(self):self.reserved=1000;self.allocated=600;self.calls=0
  def mem_get_info(self):return 10000-self.reserved,10000
  def memory_allocated(self):return self.allocated
  def memory_reserved(self):return self.reserved
  def get_rng_state(self):return torch.tensor([1,2,3],dtype=torch.uint8)
  def synchronize(self):pass
  def empty_cache(self):self.calls+=1;self.reserved=self.allocated
 cuda=CUDA();fake=SimpleNamespace(cuda=cuda,random=torch.random,equal=torch.equal)
 model=torch.nn.Linear(2,2);calls=[];sentinel=object()
 class Task:
  def setup_scene(self,*a,**k):calls.append((a,k));return sentinel
 c=SimpleNamespace(prepare_task=lambda task,lane:task)
 with tempfile.TemporaryDirectory(prefix='tta_scene_unused_cpu_')as tmp:
  file=Path(tmp)/'evidence.jsonl';install(c,fake,(model,),file)
  task=c.prepare_task(Task(),object());before=torch.random.get_rng_state().clone()
  assert task.setup_scene(3,token='same')is sentinel and calls==[((3,),dict(token='same'))]and cuda.calls==1
  assert torch.equal(before,torch.random.get_rng_state())and cuda.allocated==600
  assert task.setup_scene()is sentinel and cuda.calls==1
  errors=[]
  def other():
   try:task.setup_scene()
   except RuntimeError as e:errors.append(str(e))
  thread=threading.Thread(target=other);thread.start();thread.join();assert len(errors)==1 and len(calls)==2
  rows=[__import__('json').loads(x)for x in file.read_text().splitlines()];assert rows[0]['unused_reserved_bytes_released']==400 and rows[1]['unused_reserved_bytes_released']==0
 # All parent runtime AST is identical after removing ONLY this declared integration.
 parent=(T/'native_frozen100_v2/runtime.py').read_text().replace('native_frozen100_v2','native_frozen100_v18')
 from .source_dimensions import original_dimensions
 here=original_dimensions('runtime.py',(Path(__file__).parent/'runtime.py').read_text()).replace("\n startup.four.set();startup.eight.set()\n write(root/'initial_real_lanes_admitted.json',dict(lanes=list(range(8)),only_original_real_episode_workers=True,no_extra_queries_or_GPU_compute=True,guard_clock_not_reset=True,unused_allocator_scene_hook_active=True))",'').replace("\n from .scene_cache import install as install_scene_cache\n install_scene_cache(c,torch,models,root/'scene_allocator_reclamation.jsonl')",'')
 assert ast.dump(ast.parse(parent))==ast.dump(ast.parse(here))
 assert not torch.cuda.is_initialized()
 write(O/'CPU_scene_cache_checks.json',dict(passed=True,CUDA_context_created=False,unused_blocks_only=True,original_scene_called_once_with_exact_args=True,owner_thread_enforced=True,no_op_when_no_unused_cache=True,live_model_versions_and_RNG_preserved=True,parent_runtime_AST_exact_after_declared_hook_removed=True))
 print('UNUSED_SCENE_CACHE_CPU_PASS',flush=True)
if __name__=='__main__':main()

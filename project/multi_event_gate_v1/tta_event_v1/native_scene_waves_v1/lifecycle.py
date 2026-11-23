"""Scene-free cache release in a continuous model/guard lifetime."""
import gc, threading
TASKS=('stack_blocks_two','stack_bowls_two')
def plan(scenes,indices):
 if len(indices)!=len(set(indices)) or indices!=sorted(indices):raise ValueError('Unique original order required')
 pending={t:[i for i in indices if scenes[i]['task']==t]for t in TASKS}
 if sum(map(len,pending.values()))!=len(indices):raise ValueError('Unsupported task')
 waves=[]
 while any(pending.values()):
  remaining=sum(map(len,pending.values()))
  counts={t:len(pending[t])if remaining<=18 else min(6,len(pending[t]))for t in TASKS}
  wave=sorted(i for t in TASKS for i in pending[t][:counts[t]])
  if not wave:raise ValueError('No progress in wave planner')
  waves.append(wave)
  for t in TASKS:pending[t]=pending[t][counts[t]:]
 if sorted(i for w in waves for i in w)!=indices:raise ValueError('Coverage changed')
 return waves

def assert_released(coordinator,threads,pools):
 if any(t.is_alive()for t in threads):raise RuntimeError('Live worker forbids scene cache release')
 if any(p.tasks for p in pools):raise RuntimeError('Retained task pool forbids scene cache release')
 gc.collect()
 tasks=sum(r()is not None for r in coordinator.task_lifetimes)
 scenes=sum(r()is not None for r in coordinator.scene_lifetimes)
 if tasks or scenes:raise RuntimeError(f'Live scene references: tasks={tasks},scenes={scenes}')
 return dict(task_objects_released=len(coordinator.task_lifetimes),scene_objects_released=len(coordinator.scene_lifetimes),live_tasks_after_gc=0,live_scenes_after_gc=0)

def release(coordinator,threads,pools,model,*,torch):
 import sapien.render,time
 started=time.monotonic()
 if threading.get_ident()!=coordinator.service.owner:raise RuntimeError('Cache release requires model owner thread')
 with coordinator.scene_gate.write():
  torch.cuda.synchronize();counts=assert_released(coordinator,threads,pools)
  before_cpu=torch.random.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state().clone()
  pointers=[(p.data_ptr(),tuple(p.shape),str(p.dtype))for p in model.parameters()]
  free,total=torch.cuda.mem_get_info();before=dict(device_used_bytes=int(total-free),allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
  sapien.render.clear_cache(models=True,images=True,shaders=False)
  torch.cuda.empty_cache();torch.cuda.synchronize()
  assert torch.equal(before_cpu,torch.random.get_rng_state())and torch.equal(before_cuda,torch.cuda.get_rng_state())
  assert pointers==[(p.data_ptr(),tuple(p.shape),str(p.dtype))for p in model.parameters()]
  free,total=torch.cuda.mem_get_info();after=dict(device_used_bytes=int(total-free),allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
  coordinator.task_lifetimes.clear();coordinator.scene_lifetimes.clear()
 return dict(**counts,before=before,after=after,seconds=time.monotonic()-started,model_parameter_pointers_and_RNG_unchanged=True,shader_cache_preserved=True,continuous_GPU_process=True,statistics_reset=False)

"""Scene-free wave boundaries within one continuously guarded GPU process."""
import gc,weakref

def assert_released(coordinator):
 gc.collect();tasks=sum(r()is not None for r in coordinator.task_lifetimes);scenes=sum(r()is not None for r in coordinator.scene_lifetimes)
 if tasks or scenes:raise RuntimeError(f'Wave retains live simulation objects: tasks={tasks},scenes={scenes}')
 count=dict(task_objects_released=len(coordinator.task_lifetimes),scene_objects_released=len(coordinator.scene_lifetimes),live_tasks_after_gc=tasks,live_scenes_after_gc=scenes)
 coordinator.task_lifetimes.clear();coordinator.scene_lifetimes.clear();return count

def release_wave(coordinator,threads):
 if any(t.is_alive() for t in threads):raise ValueError('Cacheclear requires allwaveworkers stopped')
 import torch,sapien.render
 torch.cuda.synchronize();count=assert_released(coordinator)
 # No live Scene may reference these assets; keepcompiled shadercache unchanged.
 sapien.render.clear_cache(models=True,images=True,shaders=False)
 torch.cuda.empty_cache()
 return dict(**count,torch_allocated_bytes=torch.cuda.memory_allocated(),torch_reserved_bytes=torch.cuda.memory_reserved(),
  model_alive_gpu_process_continuous=True,shader_cache_preserved=True,statistics_reset=False)

def checks():
 from types import SimpleNamespace
 class Sim:
  def __init__(self):self.cycle=self
 def finished_wave():
  task=Sim();scene=Sim();return SimpleNamespace(task_lifetimes=[weakref.ref(task)],scene_lifetimes=[weakref.ref(scene)])
 c=finished_wave();r=assert_released(c);assert r['live_tasks_after_gc']==r['live_scenes_after_gc']==0 and r['scene_objects_released']==1
 live=Sim();c=SimpleNamespace(task_lifetimes=[weakref.ref(live)],scene_lifetimes=[]);refused=False
 try:assert_released(c)
 except RuntimeError:refused=True
 assert refused
 boundaries=list(range(6,38,6))+[38];assert boundaries==[6,12,18,24,30,36,38]
 return dict(wave_GC_weakreferences_prove_object_release_CPU_only=True,live_scene_or_task_refuses_cacheclear=True,continuous_process_wave_boundaries_preserve_all_case_indices=True)

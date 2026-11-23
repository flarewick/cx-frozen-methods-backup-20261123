import gc,threading

def verify_dead(task_ref,scene_ref):
 collected=gc.collect()
 if task_ref()is not None or scene_ref()is not None:raise RuntimeError('Completed task or scene still retained')
 return collected

def release(c,task_ref,scene_ref,*,torch,model):
 if threading.get_ident()!=c.service.owner:raise RuntimeError('Scene destruction requires owner')
 with c.scene_gate.write():
  torch.cuda.synchronize();cpu=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
  pointers=[p.data_ptr()for p in model.parameters()];free,total=torch.cuda.mem_get_info();before=int(total-free)
  collected=verify_dead(task_ref,scene_ref)
  torch.cuda.synchronize();free,total=torch.cuda.mem_get_info();after=int(total-free)
  assert pointers==[p.data_ptr()for p in model.parameters()]and torch.equal(cpu,torch.random.get_rng_state())and torch.equal(cuda,torch.cuda.get_rng_state())
 return dict(completed_task_released=True,completed_current_scene_released=True,GC_objects=collected,device_used_before_bytes=before,device_used_after_bytes=after,
  global_render_cache_NOT_cleared=True,other_lanes_NOT_stopped=True,model_parameter_pointers_and_RNG_unchanged=True,statistics_reset=False)

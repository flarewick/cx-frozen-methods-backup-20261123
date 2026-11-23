"""Release ONLY unused Torch allocator blocks before original scene creation."""
import json,threading

def install(coordinator,torch,models,logfile):
 owner=threading.get_ident();original_prepare=coordinator.prepare_task
 def pointers():return [[(p.data_ptr(),p._version,str(p.dtype),tuple(p.shape))for p in m.parameters()]for m in models]
 def memory():
  free,total=torch.cuda.mem_get_info()
  return dict(device_used_bytes=int(total-free),allocated_bytes=int(torch.cuda.memory_allocated()),reserved_bytes=int(torch.cuda.memory_reserved()))
 def prepare(task,lane):
  value=original_prepare(task,lane);original_scene=value.setup_scene
  def scene(*args,**kwargs):
   if threading.get_ident()!=owner:raise RuntimeError('Scene allocator lifecycle escaped original GPU owner')
   before=memory();cpu=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();params=pointers()
   if before['reserved_bytes']>before['allocated_bytes']:
    torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.synchronize()
   after=memory()
   assert params==pointers()and torch.equal(cpu,torch.random.get_rng_state())and torch.equal(cuda,torch.cuda.get_rng_state())
   assert before['allocated_bytes']==after['allocated_bytes'],'Live tensor ownership changed during unused-only reclamation'
   row=dict(before=before,after=after,unused_reserved_bytes_released=before['reserved_bytes']-after['reserved_bytes'],original_setup_scene_next=True,model_pointers_versions_dtypes_and_RNG_unchanged=True,live_tensor_bytes_unchanged=True,renderer_settings_physics_and_render_count_unchanged=True,statistics_reset=False,no_extra_GPU_compute=True)
   logfile.parent.mkdir(parents=True,exist_ok=True)
   with logfile.open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   return original_scene(*args,**kwargs)
  value.setup_scene=scene
  return value
 coordinator.prepare_task=prepare
 return prepare

"""Low-headroom unused allocator release before original real forwards."""
import json,threading

def install(c,torch,models,base,cache,output):
 owner=threading.get_ident()
 def check():
  if threading.get_ident()!=owner:raise RuntimeError('Forward allocator lifecycle escaped owner')
  free,total=torch.cuda.mem_get_info();allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
  if free>=1024*2**20 or reserved<=allocated:return
  versions=[[(p.data_ptr(),p._version,str(p.dtype),tuple(p.shape))for p in m.parameters()]for m in models]
  cpu=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
  with c.scene_gate.write():torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.synchronize()
  assert allocated==torch.cuda.memory_allocated()
  assert versions==[[(p.data_ptr(),p._version,str(p.dtype),tuple(p.shape))for p in m.parameters()]for m in models]
  assert torch.equal(cpu,torch.random.get_rng_state())and torch.equal(cuda,torch.cuda.get_rng_state())
  after_free,_=torch.cuda.mem_get_info();row=dict(free_before_bytes=int(free),free_after_bytes=int(after_free),unused_reserved_bytes_released=int(reserved-torch.cuda.memory_reserved()),live_tensor_bytes_unchanged=True,model_versions_and_Torch_RNG_unchanged=True,no_extra_queries_or_GPU_compute=True,statistics_reset=False)
  output.parent.mkdir(parents=True,exist_ok=True)
  with output.open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 original_infer=base.infer;original_encode=cache.encode
 def infer(*args,**kwargs):check();return original_infer(*args,**kwargs)
 def encode(*args,**kwargs):check();return original_encode(*args,**kwargs)
 base.infer=infer;cache.encode=encode
 return check

"""Reclaim unreachable Python objects and UNUSED CUDA blocks; no Scene cache purge."""
import gc,hashlib,json,os,threading
from native_policy_execution_v1.common import write
class CacheMaintenance:
 def __init__(self,torch,coordinator,base,session,*,cpu_only_fixture=False,auxiliary_models=()):
  self.torch,self.c,self.base,self.session=torch,coordinator,base,session
  self.fixture=cpu_only_fixture;self.owner=threading.get_ident();self.epoch=0;self.encoder_calls=0;self.last_encoder=0;self.active={}
  self.models=(base._model,*auxiliary_models)
  if not self.fixture:assert torch.cuda.is_initialized()and all(p.is_cuda for model in self.models for p in model.parameters())
  self.original=base.infer
  def infer(*args,**kwargs):
   epoch=self.epoch;result=self.original(*args,**kwargs);self.record('base',epoch);return result
  base.infer=infer
 def check_owner(self):
  if threading.get_ident()!=self.owner:raise RuntimeError('Cache lifecycle escaped actual GPU owner')
 def record(self,kind,epoch):
  self.check_owner();lane=self.c.rng.current()
  if lane is not None and lane.name in self.active:
   row=self.active[lane.name];key=kind+'_epochs';row[key][str(epoch)]=row[key].get(str(epoch),0)+1
 def encode(self,fn,views,prompt):
  self.check_owner();epoch=self.epoch;result=fn(views,prompt);self.record('encoder',epoch);self.encoder_calls+=1
  self.maybe_reclaim();return result
 def memory(self):
  cuda=self.torch.cuda;free,total=cuda.mem_get_info()
  return dict(device_used_bytes=int(total-free),allocated_bytes=int(cuda.memory_allocated()),reserved_bytes=int(cuda.memory_reserved()))
 def maybe_reclaim(self):
  self.check_owner();before=self.memory();cached=before['reserved_bytes']-before['allocated_bytes']
  if self.epoch==0:reason='mandatory_first_completed_real_warm_encoder'
  elif cached<512*2**20:return False
  elif before['device_used_bytes']>=21*2**30:reason='urgent_unused_CUDA_cache_pressure'
  elif self.encoder_calls-self.last_encoder>=32 and before['device_used_bytes']>=19*2**30:reason='spaced_unused_CUDA_cache_pressure'
  else:return False
  torch=self.torch;cpu=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
  pointers=[(p.data_ptr(),tuple(p.shape),str(p.dtype))for model in self.models for p in model.parameters()]
  with self.c.scene_gate.write():
   torch.cuda.synchronize();collected=gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
  after=self.memory()
  assert pointers==[(p.data_ptr(),tuple(p.shape),str(p.dtype))for model in self.models for p in model.parameters()]
  assert torch.equal(cpu,torch.random.get_rng_state())and torch.equal(cuda,torch.cuda.get_rng_state())
  self.epoch+=1;self.last_encoder=self.encoder_calls
  event=dict(epoch=self.epoch,reason=reason,encoder_calls=self.encoder_calls,inner_pid=os.getpid(),
    cpu_only_fixture=self.fixture,before=before,after=after,python_objects_collected=int(collected),
    actual_unused_reserved_bytes_released=before['reserved_bytes']-after['reserved_bytes'],
    model_parameter_pointers_unchanged=True,participant_model_count=len(self.models),Torch_CPU_CUDA_RNG_unchanged=True,
    no_SAPIEN_global_models_images_or_shaders_cleared=True,no_extra_inference_render_action_or_statistic_reset=True)
  file=self.session/'CUDA_cache_reclamation.jsonl';file.parent.mkdir(parents=True,exist_ok=True)
  with file.open('a')as f:f.write(json.dumps(event,allow_nan=False)+'\n')
  if self.epoch==1:write(self.session/'first_CUDA_cache_reclamation.json',event)
  print('USEFUL_UNUSED_CUDA_CACHE_RECLAIMED',self.epoch,event['actual_unused_reserved_bytes_released'],flush=True)
  return True
 def begin_case(self,lane,case_id):
  self.check_owner();assert lane.name not in self.active
  self.active[lane.name]=dict(case_id=case_id,base_epochs={},encoder_epochs={},tracker_epochs={})
 def finish_case(self,lane,folder):
  self.check_owner();value=self.active.pop(lane.name)
  value.update(cpu_only_fixture=self.fixture,actual_owner_forward_calls=True,
    post_reclamation_encoder_forward=any(int(k)>0 and n>0 for k,n in value['encoder_epochs'].items()),
    post_reclamation_base_forward=any(int(k)>0 and n>0 for k,n in value['base_epochs'].items()),
    post_reclamation_tracker_forward=any(int(k)>0 and n>0 for k,n in value['tracker_epochs'].items()),
    current_cache_epoch=self.epoch)
  write(folder/'cache_forward_witness.json',value)
 def uninstall(self):self.base.infer=self.original

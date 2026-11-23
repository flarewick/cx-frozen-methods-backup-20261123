"""CPU-only cache policy, real CPU model/RNG, mocked CUDA memory counters."""
from pathlib import Path
from contextlib import nullcontext
import hashlib,json,threading,types,uuid
import numpy as np,torch
from cx_tta_feedback_confirm5.cache_maintenance import CacheMaintenance
from cx_tta_feedback_confirm5.paths import OUTPUT,read,write
C=Path(__file__).resolve().parent
class CUDAFixture:
 def __init__(self):self.used=18*2**30;self.allocated=8*2**30;self.reserved=12*2**30;self.state=torch.arange(16,dtype=torch.uint8);self.clears=0
 def mem_get_info(self):return 24*2**30-self.used,24*2**30
 def memory_allocated(self):return self.allocated
 def memory_reserved(self):return self.reserved
 def get_rng_state(self):return self.state.clone()
 def synchronize(self):pass
 def empty_cache(self):self.used-=self.reserved-self.allocated;self.reserved=self.allocated;self.clears+=1
class Gate:
 def write(self):return nullcontext()

def main():
 session=OUTPUT/'cpu_cache_policy'/uuid.uuid4().hex;session.mkdir(parents=True)
 model=torch.nn.Linear(2,2).eval().requires_grad_(False);saved={k:v.clone()for k,v in model.state_dict().items()}
 original=lambda obs,**kw:dict(actions=np.zeros((32,14),np.float32))
 base=types.SimpleNamespace(_model=model,infer=original);lane=types.SimpleNamespace(name='0')
 c=types.SimpleNamespace(rng=types.SimpleNamespace(current=lambda:lane),scene_gate=Gate())
 cuda=CUDAFixture();fake=types.SimpleNamespace(cuda=cuda,random=torch.random,equal=torch.equal)
 aux=torch.nn.Linear(3,2).eval().requires_grad_(False);aux_before={k:v.clone()for k,v in aux.state_dict().items()}
 cache=CacheMaintenance(fake,c,base,session,cpu_only_fixture=True,auxiliary_models=(aux,));checks=0
 first=cache.encode(lambda views,prompt:np.array([3.]),None,'same')
 assert first.tolist()==[3.]and cache.epoch==1 and cuda.clears==1;checks+=3
 event=read(session/'first_CUDA_cache_reclamation.json')
 assert event['cpu_only_fixture']and event['actual_unused_reserved_bytes_released']==4*2**30;checks+=2
 assert event['model_parameter_pointers_unchanged']and event['Torch_CPU_CUDA_RNG_unchanged'];checks+=2
 cache.begin_case(lane,'CPU_fixture_cache_case');base.infer({});cache.record('tracker',cache.epoch)
 cache.encode(lambda views,prompt:np.array([4.]),None,'same')
 assert cache.epoch==1;checks+=1
 cuda.used=20*2**30;cuda.reserved=cuda.allocated+2**30
 assert cache.maybe_reclaim()is False;checks+=1
 cache.encoder_calls=cache.last_encoder+32
 assert cache.maybe_reclaim()is True and cache.epoch==2;checks+=2
 cuda.used=21*2**30;cuda.reserved=cuda.allocated+2**30
 assert cache.maybe_reclaim()is True and cache.epoch==3;checks+=2
 cuda.used=23*2**30;cuda.reserved=cuda.allocated+128*2**20
 assert cache.maybe_reclaim()is False;checks+=1
 failure=[]
 def cross_thread():
  try:cache.record('base',cache.epoch)
  except RuntimeError:failure.append(True)
 thread=threading.Thread(target=cross_thread);thread.start();thread.join();assert failure==[True];checks+=1
 folder=session/'case';folder.mkdir();cache.finish_case(lane,folder);witness=read(folder/'cache_forward_witness.json')
 assert witness['cpu_only_fixture']and witness['post_reclamation_encoder_forward']and witness['post_reclamation_base_forward'];checks+=3
 assert witness['post_reclamation_tracker_forward'];checks+=1
 assert all(torch.equal(aux_before[k],v)for k,v in aux.state_dict().items());checks+=1
 cache.uninstall();assert base.infer is original;checks+=1
 assert all(torch.equal(saved[k],v)for k,v in model.state_dict().items());checks+=1
 try:CacheMaintenance(torch,c,base,session)
 except AssertionError:checks+=1
 else:raise AssertionError('Actual GPU mode accepted uninitialized CUDA/CPU model')
 assert not torch.cuda.is_initialized();checks+=1
 files=(C/'cache_checks.py',C/'cx_tta_feedback_confirm5/cache_maintenance.py')
 write(OUTPUT/'cache_checks.json',dict(passed=True,checks=checks,raw_root=str(session),
  cpu_only_fixture=True,CUDA_initialized=False,actual_CPU_model_values_unchanged=True,
  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},
  scope='Real CPU model/RNG/thread guards; MOCK CUDA memory and NO real Scene/GPU allocation. Actual post-reclaim native bridges/resources still mandatory.'))
 print('UNUSED_CUDA_CACHE_POLICY_CPU_FIXTURE_PASS',checks,flush=True)
if __name__=='__main__':main()

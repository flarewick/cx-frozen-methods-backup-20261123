"""Bounded actual RGB prefix byte-parity and timing; zero policy/simulation actions."""
import time,pickle,random,hashlib
from pathlib import Path
from .common import read,write,verify,sha

def execute(spec,root,prepared):
 import torch,numpy as np
 from adapter_cache_cpu import move_to_gpu
 from cx_tta_feedback_confirm5.views import batched_encode,view_inputs
 from .encoding import build
 base=move_to_gpu(prepared['base']);base._model.eval().requires_grad_(False);torch.set_num_threads(2)
 model=base._model;versions=[(n,p.data_ptr(),p._version,str(p.dtype),tuple(p.shape))for n,p in model.named_parameters()]
 write(root/'barrier/cuda_ready.json',dict(ready=True))
 began=time.monotonic()
 while not(root/'barrier/owned_pid_permit.json').exists():
  if time.monotonic()-began>45:raise TimeoutError('No ownedGPUpermit')
  time.sleep(.03)
 fast=build();rows=[]
 def rng():return (pickle.dumps((random.getstate(),np.random.get_state()),protocol=5),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
 def same_rng(a,b):return a[0]==b[0]and torch.equal(a[1],b[1])and torch.equal(a[2],b[2])
 with torch.inference_mode():
  for item in spec['probe_inputs']:
   file=Path(item['file']);assert sha(file)==item['sha256']
   with np.load(file,allow_pickle=False)as z:images=tuple(z['cam'+str(i)].copy()for i in range(3));state=z['command_state'].copy()
   views=view_inputs(images,state,images,state);before=rng();torch.cuda.synchronize();t=time.perf_counter();a,ma=batched_encode(base,views,item['instruction']);torch.cuda.synchronize();old=time.perf_counter()-t;mid=rng()
   t=time.perf_counter();b,mb=fast(base,views,item['instruction']);torch.cuda.synchronize();new=time.perf_counter()-t;after=rng()
   exact=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes();unchanged=same_rng(before,mid)and same_rng(mid,after)
   row=dict(file=str(file),sha256=item['sha256'],features_byte_exact=exact,metadata_exact=ma==mb,RNG_exact=unchanged,original_seconds=old,graph_seconds=new,includes_capture=len(rows)==0,max_abs_difference=float(np.max(np.abs(a.astype(np.float32)-b.astype(np.float32)))))
   rows.append(row);write(root/('case_%02d.json'%(len(rows)-1)),row);print('PREFIX_GRAPH_ACTUAL',row,flush=True)
   assert exact and ma==mb and unchanged,'Graph prefix differs from original current batch3 encoder'
 assert versions==[(n,p.data_ptr(),p._version,str(p.dtype),tuple(p.shape))for n,p in model.named_parameters()]
 verify(spec);steady=rows[1:];old=float(np.median([r['original_seconds']for r in steady]));new=float(np.median([r['graph_seconds']for r in steady]))
 write(root/'GPU_probe_result.json',dict(passed=True,actual_cases=len(rows),all_features_metadata_RNG_exact=True,model_parameter_pointers_versions_unchanged=True,original_action_sampler_unchanged=True,policy_episodes=0,policy_actions=0,original_median_seconds=old,graph_steady_median_seconds=new,steady_speedup=old/new,capture_seconds=rows[0]['graph_seconds'],rows=rows,sustained_resource_qualification_NOT_claimed=True))

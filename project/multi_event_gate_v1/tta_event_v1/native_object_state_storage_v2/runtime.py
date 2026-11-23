"""Fixed saved RGB software parity; no VLA, physics, labels or adaptation."""
import time,os,gc
import numpy as np
from .common import read,write,mapped

def compare(a,b):
 assert a.keys()==b.keys()
 for key in a:
  if isinstance(a[key],np.ndarray):assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes(),key
  else:assert a[key]==b[key],key

def state_bytes(state):
 import torch
 seen=set();sizes={'cpu':0,'cuda':0}
 def walk(value):
  if isinstance(value,torch.Tensor):
   storage=value.untyped_storage();key=(str(value.device),storage.data_ptr())
   if key not in seen:seen.add(key);sizes[value.device.type]+=storage.nbytes()
  elif isinstance(value,dict):
   for item in value.values():walk(item)
  elif isinstance(value,(list,tuple)):
   for item in value:walk(item)
 walk(state);return sizes

def execute(spec,root,prepared,barrier):
 import torch
 from native_object_feedback_v3.tracking import MaskCore
 from .core import CPUStateCore
 from cx_tta_feedback_confirm5.model_math import sam2_profile
 model=prepared['model'].to('cuda').eval().requires_grad_(False);torch.cuda.synchronize()
 write(barrier/'cuda_ready.json',dict(inner_pid=os.getpid(),gpu_uuid=spec['required_gpu_uuid']))
 began=time.monotonic()
 while not(barrier/'owned_pid_permit.json').exists():
  if time.monotonic()-began>30:raise TimeoutError('GPU owner PID permit missing')
  time.sleep(.05)
 assert read(barrier/'owned_pid_permit.json')['gpu_uuid']==spec['required_gpu_uuid']
 versions=[(id(v),v.data_ptr(),v._version)for v in model.parameters()];reports=[]
 for item in spec['probe_cases']:
  with np.load(mapped(item['file']),allow_pickle=False)as z:values={k:z[k].copy()for k in z.files}
  outputs=[];storage=[];seconds=[]
  for cls in [MaskCore,CPUStateCore]:
   start=time.monotonic();core=cls(model)
   with torch.inference_mode(),sam2_profile():
    rows=[core.initialize(values['initial'],values['masks'],tick=0,decision_tick=0)]
    for rgb,tick in zip(values['frames'],values['ticks'],strict=True):rows.append(core.append(rgb,tick=int(tick),decision_tick=int(tick)))
   torch.cuda.synchronize();storage.append(state_bytes(core.stream.state));seconds.append(time.monotonic()-start)
   assert core.count==item['actual_object_count']and list(core.stream.state['obj_id_to_idx'])==list(range(core.count))
   outputs.append(rows);del core;gc.collect();torch.cuda.empty_cache()
  for a,b in zip(*outputs,strict=True):compare(a,b)
  assert storage[1]['cuda']<storage[0]['cuda'],'No actual GPU state storage was saved'
  reports.append(dict(task=item['task'],actual_object_count=item['actual_object_count'],actual_frames=9,
   all_masks_presence_raw_logits_bytes_exact=True,original_state_bytes=storage[0],candidate_state_bytes=storage[1],
   observed_seconds_original_candidate=seconds,speedup_NOT_inferred=True))
  write(root/('case_result_'+str(len(reports))+'.json'),reports[-1])
 assert versions==[(id(v),v.data_ptr(),v._version)for v in model.parameters()]
 write(root/'GPU_probe_result.json',dict(passed=True,software_only=True,sustained_resource_qualification=False,
  all_one_two_object_raw_outputs_exact=True,actual_frames_compared=sum(r['actual_frames']for r in reports),
  cases=reports,model_parameter_pointers_and_mutation_counters_unchanged=True,no_VLA_physics_labels_updates_or_TTA_benefit=True))

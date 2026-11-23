"""Arbitrary-size prepared CPU seeds, keyed only by actual q0 RGB/language/models."""
from pathlib import Path
import time,hashlib
import numpy as np
from native_cpu_seed_v4.common import T,P,read,sha,initial,rgb_sha,language_sha
class PreparedCPUSeedStore:
 def __init__(self,root):
  self.root=Path(root);self.result=read(self.root/'result.json');self.protocol=read(self.root/'protocol.json')
  assert self.result['complete']and self.result['CPU_models_unchanged']and not self.result['CUDA_initialized']
  assert self.result['requested']==len(self.result['rows'])+len(self.result['missing'])>0
  assert sha(self.root/'protocol.json')==self.result['protocol_sha256']
  for group in('source_sha256','input_sha256'):
   for p,h in self.protocol[group].items():assert sha(p)==h
  expected=read(T/'outputs/native_arrived_mask_v3/cpu_models.json')['model_state_sha256']
  self.models={name:expected[name]for name in('DINO','SAM')};self.rows={}
  for row in self.result['rows']:
   assert row['model_state_sha256']==self.models and row['actual_CPU_inference']
   key=(row['head_RGB_sha256'],row['instruction_sha256'])
   if key in self.rows:raise ValueError('Ambiguous duplicate initial CPU input key')
   assert row['cold_CPU_inference_seconds']>0;self.rows[key]=row
 def at_initial_observation(self,images,instruction,*,query,tick):
  began=time.monotonic_ns()
  if type(query)is not int or type(tick)is not int or query!=0 or tick!=0:raise ValueError('Only actual query0/tick0')
  images=initial({'cam'+str(i):v for i,v in enumerate(images)})
  key=(rgb_sha(images[0]),language_sha(instruction));row=self.rows.get(key);masks=None
  value=dict(earliest_three_camera_bundle_sha256=hashlib.sha256(b''.join(v.tobytes()for v in images)).hexdigest(),
   instruction_sha256=key[1],head_RGB_sha256=key[0],CPU_model_state_sha256=self.models,
   actual_query=0,actual_tick=0,no_later_masks_imported=True,preparation_result_sha256=sha(self.root/'result.json'))
  if row is None:value.update(available=False,reason='no_exact_prepared_CPU_initial_RGB_language',initialization_preparation_required=True)
  else:
   value.update(available=row['available'],reason=row['reason'],initialization_preparation_required=False,
     cold_CPU_inference_seconds=row['cold_CPU_inference_seconds'],prepared_mask_semantic_GT_certified=False)
   if row['available']:
    assert sha(row['file'])==row['sha256']
    with np.load(row['file'],allow_pickle=False)as z:masks=z['masks'].copy()
    assert masks.shape==(2,240,320)and masks.dtype==bool
    value.update(mask_file=row['file'],mask_sha256=row['sha256'])
  value.update(accepted_monotonic_ns=time.monotonic_ns(),lookup_seconds=(time.monotonic_ns()-began)/1e9,initialization_cost_NOT_hidden_in_lookup=True)
  return value,masks

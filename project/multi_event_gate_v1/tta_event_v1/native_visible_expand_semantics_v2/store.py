"""Content-only exact q0 deployment acceptance; a cache miss fails before sampling."""
from pathlib import Path
import hashlib,time
import numpy as np
from native_policy_execution_v1.common import read,sha

def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def key(images,command,instruction):
 if len(images)!=3 or any(im.shape!=(240,320,3)or im.dtype!=np.uint8 for im in images):raise ValueError('Actual original3 camera RGB required')
 if command.shape!=(14,)or not np.isfinite(command).all():raise ValueError('Actual initial commanded state required')
 return tuple(array_sha(im)for im in images),array_sha(command),hashlib.sha256(instruction.encode()).hexdigest()
class SeedStore:
 def __init__(self,folder):
  self.folder=Path(folder);runs=list(self.folder.glob('host_run_*'));assert len(runs)==1 and read(runs[0]/'exit.json')['exit_code']==0 and read(runs[0]/'exit.json')['actual_Popen_wait_return'];file=self.folder/'result.json';self.result=read(file);protocol=read(self.folder/'protocol.json')
  assert self.result['complete']and self.result['CPU_models_unchanged']and not self.result['CUDA_initialized']and self.result['all_RNG_preserved']
  assert self.result['requested']==len(self.result['rows'])==protocol['requested']==12
  assert self.result['protocol_sha256']==sha(self.folder/'protocol.json')
  for field in('source_sha256','input_sha256'):
   for p,h in protocol[field].items():assert sha(p)==h
  self.rows={};self.result_sha256=sha(file)
  for row in self.result['rows']:
   assert row['query']==row['tick']==0 and row['initialization_eval_mode']
   assert row['models_state_sha256']==self.result['models_state_sha256']
   if not row['frame_available']:
    assert not row['available']and row['reason']=='native_initialization_missing';continue
   assert sha(row['frame_file'])==row['frame_sha256']
   with np.load(row['frame_file'],allow_pickle=False)as z:
    actual=key(tuple(z['cam'+str(i)]for i in range(3)),z['command_state'],next(j['instruction']for j in protocol['jobs']if j['case_id']==row['case_id']))
   expected=(tuple(row['RGB_sha256']),row['command_state_sha256'],row['instruction_sha256']);assert actual==expected
   if expected in self.rows:raise ValueError('Ambiguous duplicate content cache key')
   self.rows[expected]=row
 def at_initial(self,observed,instruction,*,query,tick):
  began=time.monotonic()
  if type(query)is not int or type(tick)is not int or query!=0 or tick!=0:raise ValueError('Only actual earliest q0/tick0 semantic acceptance')
  content=key(observed['rgb'],observed['command_state'],instruction)
  if content not in self.rows:raise ValueError('No exact evaluation-mode q0 RGB/command/language preparation; stop before actor draw')
  row=self.rows[content];masks=None
  if row['available']:
   assert sha(row['mask_file'])==row['mask_sha256']
   with np.load(row['mask_file'],allow_pickle=False)as z:masks=z['masks'].copy()
   assert masks.shape==(2,240,320)and masks.dtype==bool
  expected=dict(rgb=tuple(im.copy()for im in observed['rgb']),command_state=observed['command_state'].copy(),instruction_sha256=content[2],available=bool(row['available']))
  report=dict(available=bool(row['available']),reason=row['reason'],models_state_sha256=row['models_state_sha256'],CPU_result_sha256=self.result_sha256,cold_CPU_inference_seconds=row['cold_CPU_inference_seconds'],actual_lookup_seconds=time.monotonic()-began,query=0,tick=0,all_three_RGB_command_language_exact=True,old_expert_masks_NOT_imported=True,case_task_seed_not_cache_keys=True)
  return expected,masks,report

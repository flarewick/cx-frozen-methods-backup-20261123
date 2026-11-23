"""Actual q0 cache acceptance and old/new-mode differences on developer cases."""
from pathlib import Path
import numpy as np,torch
from native_policy_execution_v1.common import T,read,write,sha
from native_trained_semantics_v1.store import SeedStore

def main():
 store=SeedStore(T/'outputs/native_trained_semantics_v1');protocol=read(store.folder/'protocol.json');index=read(T/'outputs/native_expert_full_feedback_v1/index.json');records=[];rejections=0
 for frame in protocol['jobs']:
  with np.load(frame['file'],allow_pickle=False)as z:observed=dict(rgb=tuple(z['cam'+str(j)].copy()for j in range(3)),command_state=z['command_state'].copy())
  expected,masks,report=store.at_initial(observed,frame['instruction'],query=0,tick=0)
  assert report['available']and masks.shape==(2,240,320)and masks.dtype==bool
  assert all(np.array_equal(a,b)for a,b in zip(expected['rgb'],observed['rgb'],strict=True))and np.array_equal(expected['command_state'],observed['command_state'])
  bad=dict(observed,rgb=tuple(x.copy()for x in observed['rgb']));bad['rgb'][2][0,0,0]^=1
  badstate=dict(observed,command_state=observed['command_state'].copy());badstate['command_state'][0]+=np.float32(.01)
  for current,instruction,q,tick in((bad,frame['instruction'],0,0),(badstate,frame['instruction'],0,0),(observed,frame['instruction']+' changed',0,0),(observed,frame['instruction'],1,0),(observed,frame['instruction'],0,1)):
   try:store.at_initial(current,instruction,query=q,tick=tick)
   except ValueError:rejections+=1
   else:raise AssertionError('Misbound initial semantic cache accepted')
  old=next(c for c in index['cases']if c['case_id']==frame['case_id']);first=old['observations'][0]
  with np.load(first['file'],allow_pickle=False)as z:
   camera_equal=[bool(np.array_equal(observed['rgb'][j],z['cam'+str(j)]))for j in range(3)]
   command_equal=bool(np.array_equal(observed['command_state'],z['command_state']))
  records.append(dict(case_id=frame['case_id'],actual_eval_mode_semantic_available=True,old_expert_q0_camera_equal=camera_equal,old_expert_q0_command_equal=command_equal))
 assert rejections==20 and not torch.cuda.is_initialized()
 write(T/'outputs/native_trained_runtime_v1/CPU_actual_initial_checks.json',dict(passed=True,actual_prepared_cases=4,initial_misbinding_rejections=rejections,records=records,new_policy_queries=0,new_native_policy_rollouts=0,CUDA_initialized=False,source_sha256={str(p):sha(p)for p in(Path(__file__),T/'native_trained_semantics_v1/store.py')}))
 print('ACTUAL_DEVELOPER_EVAL_Q0_BINDING_CHECKS_PASS',4,rejections,records,flush=True)
if __name__=='__main__':main()

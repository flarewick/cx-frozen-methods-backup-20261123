"""CPU audit of actual native q0 provenance, not semantic/rollout benefit."""
from .paths import *
def audit(spec,root):
 import numpy as np
 verify(spec);complete=read(root/'GPU_frames_complete.json')
 assert complete['complete']and complete['protocol_identity']==spec['identity']
 assert len(complete['rows'])==len(spec['scenes'])==12
 for source,row in zip(spec['scenes'],complete['rows'],strict=True):
  assert row['case_id']==source['case_id']and row['instruction']==source['instruction']
  assert row['query']==row['tick']==row['actual_native_action_count']==0
  if not row['frame_available']:
   assert row['file']is None and row['sha256']is None and row['initialization_failure']['type']=='UnStableError'and row['no_substitution'];continue
  assert sha(row['file'])==row['sha256']
  assert row['initialization_eval_mode']and row['native_step_limit']>0 and not row['new_independent_scene']
  assert len(row['physical_sha256'])==len(row['rng_sha256'])==64
  with np.load(row['file'],allow_pickle=False)as z:
   assert set(z.files)=={'cam0','cam1','cam2','command_state'}
   assert all(z['cam'+str(i)].shape==(240,320,3)and z['cam'+str(i)].dtype==np.uint8 for i in range(3))
   assert z['command_state'].shape==(14,)and np.isfinite(z['command_state']).all()
 result=dict(passed=True,requested_developer_cases=12,actual_developer_q0_frames=sum(r['frame_available']for r in complete['rows']),initialization_missing=[r['case_id']for r in complete['rows']if not r['frame_available']],eval_mode=True,actual_policy_actions=0,new_independent_scenes=0,semantic_masks_not_yet_prepared=True,sustained_resource_qualification_claimed=False,TTA_utility_proven=False)
 if not(root/'CPU_frame_audit.json').exists():write(root/'CPU_frame_audit.json',result)
 else:assert read(root/'CPU_frame_audit.json')==result
 return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();print(audit(read(O/'protocol.json'),a.dataset),flush=True)

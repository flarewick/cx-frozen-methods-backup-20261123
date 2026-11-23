"""Fail closed on same-world readout mismatch; no simulation or actuation."""
import hashlib,json,threading
import numpy as np
KEYS={'arm_position_target','arm_velocity_target','gripper_position_target','gripper_velocity_target','gripper_logical_command','gripper_joint_counts'}
class ReadoutBridge:
 def __init__(self,original,candidate):self.original=original;self.candidate=candidate;self.local=threading.local()
 def begin_case(self,case_id,lane):
  if getattr(self.local,'case',None)is not None:raise RuntimeError('Previous case not closed')
  self.local.case=dict(case_id=case_id,lane=lane,native_calls=0,sampled_pairs=[],poisoned=False)
 def capture(self,robot):
  case=getattr(self.local,'case',None)
  if case is None or case['poisoned']:raise RuntimeError('No healthy native readout case')
  index=case['native_calls']
  try:
   row=self.candidate(robot)
   if set(row)!=KEYS:raise ValueError('Unexpected commanded readout fields')
   if index<128 or index%256==0:
    reference=self.original(robot)
    if set(reference)!=KEYS:raise ValueError('Original commanded readout fields changed')
    digest=hashlib.sha256()
    for key in sorted(KEYS):
     a,b=np.asarray(row[key]),np.asarray(reference[key])
     if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):raise ValueError('Same-state commanded readout mismatch: '+key)
     digest.update(json.dumps([key,a.dtype.str,list(a.shape)]).encode());digest.update(np.ascontiguousarray(a).tobytes())
    case['sampled_pairs'].append(dict(call_index=index,all_fields_byte_exact=True,readout_sha256=digest.hexdigest()))
   case['native_calls']+=1
   return row
  except BaseException:
   case['poisoned']=True;raise
 def finish_case(self):
  case=getattr(self.local,'case',None)
  if case is None or case['poisoned']:raise RuntimeError('Missing/poisoned native bridge cannot qualify')
  self.local.case=None
  return dict(**case,paired_calls=len(case['sampled_pairs']),first_paired_calls_per_case=128,subsequent_pair_interval=256,
   same_state_before_native_physics=True,no_extra_actions_queries_or_holds=True,all_sampled_fields_byte_exact=True,
   full_dynamics_or_TTA_utility_NOT_claimed=True)

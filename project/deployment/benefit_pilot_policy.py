"""Frozen shared250 alpha.5, zero-repeat controls and one first-event correction."""
import numpy as np
from task_adapters.policy import StackingPolicy as Original
from candidate_eval_checkpoint import candidate
from benefit_pilot_protocol import decode,sha
from cx_tta.structured_action_adapter import ARM_INDICES
from cx_tta.eval_diagnostics import array_hash
from cx_tta.official_onset_domain import observe_native_arms
class PilotPolicy(Original):
 def __init__(self,base,training_run,condition,config):
  try:task,branch=decode(condition)
  except ValueError:
   super().__init__(base,training_run,condition,config);self.pilot_branch=None;return
  super().__init__(base,training_run,'M_first_full'if branch=='B'else'Z_zero',config)
  payload,path,spec=candidate('M_shared250_a050');self.model.load_state_dict(payload['model'],strict=True);self.model.eval().requires_grad_(False)
  self.condition=condition;self.pilot_branch=branch;self.derived_checkpoint_sha256=sha(path)
 def predict(self,*args,**kwargs):
  if self.pilot_branch is None:return super().predict(*args,**kwargs)
  external=self.condition;self.condition='M_first_full'if self.pilot_branch=='B'else'Z_zero'
  try:_,raw_applied,base=super().predict(*args,**kwargs)
  finally:self.condition=external
  delta=np.zeros_like(base);delta[:,ARM_INDICES]=raw_applied[:,ARM_INDICES]*np.float32(.5)
  actions=base.copy();mask=delta!=0;actions[mask]=base[mask]+delta[mask]
  self.last_numeric.update(actions=actions.copy(),applied_delta=delta.copy())
  active=self.last_diagnostics['multi_event']['active']
  self.last_diagnostics.update(evaluated_condition=external,pilot_branch=self.pilot_branch,candidate_alpha=.5,arm_strength=.5 if active else 0.,action_chunk_sha256=array_hash(actions),applied_delta_sha256=array_hash(delta),applied_max_abs_correction=float(np.abs(delta).max()),max_abs_correction=float(np.abs(delta).max()),robot_domain=observe_native_arms(actions,self.joint_limits),candidate_checkpoint_sha256='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7')
  self.last_diagnostics['prescription']['strength']=.5 if active else 0.
  return actions,delta,base

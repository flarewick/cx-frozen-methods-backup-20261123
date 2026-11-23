"""Frozen shared250/alpha.5 event_full with optional first-candidate veto only."""
import json
import numpy as np
from task_adapters.policy import StackingPolicy as Original
from candidate_eval_checkpoint import candidate
from candidate_eval_policy import scaled_actions
from cx_tta.multi_event import EventConfig
from cx_tta.eval_diagnostics import array_hash
from cx_tta.official_onset_domain import observe_native_arms
from benefit_event_full_veto_v132 import FirstCandidateVeto
import benefit_followup_protocol_v2 as p
class FactorySafeFirstCandidateVeto(FirstCandidateVeto):
 # deepcopy probes __setstate__ before it has populated original; use a direct
 # attribute lookup so that an uninitialized wrapper raises AttributeError.
 # The entire scored decision step remains the inherited, verified132 method.
 def __getattr__(self,key):return getattr(object.__getattribute__(self,'original'),key)

class FrozenFullPolicyProbeCompatible(Original):
 def __init__(self,base,training_run,condition,config=EventConfig()):
  try:task,branch=p.decode(condition)
  except ValueError:
   if condition not in('L0','Z_zero','M_first_full','M_event_full'):raise
   self.stage_branch=None;self.pilot_branch=None;self.external_condition=condition;self.candidate_alpha=1.
   Original.__init__(self,base,training_run,condition,config)
   return
  if config!=EventConfig()or config.max_interventions!=3:raise ValueError('Exact original full event configuration required')
  self.stage_branch=branch;self.pilot_branch=branch;self.external_condition=condition
  Original.__init__(self,base,training_run,'M_event_full'if branch in('B','G')else'Z_zero',config)
  payload,path,spec=candidate('M_shared250_a050')
  original={k:v.clone()for k,v in self.model.state_dict().items()if k.startswith('geometry_head.')or k in('caps','reg_mean','time_encoding')}
  self.model.load_state_dict(payload['model'],strict=True);self.model.eval().requires_grad_(False)
  import torch
  assert all(torch.equal(self.model.state_dict()[k],v)for k,v in original.items())
  assert spec['alpha']==.5;self.candidate_alpha=.5;self.derived_checkpoint_sha256=p.sha(path)
  if branch=='G':
   assert self.controller.config==self.event_config
   self.controller=FactorySafeFirstCandidateVeto(self.controller,json.loads((p.ROOT/'gate.json').read_text()))
  self.condition=condition
 def reset_history(self):Original.reset_history(self)
 def stage_admit(self,q,packet,context):
  # The archived StagePolicy scorer hook remains always-admit here. The optional
  # controller wrapper is the ONLY source of candidate vetoes.
  return dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.)
 def predict(self,observation,instruction,noise,*,enabled):
  if self.stage_branch is None:return Original.predict(self,observation,instruction,noise,enabled=enabled)
  external=self.condition;self.condition='M_event_full'if self.stage_branch in('B','G')else'Z_zero'
  try:_,raw_delta,base=Original.predict(self,observation,instruction,noise,enabled=enabled)
  finally:self.condition=external
  actions,delta=scaled_actions(base,raw_delta,self.candidate_alpha);active=self.last_diagnostics['multi_event']['active'];self.last_numeric.update(actions=actions.copy(),applied_delta=delta.copy())
  self.last_diagnostics.update(evaluated_condition=external,pilot_branch=self.stage_branch,candidate_alpha=.5,candidate_checkpoint_sha256=self.derived_checkpoint_sha256,arm_strength=.5 if active else 0.,action_chunk_sha256=array_hash(actions),applied_delta_sha256=array_hash(delta),max_abs_correction=float(np.abs(delta).max()),applied_max_abs_correction=float(np.abs(delta).max()),robot_domain=observe_native_arms(actions,self.joint_limits),full_event_budget=3,candidate_veto_scope='first_scored_candidate_only; later candidates unchanged',candidate_veto_records=list(getattr(self.controller,'last_veto_records',[])))
  self.last_diagnostics['prescription']['strength']=.5 if active else 0.
  return actions,delta,base

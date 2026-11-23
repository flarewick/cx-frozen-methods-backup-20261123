"""Candidate veto around an EXISTING event_full controller; predictor/executor untouched.

Only the first scored candidate is covered by the first-event-trained model.
Later candidates keep the original scorer. This is not an all-event-trained gate.
"""
import copy
class FirstCandidateVeto:
 def __init__(self,controller,model=None):
  from cx_tta.multi_event import LocalController
  if type(controller)is not LocalController:raise TypeError('An original LocalController instance is required')
  self.original=controller;self.model=copy.deepcopy(model);self.first_scored_query=None;self.last_veto_records=[]
 def __getattr__(self,key):return getattr(self.original,key)
 def reset(self):
  self.original.reset();self.first_scored_query=None;self.last_veto_records=[]
 def step(self,query,event,packet,scorer,*,enabled):
  self.last_veto_records=[]
  if self.model is None:return self.original.step(query,event,packet,scorer,enabled=enabled)
  from benefit_fixed_ensemble_gate_v119 import predict
  def veto(p,context):
   if self.first_scored_query is None:self.first_scored_query=query
   if query!=self.first_scored_query:return scorer(p,context)
   result=predict(self.model,p,context)
   self.last_veto_records.append(dict(query=query,context=list(map(float,context)),scores=dict(result),admit=result['rescue']-2*result['harm']>0))
   return result
  return self.original.step(query,event,packet,veto,enabled=enabled)
def bind_first_candidate_veto(policy,model):
 """Call after constructing the unchanged frozen event_full policy, once per lane."""
 if policy.event_config.max_interventions!=3:raise ValueError('This integration requires the original event_full budget=3, not first-only budget=1')
 if isinstance(policy.controller,FirstCandidateVeto):raise RuntimeError('Controller already wrapped')
 if policy.controller.config!=policy.event_config:raise ValueError('Original event config mismatch')
 policy.controller=FirstCandidateVeto(policy.controller,model)
 return policy

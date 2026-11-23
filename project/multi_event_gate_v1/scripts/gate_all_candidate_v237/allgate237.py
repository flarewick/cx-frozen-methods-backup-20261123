"""Frozen v118 at every controller scoring opportunity; uncalibrated later-event diagnostic."""
import copy
from benefit_event_full_veto_v132 import FirstCandidateVeto
class AllCandidateVeto(FirstCandidateVeto):
 def __getattr__(self,key):return getattr(object.__getattribute__(self,'original'),key)
 def step(self,query,event,packet,scorer,*,enabled):
  self.last_veto_records=[]
  if self.model is None:return self.original.step(query,event,packet,scorer,enabled=enabled)
  from benefit_fixed_ensemble_gate_v119 import predict
  def veto(p,context):
   if self.first_scored_query is None:self.first_scored_query=query
   result=predict(self.model,p,context)
   self.last_veto_records.append(dict(query=query,context=list(map(float,context)),scores=dict(result),admit=result['rescue']-2*result['harm']>0))
   return result
  return self.original.step(query,event,packet,veto,enabled=enabled)

def install():
 import benefit_event_full_policy_v142 as policy
 if getattr(policy,'_all237',False):return
 policy.FactorySafeFirstCandidateVeto=AllCandidateVeto
 Original=policy.FrozenFullPolicyProbeCompatible
 class AllEventPolicy(Original):
  def predict(self,*args,**kw):
   result=super().predict(*args,**kw)
   if self.stage_branch is not None:self.last_diagnostics['candidate_veto_scope']='all_legal_scored_candidates; frozen118 uncalibrated later-event transfer'
   return result
 import benefit_stage_policy as stage
 stage.StagePolicy=AllEventPolicy
 policy._all237=True

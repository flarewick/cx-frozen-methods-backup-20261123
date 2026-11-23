"""Reproduce the actual original FrozenFactory probe AND G lane construction on CPU."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import benefit_followup_protocol_v2 as p
from frozen_factory import FrozenFactory
from cx_tta.multi_event import EventConfig
from benefit_event_full_policy_v142 import FrozenFullPolicyProbeCompatible,Original

def main():
 torch.set_num_threads(1);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized()
 def no_inference(*a,**kw):raise AssertionError('Factory CPU verification must not run VLA inference')
 base=SimpleNamespace(_model=object(),_sample_actions=no_inference);factory=FrozenFactory(FrozenFullPolicyProbeCompatible);checked=[]
 for condition in ['L0','Z_zero',*[p.condition('lift_pot',b)for b in ['A0','B','G']]]:
  a=factory(base,p.TRAINING,condition,EventConfig());b=factory(base,p.TRAINING,condition,EventConfig())
  assert a.base_policy is b.base_policy is base and a.previous is b.previous is None and a.query_count==b.query_count==0
  assert a.controller is not b.controller and a.detector is not b.detector
  assert all(torch.equal(v,b.model.state_dict()[k])for k,v in a.model.state_dict().items())
  if condition in('L0','Z_zero'):
   original=Original(base,p.TRAINING,condition,EventConfig());assert all(torch.equal(v,original.model.state_dict()[k])for k,v in a.model.state_dict().items());assert a.stage_branch is None
  else:assert a.event_config.max_interventions==b.event_config.max_interventions==3 and a.candidate_alpha==b.candidate_alpha==.5
  if condition==p.condition('lift_pot','G'):
   assert a.controller.model==b.controller.model and a.controller.model is not b.controller.model
   a.controller.original.interventions=1;assert b.controller.original.interventions==0;a.reset_history();assert a.controller.original.interventions==0 and a.controller.first_scored_query is None
  print('FACTORY149_CONDITION_PASSED',condition,flush=True)
  checked.append(dict(condition=condition,actual_original_FrozenFactory_two_clones=True,independent_controller_detector=True,model_weights_identical=True))
 p.atomic(p.ROOT/'policy_probe_factory_CPU_proof_v149.json',dict(passed=True,no_GPU=True,no_VLA_inference=True,original_L0_Zzero_probe_fallback_weights_exact=True,actual_G_controller_deepcopy_and_reset_passed=True,conditions=checked,policy_sha256=p.sha(p.D/'benefit_event_full_policy_v142.py'),checker_sha256=p.sha(__file__)));print('FULL_EVENT_FACTORY_PROBES_AND_G_CPU_PASSED',flush=True)
if __name__=='__main__':main()

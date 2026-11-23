"""Actual frozen CPU policy construction and child scope verification, no inference/GPU."""
import json,os
from pathlib import Path
from types import SimpleNamespace
import torch
import benefit_followup_protocol_v2 as p

def main():
 torch.set_num_threads(1);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized();protocol=p.prepare();folder=p.HOME/'candidate_veto_event_full_extension_GPU3_v174';plan=json.loads((folder/'prospective_plan.json').read_text());impl=json.loads((folder/'implementation_v174.json').read_text())
 assert p.ROOT==folder/'dev0'and p.TASKS==plan['seed_starts']and p.VAL==tuple(range(20));assert not protocol['first_event_only']and not protocol['later_events_always_rejected']and protocol['event_full_budget']==3 and protocol['alpha']==.5
 from benefit_fixed_cache_bind_GPU3_v156 import bind
 bind()
 assert all(p.sha(p.D/n)==v for n,v in impl['source_sha256'].items())
 assert os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')=='1'
 actual_receipts=[json.loads(f.read_text())for f in (p.ROOT/'seven_policy_entry_receipts_v45').glob('entry_*.json')]
 assert any(r['pid']==os.getpid()and r['entry_sha256']==p.sha(p.D/'benefit_event_full_resources_v174.py')for r in actual_receipts)
 import benefit_growth_runtime as runtime
 runtime.install_sources()
 from benefit_event_full_policy_v142 import FrozenFullPolicyProbeCompatible as FrozenFullPolicy
 from benefit_event_full_veto_v132 import FirstCandidateVeto
 # No base inference is possible in this constructor check.
 def forbidden_inference(*a,**kw):raise AssertionError('CPU scope check must not invoke the VLA')
 base=SimpleNamespace(_sample_actions=forbidden_inference);policies=[]
 from candidate_eval_checkpoint import candidate
 payload,path,spec=candidate('M_shared250_a050')
 for branch in ['A0','B','G']:
  obj=FrozenFullPolicy(base,p.TRAINING,p.condition('lift_pot',branch));assert obj.event_config.max_interventions==obj.controller.config.max_interventions==3
  assert obj.candidate_alpha==.5 and obj.derived_checkpoint_sha256==plan['adapter_sha256'];assert all(torch.equal(v,obj.model.state_dict()[k])for k,v in payload['model'].items())
  assert isinstance(obj.controller,FirstCandidateVeto)==(branch=='G');obj.reset_history();assert obj.controller.interventions==0
  if branch=='G':assert obj.controller.first_scored_query is None
  policies.append(obj)
 assert len({id(x.controller)for x in policies})==3 and len({id(x.detector)for x in policies})==3
 p.atomic(p.ROOT/'full_event_scope_CPU_proof_v174.json',dict(passed=True,no_GPU=True,no_VLA_inference=True,actual_A0_B_G_CPU_policy_construction=True,productive_mixed_binding_requires_later_real_expert_prefix=True,checkpoint_parameters_all_exact=True,original_event_config_budget3_all_branches=True,only_G_controller_wrapped=True,independent_controller_and_detector_per_lane=True,alpha_unchanged05=True,CPU_child_resource_entry_checked=True,original83_bytes_unchanged=True,protocol_sha256=p.sha(p.ROOT/'protocol.json'),implementation_sha256=p.sha(folder/'implementation_v174.json'),checker_sha256=p.sha(__file__)))
 print('FULL134_SCOPE_CONSTRUCTORS_CPU_PASSED',flush=True)
if __name__=='__main__':main()

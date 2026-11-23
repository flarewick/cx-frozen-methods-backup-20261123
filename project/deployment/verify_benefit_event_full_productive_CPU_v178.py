"""After REAL finite experts: verify actual productive binding and full-event plan."""
import json
from pathlib import Path
import torch
import benefit_followup_protocol_v2 as p

def main():
 torch.set_num_threads(1);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized()
 receipt=json.loads((p.ROOT/'finite_bootstrap_host_receipt_v6.json').read_text());assert receipt['passed']and receipt['actual_gpu']==3
 import benefit_growth_runtime as runtime;runtime.install_sources();runtime.install()
 import benefit_stage_policy as stage
 from benefit_event_full_policy_v142 import FrozenFullPolicyProbeCompatible
 assert stage.StagePolicy is FrozenFullPolicyProbeCompatible
 from benefit_event_full_policy_v142 import FactorySafeFirstCandidateVeto
 from benefit_event_full_veto_v132 import FirstCandidateVeto
 assert FactorySafeFirstCandidateVeto.step is FirstCandidateVeto.step
 import shared6_eval_backend as backend
 plan=backend.make_plan('benefit40_collect',p.TRAINING,'GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a',lanes=4,episodes=100)
 assert len(plan['jobs'])==6
 for j in plan['jobs']:
  assert p.decode(j['condition'])==(j['task'],j['branch'])
  assert j['manifest']['pilot']['max_interventions']==(3 if j['branch']=='B'else 0)
  assert j['manifest']['checkpoint_sha256']==p.sha(p.CANDIDATE)
 p.atomic(p.ROOT/'actual_productive_policy_binding_CPU_v178.json',dict(passed=True,no_GPU=True,actual_real_expert_prefix_required_and_present=True,actual_stage_policy_bound_to_full_class=True,all6_actual_preflight_manifest_budgets_match_full_event_policy=True,plan_source_sha256=plan['source_sha256'],checker_sha256=p.sha(__file__),protocol_sha256=p.sha(p.ROOT/'protocol.json')));print('FULL_EVENT_PRODUCTIVE_BINDING_CPU_PASSED',flush=True)
if __name__=='__main__':main()

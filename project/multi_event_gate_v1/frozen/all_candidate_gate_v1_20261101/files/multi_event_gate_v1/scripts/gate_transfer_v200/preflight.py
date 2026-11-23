"""Fresh transfer CPU construction + AST scope audit; no outcome/model fitting."""
import copy,json,sys,os
from pathlib import Path
import torch
import benefit_followup_protocol_v2 as p
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import entry

def main():
 torch.set_num_threads(1);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized()
 p.prepare();assert set(p.TASKS)=={'place_empty_cup','stack_bowls_two'}
 from benefit_fixed_cache_bind_GPU3_v156 import bind;bind()
 import verify_benefit_event_full_scope_CPU_v174 as scope;scope.main()
 import verify_benefit_event_full_factory_CPU_v149 as factory
 # Factory implementation uses a condition string only: bind its historic lift alias to new task.
 old=p.condition
 p.condition=lambda task,b:old('place_empty_cup'if task=='lift_pot'else task,b)
 try:factory.main()
 finally:p.condition=old
 from benefit_event_full_policy_v142 import FrozenFullPolicyProbeCompatible
 from frozen_factory import FrozenFactory
 from types import SimpleNamespace
 from cx_tta.multi_event import EventConfig
 def no(*a,**kw):raise AssertionError('No VLA inference allowed')
 ff=FrozenFactory(FrozenFullPolicyProbeCompatible);base=SimpleNamespace(_model=object(),_sample_actions=no)
 for task in p.TASKS:
  for branch in ('A0','B','G'):
   policy=ff(base,p.TRAINING,p.condition(task,branch),EventConfig());assert policy.stage_branch==branch and policy.candidate_alpha==.5 and policy.event_config.max_interventions==3
   policy.reset_history()
 from benefit_event_full_veto_v132 import FirstCandidateVeto
 assert FrozenFullPolicyProbeCompatible.stage_admit(None,None,None,None)==dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.)
 impl=json.loads((p.ROOT.parent/'implementation_v174.json').read_text());assert all(p.sha(p.D/n)==h for n,h in impl['source_sha256'].items())
 p.atomic(p.ROOT/'transfer_scope_CPU.json',dict(passed=True,no_GPU=True,no_fit=True,all_task_branch_FrozenFactory_construction=True,original_gate_model_sha256=p.sha(p.ROOT/'gate.json'),adapter_sha256=p.sha(p.CANDIDATE),task_scope=entry.TASKS,AST_scope_edits=entry.EDITS,all_original_sources_verified=True,helper_sha256=p.sha(__file__)))
 print('TRANSFER200_SCOPE_CPU_PASSED',flush=True)
if __name__=='__main__':main()

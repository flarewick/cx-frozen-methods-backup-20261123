"""Check full-event budget extension against original action/clock audit on real TRAIN data."""
import json
from pathlib import Path
import torch
import benefit_followup_protocol_v2 as p
from benefit_event_full_audit_v134 import install_full_budget_audit

def main():
 torch.set_num_threads(1);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized()
 import benefit_stage_audit as stage
 from benefit_pilot_audit import episode
 original=stage.audit_episode;proof=install_full_budget_audit();root=p.HOME/'supervision_growth_canonical_v96/train2';plan=json.loads((root/'collect_plan.json').read_text());a=json.loads((root/'collection_audit.json').read_text());assert a['passed']and a['complete'];checked=[]
 for task in p.TASKS:
  case=next(c for c in a['cases']if c['task']==task);job=next(j for j in plan['jobs']if j['task']==task and j['branch']=='B');value=episode(job,case['seed']);before=original('B',value);after=stage.audit_episode('B',value);assert before==after;checked.append(dict(task=task,seed=case['seed'],proof=after))
 p.atomic(p.ROOT/'full_event_audit_CPU_proof_v135.json',dict(passed=True,no_GPU=True,no_new_outcomes=True,actual_recorded_episode_audits_equal=checked,only_full_event_budget_extension=proof,checker_sha256=p.sha(__file__),auditor_sha256=p.sha(p.D/'benefit_event_full_audit_v134.py')));print('FULL_EVENT_AUDIT_CPU_PASSED',flush=True)
if __name__=='__main__':main()

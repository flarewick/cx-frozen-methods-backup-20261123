"""Append the remaining original expert decisions with explicit origin; no new sampling."""
import json,os,copy
from pathlib import Path
import torch
assert not torch.cuda.is_initialized()and not list(Path('/dev').glob('nvidia*'));torch.set_num_threads(1)
import benefit_followup_protocol_v2 as p,benefit_growth_runtime as runtime
runtime.install_sources()
import benefit_followup_mixed_core_v3 as core
from shared6_eval_backend import sources
cohort=int(os.environ['CX_GATE_TRANSFER_COHORT']);old=p.HOME/('candidate_veto_cup_bowls_v200_'+str(cohort))/'dev0'
reg=json.loads((p.ROOT.parent/'preregistration237.json').read_text())
for f,h in reg['controls_origin_sha256'].items():
 file=Path(f)
 if not file.exists():file=p.R/file.relative_to('/data/qiufw/cx_migration/cx_new_direction_guide_TTA')
 assert p.sha(file)==h
q=json.loads((old/'qualified.json').read_text());ledger=core.state()
for task in p.TASKS:
 existing=ledger.ledger.tasks[task]['attempts'];rows=q['tasks'][task]['attempts'];assert existing==rows[:len(existing)]
 for row in rows[len(existing):]:ledger.submit(task,copy.deepcopy(row),p.ROOT/core.EVENTS,origin='explicit237_import_of_certified_original_expert:'+p.sha(old/'qualified.json'))
assert ledger.ledger.complete()and ledger.ledger.tasks==q['tasks']
ledger.persist(p.ROOT/core.EVENTS)
value=ledger.seal(sources());value.update(qualification_stream_scope_sha256=p.sha(core.scope_path()),gpu_uuid='GPU-fa87bc41-1919-efe1-73f9-11f38008763e',expert_runtime_audits={},original_expert_source=str(old/'qualified.json'),original_expert_source_sha256=p.sha(old/'qualified.json'),no_new_expert_execution_claim=True)
p.atomic(p.ROOT/'qualified.json',value)
p.atomic(p.ROOT/'expert_membership_import237.json',dict(passed=True,source=str(old/'qualified.json'),source_sha256=p.sha(old/'qualified.json'),new_sha256=p.sha(p.ROOT/'qualified.json'),all_original_decisions_order_physical_hashes_exact=True,only_protocol_provenance_rebound=True,no_new_GPU_collection=True,old_policy_outcomes_not_imported=True))
print('237_FULL_FIXED_EXPERT_MEMBERSHIP_PASSED',flush=True)

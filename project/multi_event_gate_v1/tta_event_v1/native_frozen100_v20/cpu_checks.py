import ast,json,threading
from .common import C,T,O,TASKS,read,write,sha

def main():
 for p in C.rglob('*.py'):ast.parse(p.read_text())
 from .mixed_contracts import audit
 assert audit()
 from .episode import frozen_execute
 from native_best100_endpoint_v1.episode import execute
 assert frozen_execute is execute
 assert len(TASKS)==4 and read(O/'retained_candidate.json')['checkpoint_sha256']=='0937511affea00771ed7f595d07220ae390ba85e06e398fce678446e7078a1d5'
 # Original record validator rejects non-contiguous seeds and wrong scene initial identity.
 from .preparation import validate_records
 start=dict(next_seed=100000,accepted=0);items=[];rows=[]
 for i in range(3):
  item=dict(task=TASKS[0],seed=100000+i,episode_index=i,instruction='place object on pad',case_id=TASKS[0]+':'+str(100000+i),initial_physical_sha256='a'*64);items.append(item)
  rows.append(dict(task=TASKS[0],seed=100000+i,episode_index=i,eligible=True,expert_plan_success=True,expert_success=True,initial_physical_sha256='a'*64))
 assert validate_records(start,rows,items,TASKS[0])['end_state']==dict(next_seed=100003,accepted=3)
 try:validate_records(start,list(reversed(rows)),items,TASKS[0])
 except AssertionError:pass
 else:raise AssertionError('Noncontiguous preparation accepted')
 # All feasible remaining paired counts used by this continuation produce exact totals.
 for p in range(24,45):
  q=67;rounds=(q+2)//3;short=rounds*3-q;double=p-rounds
  if not 0<=short<=double<=rounds:continue
  widths=[(2,3)]*(double-short)+[(1,3)]*(rounds-double)+[(2,2)]*short
  assert sum(a for a,b in widths)==p and sum(b for a,b in widths)==q and all(2*(a+b)in(8,10)for a,b in widths)
 import torch
 assert not torch.cuda.is_initialized()
 write(O/'CPU_entry_contract_checks.json',dict(passed=True,scientific_runtime_AST_exact_v13_after_only_explicit_task_slot_and_reference_substitutions=True,original_baseline_full_TTA_implementations_unchanged=True,original_models_and_update_math_unchanged=True,remaining_four_task_denominators_exact100=True,per_task_original_seed_order_and_expert_validation_preserved=True,mixed_only_real_requested_episodes=True,source_AST_parsed=True,GPU_started=False))
 print('MIXED_FOUR_TASK_CPU_CONTRACTS_PASS')
if __name__=='__main__':main()

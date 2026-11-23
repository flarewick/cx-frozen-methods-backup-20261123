"""All261 mapping checks reject future-target cherry-picking; no model fit."""
from pathlib import Path
import torch
from .data import T,read,sha,inventory
from .qualified_reader import QualifiedSelectedReader
from native_policy_execution_v1.common import write

def main():
 manifest,cases,plans,excluded=inventory();reader=QualifiedSelectedReader();accepted=0;rejected=0
 for plan in plans:
  p=plan['packet'];q=plan['alignment']['next_selected_at_or_after_feedback']['query'];assert reader.check_selection(p,q)['query']==q;accepted+=1
  for alternative in(None,q+1,p['query']if p['query']!=q else q-1):
   try:reader.check_selection(p,alternative)
   except ValueError:rejected+=1
   else:raise AssertionError('Alternative target accepted')
 for p in manifest['packets']:
  if reader.targets[p['case_id'],p['source_query'],p['query']]is None:
   try:reader.check_selection(p,p['query'])
   except ValueError:rejected+=1
   else:raise AssertionError('No-future packet got a fabricated target')
 assert accepted==174 and rejected==609 and not torch.cuda.is_initialized()
 C=Path(__file__).resolve().parent
 write(T/'outputs/native_selected_outer_target_v1/CPU_selection_gate_checks.json',dict(passed=True,accepted_exact_targets=accepted,rejected_alternate_or_absent_targets=rejected,optimizer_steps=0,CUDA_initialized=False,source_sha256={str(p):sha(p)for p in(C/'qualified_reader.py',Path(__file__),C/'data.py')}))
 print('ALL261_SELECTED_ACTION_TARGET_GATE_PASS',accepted,rejected,flush=True)
if __name__=='__main__':main()

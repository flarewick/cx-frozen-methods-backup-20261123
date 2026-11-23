"""CPU-only whole original full/baseline comparison, never an online target."""
from .common import T,read,write,sha,mapped
REFERENCE=T/'outputs/native_object_transfer_pilot_v2/development/batches/000/runs/f2bb5f409e694dc1b64773507f3ac04a'
def reference_inputs():
 files=[REFERENCE/'protocol.json',REFERENCE/'terminal_audit.json']
 for case in sorted((REFERENCE/'cases').glob('*')):
  files.append(REFERENCE/'accepted'/(case.name+'.json'))
  for branch in ('baseline','M_event_full'):files.append(case/branch/'result.json')
 return {str(p):sha(p)for p in files}
def verify_original_prefix(spec,root):
 from native_visible_expand_c_v1.analysis import same_query
 audit=read(root/'CPU_episode_audit.json');assert audit['passed']and audit['same_prefix_query_checks']>0
 checked=cases=0;bindings={}
 if spec['stage']=='development'and spec['batch_index']==0:
  for case in sorted((root/'cases').glob('*')):
   assert read(root/'accepted'/(case.name+'.json'))==read(REFERENCE/'accepted'/(case.name+'.json'))
   for branch in ('baseline','M_event_full'):
    old=read(REFERENCE/'cases'/case.name/branch/'result.json');fresh=read(case/branch/'result.json')
    assert old['success']==fresh['success']and old['native_action_count']==fresh['native_action_count']and len(old['queries'])==len(fresh['queries'])
    for a,b in zip(old['queries'],fresh['queries'],strict=True):same_query(a,b);checked+=1
    for p in(REFERENCE/'cases'/case.name/branch/'result.json',case/branch/'result.json'):bindings[str(p)]=sha(p)
    cases+=1
  assert cases==20 and checked>0
 write(root/'CPU_original_prefix_comparison.json',dict(passed=True,historical_prefix_comparison_applicable=cases>0,original_frozen_full_and_baseline_whole_branches_exact=cases>0,matched_branches=cases,matched_policy_queries=checked,original_reference=str(REFERENCE)if spec['stage']=='development'else None,new_independent_confirmation_has_no_old_run=spec['stage']=='confirmation',actual_current_zero_shadow_controls_checked=True,reference_not_forwarded_to_models=True,input_sha256=bindings))

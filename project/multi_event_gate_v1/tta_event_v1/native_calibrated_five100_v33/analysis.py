"""Whole paired native validation; no teacher labels and no outcome filtering."""
from .common import read, write, verify, sha


def audit(spec, root):
 from native_visible_expand_c_v1.analysis import branch, same_query, arrays
 from native_onpolicy_calibration_data_v24.reader import Reader
 import numpy as np
 verify(spec)
 complete=read(root/'GPU_episodes_complete.json')
 assert complete['complete'] and complete['protocol_identity']==spec['identity']
 assert complete['frozen_parameter_pointers_and_mutation_counters_unchanged']
 assert complete['official_EVAL_excluded_from_fitting'] and len(complete['cases'])==spec['cases_per_batch']
 records=[]
 for bundle in complete['cases']:
  assert bundle['initialized'] and bundle['all_requested_conditions_retained']
  folder=root/'cases'/bundle['case_id']; accepted=read(root/'accepted'/(bundle['case_id']+'.json'))
  controls=accepted['episode_index']==0
  wanted={'baseline','M_event_full','TTA'}|({'zero_update','shadow_update'} if controls else set())
  assert {r['condition']for r in bundle['branches']}==wanted
  results={}; summaries={}
  for witness in bundle['branches']:
   condition=witness['condition']; assert sha(witness['result_file'])==witness['result_sha256']
   result, summary=branch(folder/condition,condition,spec)
   semantic=read(folder/condition/'semantic_initial_receipt.json')
   assert semantic['initial_diagnostic_identity']==bundle['initial_identity']
   summary.update(semantic_available=semantic['available'],semantic_reason=semantic['reason'])
   if condition in('TTA','shadow_update'):
    assert result['feedback']['action_visible_fixed_parameters_and_buffers_unchanged']
   results[condition]=result; summaries[condition]=summary
  Reader().overlap_proof(folder)
  base,full,tta=(results[c]for c in('baseline','M_event_full','TTA'))
  assert not base['feedback']['updates'] and base['completed_sources']==0
  for query in base['queries']:
   action=arrays(query['actions_file']); policy=arrays(query['policy']['policy_file'])
   assert np.array_equal(action['actions'],policy['base']) and not np.any(action['applied_delta'])
   assert query['event']['baseline_once_base_verbatim']
  assert not full['feedback']['updates']
  first=None
  for a,b in zip(full['queries'],tta['queries']):
   same=np.array_equal(arrays(a['actions_file'])['actions'],arrays(b['actions_file'])['actions'])
   same_query(a,b,actions=same,trace=same,actual_actions_only=True)
   if not same:
    first=b['query']
    assert b['event']['adaptation_applied']
    assert any(u['accepted'] and u['query']<=first for u in tta['feedback']['updates'])
    break
  if first is None:
   assert len(full['queries'])==len(tta['queries']) and full['success']==tta['success']
  for condition in('zero_update','shadow_update'):
   if condition not in results:continue
   control=results[condition]
   assert control['success']==full['success'] and control['native_action_count']==full['native_action_count']
   assert len(control['queries'])==len(full['queries'])
   for a,b in zip(full['queries'],control['queries'],strict=True):same_query(a,b)
  records.append(dict(**accepted,baseline=base['success'],full=full['success'],TTA=tta['success'],
   rescue=not full['success'] and tta['success'],harm=full['success'] and not tta['success'],
   first_changed_query=first,branches=summaries,original_zero_shadow_controls=controls,
   initial_identity=bundle['initial_identity'],no_policy_outcome_filter=True))
 assert all(sum(r['task']==t for r in records)==spec['task_slots'].count(t) for t in spec['tasks'])
 from .closed_audit import audit as audit_closed
 closed=audit_closed(root,spec)
 value=dict(passed=True,records=records,actual_native_episodes=sum(len(r['branches'])for r in records),
  original_zero_shadow_controls_exact=True,closed_expert_task_lifetimes=closed,
  no_recovery_teacher_or_fitting=True,official_EVAL_used=True,formal_success_not_used_for_fitting=True,
  qualification_sha256=sha(spec['trained_qualification']),scene_role=spec['scene_role'])
 write(root/'CPU_episode_audit.json',value);write(root/'CPU_full_capture_audit.json',value)
 return value

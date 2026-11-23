"""Whole native branches and same-prefix paired outcomes; no fitting."""
from .common import read,write,verify,sha

def audit(spec,root):
 from native_visible_expand_c_v1.analysis import branch,same_query,arrays
 import numpy as np
 verify(spec);complete=read(root/'GPU_episodes_complete.json');assert complete['complete']and complete['protocol_identity']==spec['identity']and len(complete['cases'])==spec['cases_per_batch'] and complete['frozen_parameter_pointers_and_mutation_counters_unchanged']
 records=[];checks=0
 for bundle in complete['cases']:
  assert bundle['initialized']and bundle['all_requested_conditions_retained'];folder=root/'cases'/bundle['case_id'];accepted=read(root/'accepted'/(bundle['case_id']+'.json'));results={};summaries=[]
  for witness in bundle['branches']:
   condition=witness['condition'];assert sha(witness['result_file'])==witness['result_sha256'];result,summary=branch(folder/condition,condition,spec)
   assert read(folder/condition/'semantic_initial_receipt.json')['initial_diagnostic_identity']==bundle['initial_identity']
   if condition in('shadow_update','TTA'):assert result['feedback']['action_visible_fixed_parameters_and_buffers_unchanged']
   semantic=read(folder/condition/'semantic_initial_receipt.json')
   summary.update(semantic_available=semantic['available'],semantic_unavailable_reason=semantic['reason'],perception_interface=spec['perception_interface'],selected_event_queries=sum(bool(q['event']['decision']['active'])for q in result['queries']))
   results[condition]=result;summaries.append(summary)
  baseline=results['baseline'];full=results['M_event_full'];tta=results['TTA'];first=None
  for q in baseline['queries']:
   act=arrays(q['actions_file']);policy=arrays(q['policy']['policy_file'])
   assert np.array_equal(act['actions'],policy['base'])and not np.any(act['applied_delta'])
   assert q['event']['baseline_once_base_verbatim']and not q['event']['decision']['active']and not q['event']['adaptation_applied']
  assert baseline['completed_sources']==0 and not baseline['feedback']['updates']
  full_first=None
  for a,b in zip(baseline['queries'],full['queries']):
   same=np.array_equal(arrays(a['actions_file'])['actions'],arrays(b['actions_file'])['actions'])
   same_query(a,b,actions=same,trace=same,actual_actions_only=True);checks+=1
   if not same:
    full_first=b['query'];assert b['event']['decision']['active'];break
  if full_first is None:assert len(baseline['queries'])==len(full['queries'])and baseline['success']==full['success']
  for a,b in zip(full['queries'],tta['queries']):
   same=np.array_equal(arrays(a['actions_file'])['actions'],arrays(b['actions_file'])['actions']);same_query(a,b,actions=same,trace=same,actual_actions_only=True);checks+=1
   if not same:
    first=b['query'];assert b['event']['adaptation_applied']and any(u['accepted']and u['query']<=first for u in tta['feedback']['updates']);break
  if first is None:assert len(full['queries'])==len(tta['queries'])and full['success']==tta['success']
  for condition in('zero_update','shadow_update'):
   if condition not in results:continue
   z=results[condition];assert z['success']==full['success']and z['native_action_count']==full['native_action_count']and len(z['queries'])==len(full['queries'])
   for a,b in zip(full['queries'],z['queries'],strict=True):same_query(a,b);checks+=1
  records.append(dict(**accepted,baseline=baseline['success'],baseline_to_full_first_changed_query=full_first,full=full['success'],TTA=tta['success'],rescue=not full['success']and tta['success'],harm=full['success']and not tta['success'],first_changed_query=first,initial_identity=bundle['initial_identity'],branches=summaries))
 assert len(records)==spec['cases_per_batch'] and all(sum(r['task']==t for r in records)==spec['task_slots'].count(t) for t in spec['tasks'])
 projection_missing={condition:0 for condition in spec['conditions']};projection_sources=[]
 for source_file in sorted((root/'cases').glob('*/*/source*/source.json')):
  source=read(source_file);missing=[r for r in source['missing']if r['reason']=='commanded_TCP_projection_unsupported']
  for row in missing:
   assert row['no_target_generated']and row['calibrated_projection_unchanged']
   assert not any(r['start_tick']==row['start_tick']and r['end_tick']==row['end_tick']for r in source['rows'])
  condition=source_file.parents[1].name
  if condition in projection_missing:projection_missing[condition]+=len(missing)
  projection_sources.append(dict(file=str(source_file),sha256=sha(source_file),unsupported_windows=len(missing)))
 endpoint_evidence=[]
 for f in sorted((root/'cases').glob('*/*/execution*.json')):
  meta=read(f);w=meta.get('endpoint_readback')
  if w is not None:
   assert all(w[k]for k in('native_observation_cadence_preserved','physical_and_own_command_exact','callback_not_previously_dispatched','only_one_feedback_frame_dispatched'))
   assert meta['complete_source_chunk']and w['tick']==meta['committed_ticks']
   endpoint_evidence.append(dict(file=str(f),sha256=sha(f),**w))
 result=dict(passed=True,endpoint_readback_version='native_best100_endpoint_v1',endpoint_readback_evidence=endpoint_evidence,projection_guard_version='native_best100_projection_guard_v1',projection_missing_windows=projection_missing,projection_source_evidence=projection_sources,records=records,actual_native_episodes=sum(len(r['branches'])for r in records),same_prefix_query_checks=checks,all_native_receipts_source_phase_once_noise_reset_and_release_checked=True,qualification_sha256=sha(spec['trained_qualification']),screening_state=read(root/'screening_complete.json')['state'],formal_EVAL_not_used=True,scene_role=spec['scene_role'],interface_qualification_only=True)
 write(root/'CPU_episode_audit.json',result);return result

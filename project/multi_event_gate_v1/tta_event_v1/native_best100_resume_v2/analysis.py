"""Whole native branches and same-prefix paired outcomes; no fitting."""
from .common import read,write,verify,sha

def audit(spec,root):
 from native_visible_expand_c_v1.analysis import branch,same_query,arrays
 import numpy as np
 verify(spec);complete=read(root/'GPU_episodes_complete.json');assert complete['complete']and complete['protocol_identity']==spec['identity']and len(complete['cases'])==8 and complete['frozen_parameter_pointers_and_mutation_counters_unchanged']
 records=[];checks=0
 for bundle in complete['cases']:
  assert bundle['initialized']and bundle['all_requested_conditions_retained'];folder=root/'cases'/bundle['case_id'];accepted=read(root/'accepted'/(bundle['case_id']+'.json'));results={};summaries=[]
  for witness in bundle['branches']:
   condition=witness['condition'];assert sha(witness['result_file'])==witness['result_sha256'];result,summary=branch(folder/condition,condition,spec)
   assert read(folder/condition/'semantic_initial_receipt.json')['initial_diagnostic_identity']==bundle['initial_identity']
   if condition in('shadow_update','TTA'):assert result['feedback']['action_visible_fixed_parameters_and_buffers_unchanged']
   results[condition]=result;summaries.append(summary)
  full=results['M_event_full'];tta=results['TTA'];first=None
  for a,b in zip(full['queries'],tta['queries']):
   same=np.array_equal(arrays(a['actions_file'])['actions'],arrays(b['actions_file'])['actions']);same_query(a,b,actions=same,trace=same,actual_actions_only=True);checks+=1
   if not same:
    first=b['query'];assert b['event']['adaptation_applied']and any(u['accepted']and u['query']<=first for u in tta['feedback']['updates']);break
  if first is None:assert len(full['queries'])==len(tta['queries'])and full['success']==tta['success']
  for condition in('zero_update','shadow_update'):
   if condition not in results:continue
   z=results[condition];assert z['success']==full['success']and z['native_action_count']==full['native_action_count']and len(z['queries'])==len(full['queries'])
   for a,b in zip(full['queries'],z['queries'],strict=True):same_query(a,b);checks+=1
  records.append(dict(**accepted,full=full['success'],TTA=tta['success'],rescue=not full['success']and tta['success'],harm=full['success']and not tta['success'],first_changed_query=first,initial_identity=bundle['initial_identity'],branches=summaries))
 assert len(records)==8 and all(sum(r['task']==t for r in records)==4 for t in spec['tasks'])
 result=dict(passed=True,records=records,actual_native_episodes=sum(len(r['branches'])for r in records),same_prefix_query_checks=checks,all_native_receipts_source_phase_once_noise_reset_and_release_checked=True,qualification_sha256=sha(spec['trained_qualification']),screening_state=read(root/'screening_complete.json')['state'],formal_outcomes_never_used_for_training=True)
 write(root/'CPU_episode_audit.json',result);return result

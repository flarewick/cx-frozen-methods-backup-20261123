"""Original full native integrity plus actual packet/phase re-audit on CPU."""
from .common import read,write,verify,sha
from .index import index

def audit(spec,root):
 from native_visible_expand_c_v1.analysis import branch
 verify(spec);complete=read(root/'GPU_episodes_complete.json');assert complete['complete']and complete['protocol_identity']==spec['identity']and complete['frozen_parameter_pointers_and_mutation_counters_unchanged']
 records=[]
 for bundle in complete['cases']:
  assert bundle['initialized']and len(bundle['branches'])==1 and bundle['branches'][0]['condition']=='M_event_full'
  folder=root/'cases'/bundle['case_id']/'M_event_full';result,summary=branch(folder,'M_event_full',spec)
  value=index(folder,spec['event_config']);write(folder/'offline_feedback_index.json',value)
  assert not result['feedback']['updates']and not any(q['event']['adaptation_applied']for q in result['queries'])
  semantic=read(folder/'semantic_initial_receipt.json');records.append(dict(case_id=bundle['case_id'],queries=len(result['queries']),semantic_available=semantic['available'],actual_feedback_packets=len(value['packets']),potential_actions=len(value['potential_actions']),success_diagnostic_only=result['success'],expert_labels_available=False))
 assert len(records)==spec['cases_per_batch']==25
 out=dict(passed=True,scope='Frozen full actual TRAIN state capture; no expert labels yet',actual_native_episodes=25,records=records,optimizer_steps=0,no_expert_labels_online=True,no_official_EVAL=True)
 from native_onpolicy_mixed_teacher_v3.stream import audit as recovery_audit
 out['recovery_stream_audit']=recovery_audit(spec,root)
 from .future_preparation import audit as future_audit
 out['required_next_stage_preparation']=future_audit(root,spec)
 from .closed_audit import audit as audit_closed
 out['closed_expert_task_lifetimes']=audit_closed(root,spec)
 write(root/'CPU_full_capture_audit.json',out);write(root/'CPU_episode_audit.json',out);return out

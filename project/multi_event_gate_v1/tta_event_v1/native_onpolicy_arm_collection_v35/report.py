from .common import read,sha

def summarize(root,spec):
 assert read(root/'terminal_audit.json')['passed'];audit=read(root/'CPU_full_capture_audit.json');groups={};probes=[]
 for task in spec['tasks']:
  rows=[r for r in audit['records']if r['case_id'].startswith(task+':')];assert len(rows)==5
  groups[task]=dict(scenes=5,semantic_available=sum(r['semantic_available']for r in rows),causal_packets=sum(r['actual_feedback_packets']for r in rows),potential_actions=sum(r['potential_actions']for r in rows),expert_labels_available=0)
  cases=sorted([read(f)for f in(root/'accepted').glob(task+':*.json')],key=lambda r:r['episode_index'])
  for case in cases:
   folder=root/'cases'/case['case_id']/'M_event_full';index=read(folder/'offline_feedback_index.json')
   if index['potential_actions']:
    probes.append(dict(task=task,case=case,full_folder=str(folder),selection='first scene in fixed expert-valid order with a legal feedback-after action; no success filtering',**index['potential_actions'][0]));break
 return dict(complete=True,dataset_integrity_passed=True,actual_native_expert_episodes=25,actual_native_full_episodes=25,groups=groups,teacher_probe_states=probes,all_actual_GPU_child_parent_and_CPU_exits_and_resource_audits=True,coverage_is_potential_only=True,coverage_passed=False,calibration_training_started=False,optimizer_steps=0,full_FIT_VAL_still_required=True,input_sha256={str(f):sha(f)for f in(root/'terminal_audit.json',root/'CPU_full_capture_audit.json',root/'protocol.json')})

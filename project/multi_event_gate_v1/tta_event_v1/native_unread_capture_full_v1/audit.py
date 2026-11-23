"""Whole original references are software controls, never expert training labels."""
from native_policy_execution_v1.common import read,write,sha
from native_onpolicy_recovery_v3.common import verify

def compare(folder,spec):
 from native_q0_task_handoff_v3.control import compare as original_compare
 from native_visible_expand_c_v1.analysis import arrays,equal
 original_compare(folder,spec)
 left=read(folder/'M_event_full/result.json');right=read(folder/'handoff_reference/result.json')
 assert len(left['queries'])==len(right['queries'])>0
 proof=folder/'q0_handoff_whole_reference_audit.json';bindings={str(proof):sha(proof)}
 for a,b in zip(left['queries'],right['queries'],strict=True):
  assert a['query']==b['query']
  assert a['feedback_source_admission']==b['feedback_source_admission']
  name='execution%03d_arrivals.npz'%a['query']
  x,y=folder/'M_event_full'/name,folder/'handoff_reference'/name
  receipts=[folder/branch/('execution%03d.json'%a['query'])for branch in ['M_event_full','handoff_reference']]
  declared=[]
  for image,receipt in zip([x,y],receipts,strict=True):
   value=read(receipt);present='dense_file'in value
   assert present==image.exists(),('Declared arrived feedback file missing',folder.name,name)
   if present:assert sha(image)==value['dense_sha256'],'Arrived feedback file hash mismatch'
   declared.append(present);bindings[str(receipt)]=sha(receipt)
  assert declared[0]==declared[1],('Arrived feedback sampling plan changed',folder.name,name)
  if declared[0]:
   assert equal(arrays(x),arrays(y)),('Arrived feedback RGB/clock/state mismatch',folder.name,name)
   bindings.update({str(p):sha(p)for p in [x,y]})
 write(folder/'unread_capture_pair_audit.json',dict(passed=True,whole_original_control_passed=True,
  all_arrived_feedback_RGB_clocks_commands_byte_exact=True,queries=len(left['queries']),
  engineering_only_excluded_from_training=True,input_sha256=bindings))

def record_usage(task,folder,branch,result):
 enabled=bool(getattr(task,'_cx_defer_unread_capture',False))
 count=int(getattr(task,'_cx_deferred_capture_count',0))
 assert enabled==(branch=='M_event_full')and count>=0
 assert enabled or count==0
 write(folder/'unread_capture_usage.json',dict(enabled=enabled,skipped_unread_captures=count,
  original_actual_observation_reads_and_updates_retained=True,
  native_actions=result['native_action_count'],reference_excluded_from_dataset=not enabled))

def audit(root,spec):
 assert spec['unread_capture_whole_reference_each_case']
 assert not spec.get('scientific_qualification_forbidden',False)
 completed=read(root/'GPU_episodes_complete.json');assert completed['complete']
 cases=completed['cases'];assert len(cases)==spec['cases_per_batch']==25
 assert len({c['case_id']for c in cases})==25
 bindings={str(root/'GPU_episodes_complete.json'):sha(root/'GPU_episodes_complete.json')}
 skipped=0
 for item in cases:
  folder=root/'cases'/item['case_id']
  # Verify the runtime's original full-horizon proof and every bound input.
  proof=folder/'q0_handoff_whole_reference_audit.json';value=read(proof)
  assert value['passed']and value['whole_native_episode_exact']and value['engineering_only_excluded_from_training']
  assert value['whole_observations_policy_noise_actions_physics_RNG_and_native_traces_exact']
  verify(value);bindings[str(proof)]=sha(proof);bindings.update(value['input_sha256'])
  pair=folder/'unread_capture_pair_audit.json';pv=read(pair)
  assert pv['passed']and pv['whole_original_control_passed']and pv['all_arrived_feedback_RGB_clocks_commands_byte_exact']
  assert pv['engineering_only_excluded_from_training']and pv['queries']>0
  verify(pv);bindings[str(pair)]=sha(pair);bindings.update(pv['input_sha256'])
  for branch,enabled in [('M_event_full',True),('handoff_reference',False)]:
   usage=folder/branch/'unread_capture_usage.json';row=read(usage)
   result=read(folder/branch/'result.json')
   assert row['enabled']is enabled and row['skipped_unread_captures']>=0
   assert row['original_actual_observation_reads_and_updates_retained']
   assert row['native_actions']==result['native_action_count']
   assert row['reference_excluded_from_dataset']is(not enabled)
   if enabled:skipped+=row['skipped_unread_captures']
   else:assert row['skipped_unread_captures']==0
   bindings[str(usage)]=sha(usage)
 assert skipped>0,'No actual omitted unread picture captures were exercised'
 value=dict(passed=True,whole_original_reference_pairs=25,all_whole_observations_actions_noise_physics_RNG_exact=True,
  original_reference_never_elides_captures=True,reference_excluded_from_labels=True,
  skipped_unread_captures=skipped,input_sha256=bindings)
 write(root/'CPU_unread_capture_whole_reference_audit.json',value)
 return value

from native_policy_execution_v1.common import read,write,sha
from native_onpolicy_recovery_v3.common import verify
def compare(folder,spec):
 from native_unread_capture_full_v1.audit import compare as original
 original(folder,spec);files=[];sources=0;frames=0
 left=folder/'M_event_full';right=folder/'handoff_reference'
 ll=sorted(left.glob('tracking_equivalence_source*.json'));rr=sorted(right.glob('tracking_equivalence_source*.json'))
 assert [p.name for p in ll]==[p.name for p in rr]
 assert len(ll)==read(left/'result.json')['completed_sources']==read(right/'result.json')['completed_sources']
 for a,b in zip(ll,rr,strict=True):
  x,y=read(a),read(b);assert x==y,('Mask/presence/raw-logit/clock mismatch',folder.name,a.name)
  assert x['no_extra_model_forward']and len(x['rows'])==x['actual_tracking_calls'];sources+=1;frames+=len(x['rows']);files.extend([a,b])
 write(folder/'CPU_storage_whole_pair_audit.json',dict(passed=True,whole_original_trajectories_and_arrivals_passed=True,
  all_actual_source_rows_mask_presence_raw_logits_exact=True,sources=sources,frames=frames,
  input_sha256={str(p):sha(p)for p in [*files,folder/'unread_capture_pair_audit.json']}))
def audit(root,spec):
 assert spec['object_state_storage_CPU_whole_original_raw_controls']and spec['simulation_lanes']==10
 complete=read(root/'GPU_episodes_complete.json');assert complete['complete']and len(complete['cases'])==25
 bindings={};sources=frames=0
 for case in complete['cases']:
  p=root/'cases'/case['case_id']/'CPU_storage_whole_pair_audit.json';v=read(p);verify(v)
  assert v['passed']and v['whole_original_trajectories_and_arrivals_passed']and v['all_actual_source_rows_mask_presence_raw_logits_exact']
  sources+=v['sources'];frames+=v['frames'];bindings[str(p)]=sha(p);bindings.update(v['input_sha256'])
 assert sources>0 and frames>0
 value=dict(passed=True,whole_pairs=25,sources=sources,frames=frames,all_raw_feedback_and_original_trajectory_arrays_exact=True,
  no_extra_queries_or_frames=True,input_sha256=bindings)
 write(root/'CPU_storage_whole25_audit.json',value);return value

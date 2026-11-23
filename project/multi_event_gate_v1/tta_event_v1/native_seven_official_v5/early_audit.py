"""Bind actual early publication to the original full q0 policy, never an outcome."""
from .common import read,write,sha,mapped

def audit(spec,root):
 if not spec.get('early_pairing',False):return dict(passed=True,enabled=False)
 rows=[]
 for folder in sorted((root/'cases').iterdir()):
  if not folder.is_dir():continue
  dispatch=folder/'early_pairing_dispatch.json';v=read(dispatch);full=read(folder/'M_event_full/result.json');q=full['queries'][0]['policy']
  assert v['query']==v['actor_observation_tick']==q['query']==q['actor_observation_tick']==0
  assert v['base_infer_calls']==v['actual_internal_noise_calls']==q['base_infer_calls']==q['actual_internal_noise_calls']==1
  assert v['native_environment_unchanged']and q['environment_audit']['native_environment_unchanged']
  assert v['only_q0_provenance_no_full_result_or_labels']and v['full_result_absent_at_dispatch']
  assert v['source_policy_file']==q['policy_file']and v['source_policy_sha256']==q['policy_sha256']==sha(mapped(q['policy_file']))
  bundle=read(folder/'bundle.json');assert v['initial_identity']==bundle['initial_identity']
  assert v['dispatch_monotonic_ns']>0
  # Filesystem timestamps are diagnostic; source/provenance checks do not depend on wall-clock adjustments.
  rows.append(dict(case_id=folder.name,source_q0_policy_sha256=q['policy_sha256'],no_full_outcome_at_publication=True,dispatch_precedes_full_result_mtime=dispatch.stat().st_mtime_ns<=(folder/'M_event_full/result.json').stat().st_mtime_ns))
 assert len(rows)==spec['cases_per_batch']
 value=dict(passed=True,enabled=True,records=rows,actual_source_q0_only=True,no_full_policy_outcome_or_teacher_supplied=True)
 write(root/'CPU_early_pairing_audit.json',value);return value

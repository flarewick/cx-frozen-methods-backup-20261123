"""Whole actual native fresh-setup reference; never count reference as TRAIN scene."""
from native_visible_expand_c_v1.analysis import branch,same_query
from native_policy_execution_v1.common import read,write,sha

def compare(folder,spec):
 a,sa=branch(folder/'M_event_full','M_event_full',spec)
 b,sb=branch(folder/'handoff_reference','M_event_full',spec)
 for key in('success','reason','native_action_count','physical_ticks','original_native_step_limit'):
  assert a[key]==b[key],('Handoff whole terminal difference',key,a[key],b[key])
 assert len(a['queries'])==len(b['queries'])
 for left,right in zip(a['queries'],b['queries']):same_query(left,right,actual_actions_only=True)
 write(folder/'q0_handoff_whole_reference_audit.json',dict(passed=True,whole_native_episode_exact=True,actual_queries=len(a['queries']),whole_observations_policy_noise_actions_physics_RNG_and_native_traces_exact=True,engineering_only_excluded_from_training=True,input_sha256={str(p):sha(p)for p in(folder/'M_event_full/result.json',folder/'handoff_reference/result.json')}))

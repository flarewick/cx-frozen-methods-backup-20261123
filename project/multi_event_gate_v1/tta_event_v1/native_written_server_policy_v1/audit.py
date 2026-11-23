"""Independent acceptance of the documented/user34 resource policy, no extra36 gate."""
import json,math
from native_policy_execution_v1.common import T,read,write,sha
RULE='/home/qiufw/服务器使用注意事项.md'
def validate(rows):
 steady=0
 for row in rows:
  if 'stopped'in row:raise AssertionError('Actual stopped workload never qualifies')
  if row.get('seconds',0)<280 or not row.get('gpu_pids'):continue
  steady+=1
  assert row['account_gpu_count']<=3
  assert row['time_weighted_mean_utilization']>=34 and row['rolling_300s_utilization']>=34
  assert 4096<=row['memory_mib']<=24563
  for pid in row['gpu_pids']:
   def val(key):
    table=row[key];return table.get(pid,table.get(str(pid)))
   for key in ['process_mean_utilization','process_rolling_300s_utilization']:
    x=val(key);assert x is not None and math.isfinite(x)and x>=34
   assert 4096<=val('process_memory_mib')<=24563
   assert 40<=val('cpu')<=1600 and 0<val('rss_bytes')<=128*1024**3
 assert steady>0
 assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']
 return steady

def audit(root,spec):
 assert spec['documented_server_policy_user34_without_extra35_36']and not spec['governor_extra_margin_required']
 file=root/'attempt/resources.jsonl';profile=root/'owner_profile_runtime.json';pv=read(profile);assert pv['passed']and pv['closed']and pv['host_wall_time_only_NOT_GPU_kernel_time']and sum(r['calls']for r in pv['rows'])>0
 assert all(0<=r['exclusive_host_ns']<=r['inclusive_host_ns']for r in pv['rows'])
 rows=[json.loads(s)for s in file.read_text().splitlines()];steady=validate(rows)
 value=dict(passed=True,original_strict34_cumulative_rolling300_and_all_caps=True,unverified_extra35_36_NOT_enforced=True,steady_samples=steady,manager_algorithm_equivalence_NOT_claimed=True,actual_signal_handling_unchanged=True,input_sha256={str(p):sha(p)for p in[file,profile,root/'protocol.json']},source_sha256={str(p):sha(p)for p in[T/'native_process_missing_guard_v2/guard.py',T/'native_written_server_policy_v1/audit.py']})
 write(root/'CPU_written_policy_audit.json',value);return value

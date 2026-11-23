"""Actual failed-run resource replay, no CUDA or qualification."""
import json,copy
from pathlib import Path
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped
from .margin import Monitor

def run():
 reports=[];files=[]
 for version in [17,18]:
  base=T/f'outputs/native_onpolicy_arm_collection_v{version}/val_000';root=mapped(read(base/'execution_state.json')['root']);file=root/'attempt/resources.jsonl';rows=[json.loads(line)for line in file.read_text().splitlines()if line.strip()];monitor=Monitor();first=None;last=None
  for row in rows:
   if 'seconds'not in row:continue
   reason,metrics=monitor.observe(row);last=metrics
   if reason and first is None:first=dict(seconds=row['seconds'],reason=reason,metrics=metrics)
  assert first is not None
  assert first['seconds']<300
  reports.append(dict(version=version,first_extra_stop=first,last_metrics=last));files.append(file)
 # A truly low recent window cannot be hidden by a long positive history.
 m=Monitor();latest=None
 for sec in range(1,501):
  util=60 if sec<=320 else 30
  row=dict(seconds=sec,gpu_pids=[7],process_utilization={'7':util},process_mean_utilization={'7':50},mean_utilization=50,time_weighted_mean_utilization=50,rolling_300s_utilization=50,recent_60s_utilization=50)
  latest=m.observe(row)
 assert latest[0]and latest[1]['process_rolling180']['7']<36
 return dict(passed=True,CPU_only=True,historical_failed_runs_remain_failed=True,real_replay=reports,short_window_cannot_hide_in_cumulative=True,no_new_GPU_or_NVML_queries=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()},input_sha256={str(p):sha(p)for p in files})
if __name__=='__main__':
 v=run();write(T/'outputs/native_governor_margin_v2/CPU_checks_v1.json',v);print('GOVERNOR_EXTRA_MARGIN_CPU_PASS')

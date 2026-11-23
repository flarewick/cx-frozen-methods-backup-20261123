"""Recompute every original process window; missing intervals cannot inflate it."""
from pathlib import Path
import json,sys,math
from native_policy_execution_v1.common import T,read,write,sha
from native_process_sampler_v13.metrics import select
from .empty import admissible_missing,zero_previous,eligible_row
sys.path.insert(0,str(T.parents[1]/'deployment'))
from gpu_utilization_window import UtilizationWindow

def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-9),(a,b)
def audit(root,spec):
 assert spec['conservative_idle_missing_lower_bound_guard']
 f=root/'attempt/resources.jsonl';rows=[json.loads(x)for x in f.read_text().splitlines()];assert not any('stopped'in r for r in rows)
 window=UtilizationWindow();recent=UtilizationWindow(60);sums={};count=0;gaps=0;steady=0
 for row in rows:
  if 'seconds'not in row:continue
  count+=1;active=row['gpu_pids'];samples=row['process_metric_samples'];values={int(k):v for k,v in row['process_utilization'].items()if v is not None}
  first,_=select(samples[0]['raw_NVML'],samples[0]['gpu']);missing=[pid for pid in active if pid not in first];expected=dict(first);unresolved=set(missing)
  for sample in samples[1:]:
   extra,_=select(sample['raw_NVML'],sample['gpu'])
   for pid in list(unresolved):
    if pid in extra:expected[pid]=0;unresolved.remove(pid)
  assert {pid:expected.get(pid)for pid in active}=={pid:row['process_utilization'][str(pid)]for pid in active}
  permitted=admissible_missing(active,values,samples,spec['required_gpu_uuid'],samples[-1]['gpu']);assert {str(k):v for k,v in permitted.items()}==row['missing_process_zero_lower_bound']
  before=zero_previous(window,recent,permitted);assert before==row['zeroed_missing_interval'];gaps+=bool(permitted)
  for pid in active:
   if pid in values:sums[pid]=sums.get(pid,0)+values[pid]
  elapsed=row['lower_bound_exact_elapsed'];cum,rolling=window.add(elapsed,0,{pid:values.get(pid,0)for pid in active});_,rr=recent.add(elapsed,0,{pid:values.get(pid,0)for pid in active})
  for pid in active:
   k=str(pid);close(min(sums.get(pid,0)/count,cum.get(pid,0)),row['process_mean_utilization'][k]);close(rolling.get(pid,0),row['process_rolling_300s_utilization'][k]);close(rr.get(pid,0),row['process_recent_60s_utilization'][k])
   if elapsed>=280:
    assert eligible_row(row,k)and row['process_mean_utilization'][k]>=34 and row['process_rolling_300s_utilization'][k]>=34
  if elapsed>=280 and active:steady+=1
 assert count and steady and rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']
 out=dict(passed=True,actual_samples=count,steady_samples=steady,accepted_fresh_empty_intervals=gaps,all_process_windows_recomputed_exact=True,unknown_interval_integrated_at_zero_not_prior_positive=True,instantaneous_missing_values_preserved=True,all_original_average_floors_34=True,no_device_to_process_substitution=True,all_transport_and_clock_validation_original=True,input_sha256={str(f):sha(f),str(root/'protocol.json'):sha(root/'protocol.json')})
 write(root/'CPU_idle_missing_lower_bound_audit.json',out);return out

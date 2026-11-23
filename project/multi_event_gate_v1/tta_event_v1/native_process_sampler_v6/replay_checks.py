"""Exact terminal reconstruction and conservative final-backfill negatives; CPU only."""
import copy,json,time,hashlib
from pathlib import Path
from .device_window import DeviceWindow
from .device_audit import audit
from .device_checks import packet,E,UUID
from . import metrics
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent;O=T/'outputs/native_process_sampler_v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows_fixture(value=50):
 w=DeviceWindow(E,UUID);rows=[]
 for end in range(2,302,2):
  p=packet(end,value);now=E+end*1000000+100000;meta=w.update(p,now,end+.1,steady_required=end>=280)
  rows.append(dict(seconds=end+.1,gpu_pids=[123],device_continuous=meta,process_utilization={'123':50},process_metric_samples=[dict(raw_NVML=p)],mean_utilization=meta['sample_mean'],time_weighted_mean_utilization=meta['cumulative_lower_bound'],rolling_300s_utilization=meta['rolling300_lower_bound']))
 return rows

def main():
 rows=rows_fixture();post=dict(collected_unix_us=E+302100000,raw_NVML=packet(302));result=audit(rows,post,UUID,E*1000+1);assert result['steady_checkpoints_recomputed_from_final_history']==11
 for mode in ['late_start','changed_snapshot','missing_raw','wrong_clock','missing_checkpoint']:
  bad=copy.deepcopy(rows);permit=E*1000+1
  if mode=='late_start':permit=E*1000-1
  if mode=='changed_snapshot':bad[0]['mean_utilization']+=1
  if mode=='missing_raw':del bad[0]['process_metric_samples']
  if mode=='wrong_clock':bad[2]['device_continuous']['started_unix_us']+=1
  if mode=='missing_checkpoint':bad[-1]['device_continuous']['prior_steady_checkpoints_rechecked']-=1
  try:audit(bad,post,UUID,permit)
  except(AssertionError,ValueError,KeyError):pass
  else:raise AssertionError(mode+' accepted')
 # A final raw metadata backfill may invalidate a check that was formerly>=34.
 bad=rows_fixture(35);post_bad=packet(302,35)
 for values in post_bad['frames'][0]['device_samples'].values():values.extend(dict(timestamp_us=E+t+50000,value=0)for t in range(288000000,300000000,200000))
 try:audit(bad,dict(collected_unix_us=E+302100000,raw_NVML=post_bad),UUID,E*1000+1)
 except ValueError:pass
 else:raise AssertionError('Final backfill incorrectly promoted to qualification')
 # Reconstruct actual passive observations from all raw records in the new worker.
 start=time.time_ns()//1000;begin=time.monotonic();w=None;actual=[]
 for i in range(6):
  metrics.process_utilization(6);raw=metrics.diagnostics()['raw_NVML'];w=w or DeviceWindow(start,raw['gpu_uuid']);now=time.time_ns()//1000;meta=w.update(raw,now,time.monotonic()-begin)
  actual.append(dict(seconds=(now-start)/1e6,gpu_pids=[],device_continuous=meta,process_utilization={},process_metric_samples=[dict(raw_NVML=raw)],mean_utilization=meta['sample_mean'],time_weighted_mean_utilization=meta['cumulative_lower_bound'],rolling_300s_utilization=meta['rolling300_lower_bound']))
  print('READ_ONLY_DEVICE_REPLAY',i,flush=True)
 metrics.process_utilization(6);post_actual=dict(collected_unix_us=time.time_ns()//1000,raw_NVML=metrics.diagnostics()['raw_NVML']);replayed=audit(actual,post_actual,w.uuid,start*1000+1);assert replayed['all_device_snapshots_replayed_exactly']
 out=dict(passed=True,CPU_fixture_only=True,no_GPU_work_started=True,synthetic_steady_reconstruction=result,late_start_missing_raw_changed_snapshot_clock_checkpoint_rejected=True,post_release_backfill_can_invalidate_never_resurrect_failed_jobs=True,actual_raw_observations=actual,actual_post_release_fixture=post_actual,actual_read_only_reconstruction=replayed,source_sha256={str(p):sha(p)for p in C.glob('*')if p.suffix in('.py','.json')})
 with(O/'CPU_replay_checks.json').open('x')as f:json.dump(out,f,indent=2)
 print('FULL_DEVICE_TERMINAL_REPLAY_CPU_PASS',flush=True)
if __name__=='__main__':main()

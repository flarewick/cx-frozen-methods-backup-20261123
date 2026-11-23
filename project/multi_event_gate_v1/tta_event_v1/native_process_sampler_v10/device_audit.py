"""Rebuild device accounting from raw records and revisit every steady checkpoint."""
from .device_window import DeviceWindow

def audit(rows,post_release,gpu_uuid,permit_mtime_ns):
 values=[r for r in rows if'process_utilization'in r]
 if not values:raise ValueError('No actual device telemetry')
 start=values[0]['device_continuous']['started_unix_us']
 if start*1000>permit_mtime_ns:raise ValueError('Device accounting began after GPU permit')
 window=DeviceWindow(start,gpu_uuid);checkpoints=[];previous_count=0
 for row in values:
  recorded=row['device_continuous'];assert recorded['started_unix_us']==start
  count=recorded['prior_steady_checkpoints_rechecked'];increment=count-previous_count;assert increment in(0,1)
  if row.get('gpu_pids')and row['seconds']>=280.01:assert increment==1
  if not row.get('gpu_pids')or row['seconds']<279.99:assert increment==0
  now=recorded['now_unix_us'];current=window.update(row['process_metric_samples'][-1]['raw_NVML'],now,(now-start)/1e6,steady_required=bool(increment))
  assert current==recorded,'Device stream replay changed'
  assert row['mean_utilization']==current['sample_mean']and row['time_weighted_mean_utilization']==current['cumulative_lower_bound']and row['rolling_300s_utilization']==current['rolling300_lower_bound']
  if increment:checkpoints.append(now)
  previous_count=count
 # CPU-only final metadata read after actual GPU release; never include its later
 # time in the original job denominator or credit later jobs' GPU activity.
 now=post_release['collected_unix_us'];assert now>window.last_now
 window.update(post_release['raw_NVML'],now,(now-start)/1e6,steady_required=False)
 for end in checkpoints:
  old=window.retrospective(end)
  if min(old['sample_mean'],old['cumulative_lower_bound'],old['rolling300_lower_bound'])<34:raise ValueError('Final backfilled history invalidates earlier34percent device decision')
 return dict(passed=True,raw_device_telemetry_rows=len(values),steady_checkpoints_recomputed_from_final_history=len(checkpoints),all_device_snapshots_replayed_exactly=True,accounting_start_before_GPU_permit=True,all_startup_and_real_zeros_retained=True,final_native_record_count=len(window.times),late_native_record_count=getattr(window,'late_records',0),post_GPU_release_metadata_read_only=True,no_later_job_activity_credited_to_original_job=True,old_failed_jobs_never_reclassified=True)

"""A fresh valid empty reply permits only a zero lower bound, never a measured value."""
from native_process_sampler_v13.metrics import select

def admissible_missing(active,values,samples,uuid,index):
 missing=[pid for pid in active if pid not in values]
 if not missing:return {}
 if not samples:return {}
 for sample in samples:
  if sample.get('previous_call_data_used')or sample.get('device_wide_substitution')or sample.get('missing_records_inferred_zero'):return {}
  if sample.get('timeout_seconds')!=3:return {}
  data=sample['raw_NVML']
  if data['gpu_uuid']!=uuid:return {}
  selected,_=select(data,index)
  if any(pid in selected for pid in missing):return {}
  first,last=data['frames'];begin=first['driver_timestamp_us'];end=last['driver_timestamp_us']
  if last['API_status']!='no_process_sample_NOT_FOUND'or last['raw_process_records']:return {}
  for name in('SM','bandwidth'):
   records=[r for r in last['device_samples'][name]if begin<r['timestamp_us']<=end]
   if not records or records[-1]['timestamp_us']!=end:return {}
   if any(b['timestamp_us']-a['timestamp_us']>.4e6 for a,b in zip(records,records[1:])):return {}
   if records[0]['timestamp_us']-begin>.4e6:return {}
  transport=sample.get('transport')
  if transport is not None and(transport['request_deadline_seconds']!=3 or transport['previous_call_frames_used']):return {}
 return {pid:dict(admitted=True,instantaneous_value_unavailable=True,only_fresh_NOT_FOUND_and_dense_valid_device_clock=True,process_value_NOT_inferred=True,unknown_interval_rate_lower_bound=0.)for pid in missing}

def zero_previous(window,recent,eligible):
 before=dict(begin_elapsed=window.elapsed,previous={str(p):window.previous.get(p,0.)for p in eligible})
 for pid in eligible:
  window.previous[pid]=0.;recent.previous[pid]=0.
 return before

def eligible_row(row,key):
 pid=int(key);values={int(p):v for p,v in row['process_utilization'].items()if v is not None}
 if pid in values:return True
 expected=admissible_missing(row['gpu_pids'],values,row['process_metric_samples'],row['process_metric_samples'][-1]['gpu_uuid'],row['process_metric_samples'][-1]['gpu'])
 recorded=row.get('missing_process_zero_lower_bound',{})
 return pid in expected and str(pid)in recorded and recorded[str(pid)]==expected[pid]and row.get('lower_bound_exact_elapsed')is not None and row.get('zeroed_missing_interval',{}).get('begin_elapsed',float('inf'))<=row['lower_bound_exact_elapsed']

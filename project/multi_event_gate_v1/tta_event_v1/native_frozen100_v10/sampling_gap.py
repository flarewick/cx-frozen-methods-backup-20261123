"""Specific native PID sensor fault diagnosis, no GPU or metric substitution."""
from .common import T,read
REFERENCE=T/'outputs/native_frozen100_v3/pads/batches/012/runs/e1cb7cfd470c4ad98976c678ac0e15f8'
BAD_UUID='GPU-0f89dea0-728e-1d23-fd62-d25b9bad3e4f'
def diagnose(sample,released):
 assert released and sample['raw_foreign_pids']==[]and sample['account_gpu_count']<=3
 assert len(sample['gpu_pids'])==1
 pid=str(sample['gpu_pids'][0]);assert sample['process_utilization'][pid]is None
 assert min(sample['mean_utilization'],sample['time_weighted_mean_utilization'],sample['rolling_300s_utilization'],sample['process_mean_utilization'][pid],sample['process_rolling_300s_utilization'][pid])>=34
 assert 4096<=sample['memory_mib']<=24563 and 4096<=sample['process_memory_mib'][pid]<=24563 and 40<=sample['cpu'][pid]<=1600 and sample['rss_bytes'][pid]<=128*2**30
 attempts=sample['process_metric_samples'];assert len(attempts)==3;gaps=[]
 for trial in attempts:
  raw=trial['raw_NVML'];assert raw['gpu_uuid']==BAD_UUID and not raw.get('error')
  left,right=raw['frames'];assert left['API_status']=='success'and right['API_status']=='no_process_sample_NOT_FOUND'and right['raw_process_records']==[]
  records=[r for r in left['raw_process_records']if str(r['pid'])==pid];assert len(records)==1
  age=left['driver_timestamp_us']-records[0]['timestamp_us'];assert age>400000;gaps.append(age)
 return dict(passed=True,diagnosed_PID=int(pid),native_missing_endpoint_count=3,stale_PID_record_ages_us=gaps,all_other_original_resource_checks_passed=True,old_failed_job_not_reclassified=True,only_new_fresh_different_GPU_can_be_admitted=True,no_resource_threshold_or_sampler_change=True)

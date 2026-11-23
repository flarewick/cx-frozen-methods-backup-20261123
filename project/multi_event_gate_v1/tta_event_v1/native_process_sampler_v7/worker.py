"""Read-only newer NVML API preserves explicit zero-valued process records."""
import json,sys,time
import pynvml as n

def frame(h,last_seen):
 try:
  rows=n.nvmlDeviceGetProcessesUtilizationInfo(h,last_seen)
  raw=[dict(pid=int(s.pid),timestamp_us=int(s.timeStamp),SM=int(s.smUtil),bandwidth=int(s.memUtil))for s in rows]
  status='success'
 except n.NVMLError_NotFound:raw=[];status='no_process_sample_NOT_FOUND'
 kind,samples=n.nvmlDeviceGetSamples(h,n.NVML_GPU_UTILIZATION_SAMPLES,0)
 if kind!=n.NVML_VALUE_TYPE_UNSIGNED_INT or not samples:raise RuntimeError('Unsigned driver utilization clock unavailable')
 clock=max(s.timeStamp for s in samples)
 memory_kind,memory_samples=n.nvmlDeviceGetSamples(h,n.NVML_MEMORY_UTILIZATION_SAMPLES,0)
 if memory_kind!=n.NVML_VALUE_TYPE_UNSIGNED_INT or not memory_samples:raise RuntimeError('Unsigned bandwidth sampling unavailable')
 device_samples=dict(SM=[dict(timestamp_us=int(s.timeStamp),value=int(s.sampleValue.uiVal))for s in samples],bandwidth=[dict(timestamp_us=int(s.timeStamp),value=int(s.sampleValue.uiVal))for s in memory_samples])
 return dict(driver_timestamp_us=int(clock),host_timestamp_us=time.time_ns()//1000,raw_process_records=raw,API_status=status,device_samples=device_samples)

def clock(h):
 kind,samples=n.nvmlDeviceGetSamples(h,n.NVML_GPU_UTILIZATION_SAMPLES,0)
 if kind!=n.NVML_VALUE_TYPE_UNSIGNED_INT or not samples:raise RuntimeError('Unsigned driver utilization clock unavailable')
 return dict(driver_timestamp_us=int(max(s.timeStamp for s in samples)),host_timestamp_us=time.time_ns()//1000)

def clock_pair(read_frame,read_clock,sleep=time.sleep,monotonic=time.monotonic):
 # Readiness depends ONLY on native timestamps; no process values are consulted.
 began=monotonic();first=read_frame(0);observed=[first];clocks=[]
 while first['host_timestamp_us']-first['driver_timestamp_us']>.75e6:
  if monotonic()-began>=.6:raise ValueError('Initial driver clock remained stale')
  sleep(.025);first=read_frame(0);observed.append(first)
 if first['host_timestamp_us']<first['driver_timestamp_us']:raise ValueError('Future initial driver clock')
 sleep(.05);last_clock=read_clock();clocks.append(last_clock)
 while last_clock['driver_timestamp_us']-first['driver_timestamp_us']<.8e6 or last_clock['host_timestamp_us']-first['host_timestamp_us']<.8e6:
  if last_clock['host_timestamp_us']-first['host_timestamp_us']>=1.75e6:raise ValueError('Driver clock did not reach existing fresh interval')
  sleep(.05);last_clock=read_clock();clocks.append(last_clock)
 last=read_frame(first['driver_timestamp_us']);observed.append(last)
 # Original select checks BOTH lower and upper bounds; overshoot is never retried or waived.
 return [first,last],observed,clocks

def collect(index):
 n.nvmlInit()
 try:
  h=n.nvmlDeviceGetHandleByIndex(index);uuid=n.nvmlDeviceGetUUID(h)
  frames,observed,clocks=clock_pair(lambda last_seen:frame(h,last_seen),lambda:clock(h))
  return dict(gpu=index,gpu_uuid=uuid if isinstance(uuid,str)else uuid.decode(),frames=frames,clock_readiness_frames=observed,clock_only_readiness_observations=clocks,full_process_queries=len(observed),process_snapshot_polling_removed=True,API='nvmlDeviceGetProcessesUtilizationInfo_v1',device_values_used_only_as_clock=True,first_clock_eligible_endpoint_regardless_utilization=True)
 finally:n.nvmlShutdown()
if __name__=='__main__':print(json.dumps(collect(int(sys.argv[1])),allow_nan=False))

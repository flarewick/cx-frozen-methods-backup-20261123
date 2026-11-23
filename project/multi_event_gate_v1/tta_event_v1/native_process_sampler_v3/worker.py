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
 return dict(driver_timestamp_us=int(clock),host_timestamp_us=time.time_ns()//1000,raw_process_records=raw,API_status=status)

def clock_pair(read_frame,sleep=time.sleep,monotonic=time.monotonic):
 # Poll ONLY driver clock readiness; never choose an endpoint by utilization.
 began=monotonic();first=read_frame(0);observed=[first]
 while first['host_timestamp_us']-first['driver_timestamp_us']>.75e6:
  if monotonic()-began>=.6:raise ValueError('Initial driver clock remained stale')
  sleep(.025);first=read_frame(0);observed.append(first)
 if first['host_timestamp_us']<first['driver_timestamp_us']:raise ValueError('Future initial driver clock')
 sleep(1.05);last=read_frame(first['driver_timestamp_us']);observed.append(last)
 while last['driver_timestamp_us']-first['driver_timestamp_us']<.8e6:
  if last['host_timestamp_us']-first['host_timestamp_us']>=1.75e6:raise ValueError('Driver clock did not reach existing fresh interval')
  sleep(.025);last=read_frame(first['driver_timestamp_us']);observed.append(last)
 return [first,last],observed

def collect(index):
 n.nvmlInit()
 try:
  h=n.nvmlDeviceGetHandleByIndex(index);uuid=n.nvmlDeviceGetUUID(h)
  frames,observed=clock_pair(lambda last_seen:frame(h,last_seen))
  return dict(gpu=index,gpu_uuid=uuid if isinstance(uuid,str)else uuid.decode(),frames=frames,clock_readiness_frames=observed,API='nvmlDeviceGetProcessesUtilizationInfo_v1',device_values_used_only_as_clock=True,first_clock_eligible_endpoint_regardless_utilization=True)
 finally:n.nvmlShutdown()
if __name__=='__main__':print(json.dumps(collect(int(sys.argv[1])),allow_nan=False))

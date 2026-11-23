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

def collect(index):
 n.nvmlInit()
 try:
  h=n.nvmlDeviceGetHandleByIndex(index);uuid=n.nvmlDeviceGetUUID(h)
  first=frame(h,0);time.sleep(1.05);last=frame(h,first['driver_timestamp_us'])
  return dict(gpu=index,gpu_uuid=uuid if isinstance(uuid,str)else uuid.decode(),frames=[first,last],API='nvmlDeviceGetProcessesUtilizationInfo_v1',device_values_used_only_as_clock=True)
 finally:n.nvmlShutdown()
if __name__=='__main__':print(json.dumps(collect(int(sys.argv[1])),allow_nan=False))

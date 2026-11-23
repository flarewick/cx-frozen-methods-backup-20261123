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

def eligible(first,last):
 # Exact original time bounds; no process status/value can influence a pair.
 return(.8e6<=last['driver_timestamp_us']-first['driver_timestamp_us']<=1.6e6 and .8e6<=last['host_timestamp_us']-first['host_timestamp_us']<=1.8e6 and all(0<=f['host_timestamp_us']-f['driver_timestamp_us']<=.75e6 for f in(first,last)))

def choose_pair(observed,start=0):
 last=observed[-1]
 for index in range(start,len(observed)-1):
  if eligible(observed[index],last):return index,len(observed)-1
 return None

def clock_pair(read_frame,read_clock=None,sleep=time.sleep,monotonic=time.monotonic):
 began=monotonic();first=read_frame(0);observed=[first]
 while first['host_timestamp_us']-first['driver_timestamp_us']>.75e6:
  if monotonic()-began>=.6:raise ValueError('Initial driver clock remained stale')
  sleep(.025);first=read_frame(0);observed.append(first)
 if first['host_timestamp_us']<first['driver_timestamp_us']:raise ValueError('Future initial driver clock')
 start=len(observed)-1
 while True:
  # Fewer native calls than50ms clock-only polling; every retained endpoint is
  # a complete actual process/SM/BW frame, including explicit zero records.
  sleep(.2);last=read_frame(first['driver_timestamp_us']);observed.append(last)
  if not 0<=last['host_timestamp_us']-last['driver_timestamp_us']<=.75e6:raise ValueError('Final driver clock stale or future')
  pair=choose_pair(observed,start)
  if pair is not None:return [observed[pair[0]],observed[pair[1]]],observed,[]
  if last['host_timestamp_us']-first['host_timestamp_us']>=1.75e6:raise ValueError('No actual endpoint pair within original fresh interval')

def verify_trace(data):
 observed=data['clock_readiness_frames'];indices=data['selected_frame_indices']
 assert len(indices)==2 and indices[1]==len(observed)-1 and data['frames']==[observed[i]for i in indices]
 start=next(i for i,f in enumerate(observed)if 0<=f['host_timestamp_us']-f['driver_timestamp_us']<=.75e6)
 for end in range(start+1,len(observed)):
  pair=choose_pair(observed[:end+1],start)
  if pair is not None:
   assert list(pair)==indices and end==len(observed)-1,'Earlier eligible complete pair was skipped'
   return True
 raise ValueError('No temporally eligible actual pair in acquisition trace')

def collect(index):
 n.nvmlInit()
 try:
  h=n.nvmlDeviceGetHandleByIndex(index);uuid=n.nvmlDeviceGetUUID(h)
  frames,observed,clocks=clock_pair(lambda last_seen:frame(h,last_seen),lambda:clock(h))
  data=dict(gpu=index,gpu_uuid=uuid if isinstance(uuid,str)else uuid.decode(),frames=frames,clock_readiness_frames=observed,clock_only_readiness_observations=clocks,full_process_queries=len(observed),process_snapshot_polling_removed=False,full_frame_period_seconds=.2,first_eligible_real_frame_pair=True,all_intermediate_full_frames_retained=True,no_clock_only_frames_promoted=True,selected_frame_indices=[next(i for i,f in enumerate(observed)if f is value)for value in frames],API='nvmlDeviceGetProcessesUtilizationInfo_v1',device_values_used_only_as_clock=True,first_clock_eligible_endpoint_regardless_utilization=True)
  verify_trace(data);return data
 finally:n.nvmlShutdown()
if __name__=='__main__':print(json.dumps(collect(int(sys.argv[1])),allow_nan=False))

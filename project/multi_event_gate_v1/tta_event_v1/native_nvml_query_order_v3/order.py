"""Move existing device SM/BW reads after the potentially blocking PID API."""
import inspect,types
from native_process_sampler_v13 import worker as original

def frame_function():
 source=inspect.getsource(original.frame)
 block=" memory_kind,memory_samples=n.nvmlDeviceGetSamples(h,n.NVML_MEMORY_UTILIZATION_SAMPLES,0)\n if memory_kind!=n.NVML_VALUE_TYPE_UNSIGNED_INT or not memory_samples:raise RuntimeError('Unsigned bandwidth sampling unavailable')\n"
 marker=" clock=max(s.timeStamp for s in samples)\n"
 assert source.count(block)==source.count(marker)==1
 changed=source.replace(block,'').replace(marker,block+marker)
 assert changed.replace(block,'').replace(marker,block+marker)==changed
 namespace=dict(original.__dict__);exec(compile(changed,original.__file__,'exec'),namespace)
 return namespace['frame']


def clock_pair(read_frame,read_clock=None,sleep=original.time.sleep,monotonic=original.time.monotonic):
 """Acquire first eligible actual pair within the unchanged total2.6/outer3s."""
 began=monotonic();observed=[];first=None;start=None
 while True:
  value=read_frame(0 if first is None else first['driver_timestamp_us']);observed.append(value)
  age=value['host_timestamp_us']-value['driver_timestamp_us']
  if age<0:raise ValueError('Future actual driver clock')
  if first is None and age<=.75e6:first=value;start=len(observed)-1
  if first is not None and len(observed)>start+1:
   pair=original.choose_pair(observed,start)
   if pair is not None:return [observed[pair[0]],observed[pair[1]]],observed,[]
  if monotonic()-began>=2.6 or (observed[-1]['host_timestamp_us']-observed[0]['host_timestamp_us'])>=2.6e6:raise ValueError('No actual eligible endpoint pair within bounded acquisition budget')
  sleep(.025 if first is None or age>.75e6 or not original.device_time_fresh(value) else .2)

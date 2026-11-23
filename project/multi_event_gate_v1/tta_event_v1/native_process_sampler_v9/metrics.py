"""Only explicit fresh final-interval NVML process values, including real zeros."""
import copy,json,subprocess,sys,time
from pathlib import Path
_last={}

def select(data,index):
 if data['gpu']!=index or not data['gpu_uuid'].startswith('GPU-')or data['API']!='nvmlDeviceGetProcessesUtilizationInfo_v1':raise ValueError('Unexpected GPU/API identity')
 frames=data['frames']
 if len(frames)!=2:raise ValueError('Two driver-clock frames required')
 first,last=frames
 if not .8e6<=last['driver_timestamp_us']-first['driver_timestamp_us']<=1.6e6:raise ValueError('Driver clock did not advance one fresh interval')
 if not .8e6<=last['host_timestamp_us']-first['host_timestamp_us']<=1.8e6:raise ValueError('Host clock interval invalid')
 for f in frames:
  if not 0<=f['host_timestamp_us']-f['driver_timestamp_us']<=.75e6:raise ValueError('Driver sampling clock stale or future')
  if f['API_status']not in('success','no_process_sample_NOT_FOUND'):raise ValueError('Unexpected NVML status')
  if f['API_status']!='success'and f['raw_process_records']:raise ValueError('Error returned process records')
 values={};components=[]
 for r in last['raw_process_records']:
  pid=r['pid'];stamp=r['timestamp_us'];sm=r['SM'];bw=r['bandwidth']
  if type(pid)is not int or pid<=0 or pid in values:raise ValueError('Invalid or duplicate PID')
  if any(type(v)is not int or not 0<=v<=100 for v in(sm,bw)):raise ValueError('Unavailable/invalid utilization component')
  if stamp>last['host_timestamp_us']:raise ValueError('Process timestamp is in the future')
  if not first['driver_timestamp_us']<stamp or not 0<=last['driver_timestamp_us']-stamp<=.4e6:continue  # Remains unavailable; NEVER inferred zero
  values[pid]=max(sm,bw);components.append(dict(gpu=index,pid=pid,SM=sm,bandwidth=bw,value=max(sm,bw),timestamp_us=stamp,explicit_numeric_record=True))
 return values,components

def process_utilization(index):
 global _last
 if type(index)is not int or not 0<=index<=6:raise ValueError('Physical GPU index required')
 started=time.monotonic()
 try:r=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('worker.py')),str(index)],check=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=3)
 except subprocess.TimeoutExpired as exc:raise RuntimeError('Actual NVML3second deadline exceeded; stderr='+str(exc.stderr))from exc
 if r.returncode:raise RuntimeError('Actual NVML worker exit='+str(r.returncode)+'; stdout='+r.stdout+'; stderr='+r.stderr)
 data=json.loads(r.stdout)
 try:values,components=select(data,index)
 except ValueError as exc:raise ValueError(str(exc)+'; raw_NVML='+json.dumps(data,sort_keys=True))from exc
 _last=dict(unix_ns=time.time_ns(),gpu=index,gpu_uuid=data['gpu_uuid'],components=components,raw_NVML=data,collection_seconds=time.monotonic()-started,timeout_seconds=3,selected_frame_index=1,previous_call_data_used=False,device_wide_substitution=False,missing_records_inferred_zero=False,explicit_zero_records_preserved=True)
 return values

def diagnostics():return copy.deepcopy(_last)

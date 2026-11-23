"""Fresh original metadata fields; one NVML initialization, no CUDA."""
import json,os,sys,time
import pynvml as n
from native_process_sampler_v13.nvml_query import form

def collect(arguments):
 kind=form(arguments);assert n.nvmlDeviceGetCount()==7
 devices=[]
 for i in ([int(arguments[1])] if kind=='device' else range(7)):
  h=n.nvmlDeviceGetHandleByIndex(i);uuid=n.nvmlDeviceGetUUID(h)
  row=dict(index=i,uuid=uuid if isinstance(uuid,str) else uuid.decode('ascii'))
  if kind=='device':
   m=n.nvmlDeviceGetMemoryInfo(h,version=n.nvmlMemory_v2)
   row.update(memory_used_bytes=int(m.used),memory_total_bytes=int(m.total))
   if arguments[2]!='--query-gpu=memory.total':
    u=n.nvmlDeviceGetUtilizationRates(h);row.update(SM=int(u.gpu),bandwidth=int(u.memory))
  else:
   row['compute_processes']=[dict(pid=int(p.pid),used_GPU_memory_bytes=None if p.usedGpuMemory is None or p.usedGpuMemory==n.NVML_VALUE_NOT_AVAILABLE_ulonglong.value else int(p.usedGpuMemory)) for p in n.nvmlDeviceGetComputeRunningProcesses(h)]
  devices.append(row)
 return dict(arguments=arguments,devices=devices,read_only_NVML=True,CUDA_context_created=False,unix_ns=time.time_ns())

def main():
 n.nvmlInit()
 try:
  for line in sys.stdin:
   request=json.loads(line);assert set(request)=={'nonce','sequence','arguments'}
   began=time.time_ns();data=collect(request['arguments'])
   print(json.dumps(dict(request=request,data=data,worker_pid=os.getpid(),request_started_unix_ns=began,response_ready_unix_ns=time.time_ns()),allow_nan=False,separators=(',',':')),flush=True)
 finally:n.nvmlShutdown()
if __name__=='__main__':main()

"""Only the four original guard query forms; read-only NVML, no CUDA."""
import sys,json,time
import pynvml as n
SINGLE={'--query-gpu=memory.total','--query-gpu=utilization.gpu,utilization.memory,memory.used'}
APPS={'--query-compute-apps=gpu_uuid,pid','--query-compute-apps=gpu_uuid,pid,used_gpu_memory'}
def collect(arguments):
 assert type(arguments)is list and all(type(x)is str for x in arguments)
 single=len(arguments)==3 and arguments[0]=='-i'and arguments[2]in SINGLE
 assert single or(len(arguments)==1 and arguments[0]in APPS)
 if single:index=int(arguments[1]);assert str(index)==arguments[1]and 0<=index<7
 n.nvmlInit()
 try:
  assert n.nvmlDeviceGetCount()==7
  devices=[]
  for i in([index]if single else range(7)):
   h=n.nvmlDeviceGetHandleByIndex(i);uuid=n.nvmlDeviceGetUUID(h);uuid=uuid if isinstance(uuid,str)else uuid.decode('ascii');r=dict(index=i,uuid=uuid)
   if single:
    m=n.nvmlDeviceGetMemoryInfo(h,version=n.nvmlMemory_v2);r.update(memory_used_bytes=int(m.used),memory_total_bytes=int(m.total))
    if arguments[2]!='--query-gpu=memory.total':u=n.nvmlDeviceGetUtilizationRates(h);r.update(SM=int(u.gpu),bandwidth=int(u.memory))
   else:
    r['compute_processes']=[dict(pid=int(p.pid),used_GPU_memory_bytes=None if p.usedGpuMemory is None or p.usedGpuMemory==n.NVML_VALUE_NOT_AVAILABLE_ulonglong.value else int(p.usedGpuMemory))for p in n.nvmlDeviceGetComputeRunningProcesses(h)]
   devices.append(r)
  return dict(arguments=arguments,devices=devices,read_only_NVML=True,CUDA_context_created=False,unix_ns=time.time_ns())
 finally:n.nvmlShutdown()
if __name__=='__main__':print(json.dumps(collect(json.loads(sys.argv[1])),allow_nan=False))

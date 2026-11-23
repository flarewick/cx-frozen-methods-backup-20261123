import os,json
from pathlib import Path
import pynvml as n

def collect():
 n.nvmlInit()
 try:
  cards=[]
  for i in range(n.nvmlDeviceGetCount()):
   h=n.nvmlDeviceGetHandleByIndex(i);mem=n.nvmlDeviceGetMemoryInfo(h,version=n.nvmlMemory_v2);u=n.nvmlDeviceGetUtilizationRates(h);rows=n.nvmlDeviceGetComputeRunningProcesses(h)+n.nvmlDeviceGetGraphicsRunningProcesses(h);by={}
   for r in rows:
    pid=int(r.pid);value=getattr(r,'usedGpuMemory',None)
    if value is None or int(value)>int(mem.total):value=None
    else:value=int(value)
    if pid not in by or(by[pid]is None and value is not None)or(value is not None and by[pid]is not None and value>by[pid]):by[pid]=value
   own=[]
   for pid in by:
    try:
     if Path('/proc',str(pid)).stat().st_uid==os.getuid():own.append(pid)
    except FileNotFoundError:pass
   known=all(by[pid]is not None for pid in own);own_bytes=sum(by[pid]for pid in own)if known else None;uuid=n.nvmlDeviceGetUUID(h);uuid=uuid.decode()if isinstance(uuid,bytes)else uuid
   cards.append(dict(index=i,uuid=uuid,pids=sorted(by),own_pids=own,own_memory_bytes=own_bytes,total_bytes=int(mem.total),memory_known=known,own_fraction=None if own_bytes is None else own_bytes/int(mem.total),attributable_utilization=max(int(u.gpu),int(u.memory))if own and set(own)==set(by)else None,CUDA_context_created=False))
  return dict(cards=cards,own_gpu_count=sum(bool(c['own_pids'])for c in cards),read_only=True)
 finally:n.nvmlShutdown()
if __name__=='__main__':print(json.dumps(collect(),allow_nan=False))

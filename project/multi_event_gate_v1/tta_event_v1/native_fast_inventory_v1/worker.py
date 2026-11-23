"""Fresh read-only NVML inventory in a killable CPU child; no CUDA context."""
import json,os,time
from pathlib import Path
import pynvml as n

def collect():
 n.nvmlInit()
 try:
  count=n.nvmlDeviceGetCount()
  if count!=7:raise RuntimeError('All seven server GPUs must be visible')
  cards=[]
  for index in range(count):
   h=n.nvmlDeviceGetHandleByIndex(index);uuid=n.nvmlDeviceGetUUID(h);memory=n.nvmlDeviceGetMemoryInfo(h,version=n.nvmlMemory_v2);util=n.nvmlDeviceGetUtilizationRates(h)
   running=n.nvmlDeviceGetComputeRunningProcesses(h)+n.nvmlDeviceGetGraphicsRunningProcesses(h)
   pids=sorted({int(p.pid)for p in running});ours=[]
   for pid in pids:
    try:owner=Path('/proc',str(pid)).stat().st_uid==os.getuid()
    except FileNotFoundError:owner=False
    if owner:ours.append(pid)
   used=(int(memory.used)+2**20-1)//2**20;total=(int(memory.total)+2**20-1)//2**20;sm=int(util.gpu)
   if not isinstance(uuid,str):uuid=uuid.decode('ascii')
   cards.append(dict(index=index,uuid=uuid,memory_mib=used,total_mib=total,sm=sm,pids=pids,own_pids=ours,available=not pids and used<=64 and sm<=5))
  return dict(max_gpus=3,cards=cards,own_gpu_count=sum(bool(c['own_pids'])for c in cards),available_indices=[c['index']for c in cards if c['available']])
 finally:n.nvmlShutdown()
if __name__=='__main__':print(json.dumps(collect(),allow_nan=False))

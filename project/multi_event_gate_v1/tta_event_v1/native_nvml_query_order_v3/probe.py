"""Real read-only NVML under an explicit CPU-injected PID-call delay."""
import json,sys,time
from native_process_sampler_v13 import worker as w
from .order import frame_function,clock_pair
w.frame=frame_function()
from native_nvml_query_order_v2.order import clock_pair as old_pair
w.clock_pair=clock_pair if sys.argv[1]=='ordered'else old_pair
w.n.nvmlInit();handle=w.n.nvmlDeviceGetHandleByIndex(0);saved=w.n.nvmlDeviceGetSamples;calls=[0];injected=[False];bandwidth_calls=[0]
def delayed(handle,kind,*args,**kwargs):
 calls[0]+=1
 if kind==w.n.NVML_MEMORY_UTILIZATION_SAMPLES:bandwidth_calls[0]+=1
 if kind==w.n.NVML_MEMORY_UTILIZATION_SAMPLES and bandwidth_calls[0]==4 and not injected[0]:
  injected[0]=True;time.sleep(1.28)
 return saved(handle,kind,*args,**kwargs)
w.n.nvmlDeviceGetSamples=delayed
print('CPU_READY',flush=True);assert sys.stdin.readline().strip()=='run';start=time.monotonic()
try:
 data=w.collect(0,handle=handle);w.verify_trace(data);result=dict(passed=True,data=data)
except BaseException as error:result=dict(passed=False,error=str(error))
finally:w.n.nvmlShutdown()
result.update(mode=sys.argv[1],seconds=time.monotonic()-start,actual_full_PID_calls=calls[0],CPU_injected_single_call_delay_seconds=1.28,read_only_no_CUDA_or_scientific_qualification=True)
print(json.dumps(result),flush=True)

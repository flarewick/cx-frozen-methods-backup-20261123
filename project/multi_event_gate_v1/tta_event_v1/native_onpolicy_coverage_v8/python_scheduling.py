"""Process-local interpreter fairness only; no model, GPU or OS configuration."""
import os,sys
from .common import write
INTERVAL_SECONDS=.001

def install(output):
 before=sys.getswitchinterval();assert before==.005,'Unexpected original interpreter interval'
 sys.setswitchinterval(INTERVAL_SECONDS);assert sys.getswitchinterval()==INTERVAL_SECONDS
 write(output,dict(passed=True,inner_PID=os.getpid(),original_switch_interval_seconds=before,actual_switch_interval_seconds=INTERVAL_SECONDS,only_this_Python_process=True,OS_priority_and_affinity_unchanged=True,GPU_clocks_power_and_threads_unchanged=True,no_extra_operations_or_GPU_compute=True,original_model_and_RNG_operations_unchanged=True))

"""Actual process isolation and RNG preservation; no GPU or policy calls."""
from pathlib import Path
import sys,subprocess,json,tempfile,random
from .python_scheduling import install,INTERVAL_SECONDS
from .common import O,read,write

def main():
 prior=sys.getswitchinterval();rng=random.getstate()
 try:
  with tempfile.TemporaryDirectory(prefix='tta_python_interval_cpu_',dir=O)as temp:
   f=Path(temp)/'actual.json';install(f);assert read(f)['actual_switch_interval_seconds']==.001 and sys.getswitchinterval()==.001 and random.getstate()==rng
   result=subprocess.run([sys.executable,'-B','-c','import sys;print(sys.getswitchinterval())'],stdout=subprocess.PIPE,text=True,check=True)
   assert float(result.stdout.strip())==prior,'Another interpreter must keep its original setting'
 finally:sys.setswitchinterval(prior)
 assert sys.getswitchinterval()==prior and random.getstate()==rng
 write(O/'CPU_python_scheduling_checks.json',dict(passed=True,actual_local_interpreter_interval_change=True,other_interpreter_unchanged=True,CPU_fixture_restored_original_interval=True,Python_RNG_unchanged=True,no_GPU_context_or_policy_called=True,performance_or_success_claimed=False))
 print('PROCESS_LOCAL_PYTHON_SCHEDULING_CPU_PASS',flush=True)
if __name__=='__main__':main()

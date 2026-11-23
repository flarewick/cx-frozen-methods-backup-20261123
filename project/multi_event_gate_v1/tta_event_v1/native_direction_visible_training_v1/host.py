"""CPU-only fixed-budget host observer; finalization follows actual wait return."""
from pathlib import Path
import sys,subprocess,uuid,time,os
from .common import *
def main():
 assert not list(O.glob('host_run_*'))and not(O/'runs').exists(),'Prior selected-target fit exists; inspect it'
 folder=O/('host_run_'+uuid.uuid4().hex);folder.mkdir(parents=True)
 command=[sys.executable,'-B',str(T/'native_selected_runtime_v2/cpu_entry.py'),'--','-m','native_direction_visible_training_v1.train','--run'];began=time.monotonic()
 with (folder/'train.log').open('x')as log:
  p=subprocess.Popen(command,cwd=T,stdout=log,stderr=subprocess.STDOUT)
  ticks=int(Path('/proc/%d/stat'%p.pid).read_text().split(') ',1)[1].split()[19])
  write(folder/'launch.json',dict(host_pid=p.pid,host_started_ticks=ticks,observer_pid=os.getpid(),command=command,CUDA_permitted=False))
  print('SELECTED_OUTER_TRAINING_CPU_HOST_STARTED',p.pid,folder,flush=True);code=p.wait()
 write(folder/'exit.json',dict(exit_code=code,actual_Popen_wait_return=True,host_pid=p.pid,host_started_ticks=ticks,seconds=time.monotonic()-began))
 if code!=0:print('SELECTED_OUTER_TRAINING_FAILED_ACTUAL_EXIT',code,flush=True);raise SystemExit(code)
 canonical=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')/folder.relative_to(T.parents[1])
 final=[sys.executable,'-B',str(T/'native_selected_runtime_v2/cpu_entry.py'),'--','-m','native_direction_visible_training_v1.finalize','--host-run',str(canonical)]
 code=subprocess.run(final,cwd=T).returncode;write(folder/'finalize_exit.json',dict(exit_code=code,actual_subprocess_wait_return=True));raise SystemExit(code)
if __name__=='__main__':main()

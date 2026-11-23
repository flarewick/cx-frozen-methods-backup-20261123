"""Actual CPU container exit recording; no CUDA devices made visible."""
import os,subprocess,sys
from pathlib import Path
from native_policy_execution_v1.common import T,write
P=T.parents[1];sys.path.insert(0,str(P/'deployment'))
import user_container as container

def main():
 canonical='/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA/multi_event_gate_v1/tta_event_v1'
 command=container.command(['bash',canonical+'/native_onpolicy_collection_v17/scripts/cpu.sh','-m','native_q0_task_handoff_v3.checks'],mount_project=True)
 out=T/'outputs/native_q0_task_handoff_v3';out.mkdir(parents=True,exist_ok=True)
 with(out/'CPU_actual_checks_v2.log').open('x')as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
 write(out/'CPU_actual_host_exit_v2.json',dict(exit_code=code,actual_Popen_wait_return=True,pid=p.pid,proc_absent_after_wait=not Path('/proc',str(p.pid)).exists(),CUDA_hidden=True))
 raise SystemExit(code)
if __name__=='__main__':main()

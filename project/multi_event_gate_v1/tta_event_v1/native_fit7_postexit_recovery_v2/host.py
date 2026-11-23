"""Actual waits for post-fit-only recovery, never rerun training."""
import subprocess,sys
from pathlib import Path
from native_policy_execution_v1.common import T,write,sha
O=T/'outputs/native_fit7_postexit_recovery_v2'
def main():
 assert not O.exists();O.mkdir(parents=True)
 P=T.parents[1];sys.path.insert(0,str(P/'deployment'));import user_container
 script=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')/(T/'native_onpolicy_collection_v21/scripts/cpu.sh').relative_to(P)
 write(O/'protocol.json',dict(postfit_checks_only=True,no_optimizer_or_new_training_steps=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')}))
 for phase,args in (('recovery',[]),('finalize',['--finalize'])):
  with(O/(phase+'.log')).open('x')as log:
   p=subprocess.Popen(user_container.command(['bash',str(script),'-m','native_fit7_postexit_recovery_v2.recover',*args],mount_project=True),stdout=log,stderr=subprocess.STDOUT);code=p.wait()
  write(O/('actual_'+phase+'_host_exit.json'),dict(exit_code=code,actual_Popen_wait_return=True,proc_absent_after_wait=not Path('/proc',str(p.pid)).exists(),pid=p.pid));assert code==0,('Actual failure retained',phase)
if __name__=='__main__':main()

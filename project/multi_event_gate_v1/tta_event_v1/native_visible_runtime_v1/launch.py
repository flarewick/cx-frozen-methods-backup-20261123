"""Actual one host wait return; run this bounded launcher as a user service."""
from pathlib import Path
import subprocess,sys,time,uuid,os,fcntl
from .paths import O,C,write,sha

def main():
 if list(O.glob('host_run_*')):raise ValueError('Prior host exists; inspect rather than retry or duplicate')
 from native_resource_policy_v4.admission import require_admission,check_admission
 from native_fast_inventory_v1.inventory import inventory
 from native_trained_admission_resume_v1.launch import verify_host
 from gpu_doctor import gpu_snapshot
 import user_container as container
 from .paths import read
 spec=read(O/'protocol.json');verify_host(spec);waiting=time.monotonic();waits=0
 while True:
  if time.monotonic()-waiting>1800:raise TimeoutError('CPU admission wait expired without GPU')
  try:
   lockdir=container.WORK/'gpu_locks'
   with(lockdir/'account_admission.lock').open('a')as account,(lockdir/'gpu_5.lock').open('a')as card:
    fcntl.flock(account,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require_admission(inventory(),spec['required_gpu_uuid']);check_admission(gpu_snapshot(5))
   break
  except(BlockingIOError,RuntimeError)as exc:
   waits+=1
   if waits==1 or waits%30==0:print('CPU_ONLY_FRESH_ADMISSION_WAIT',waits,str(exc),flush=True)
   time.sleep(2)
 folder=O/('host_run_'+uuid.uuid4().hex);folder.mkdir(parents=True)
 command=[sys.executable,'-B',str(C/'host.py'),'--gpu','5','--run'];began=time.monotonic()
 with (folder/'host.log').open('x')as log:
  process=subprocess.Popen(command,cwd=C.parent,stdout=log,stderr=subprocess.STDOUT)
  started=int(Path('/proc/%d/stat'%process.pid).read_text().split(') ',1)[1].split()[19])
  write(folder/'launch.json',dict(host_pid=process.pid,host_started_ticks=started,observer_pid=os.getpid(),command=command,observer_source_sha256=sha(__file__)))
  print('TRAINED_NATIVE_HOST_STARTED',process.pid,str(folder),flush=True);code=process.wait()
 write(folder/'exit.json',dict(exit_code=code,actual_Popen_wait_return=True,host_pid=process.pid,host_started_ticks=started,seconds=time.monotonic()-began))
 print('TRAINED_NATIVE_ACTUAL_HOST_EXIT',code,str(folder),flush=True);raise SystemExit(code)
if __name__=='__main__':main()

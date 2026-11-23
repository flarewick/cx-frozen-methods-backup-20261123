"""Actual one host wait return; run this bounded launcher as a user service."""
from pathlib import Path
import subprocess,sys,time,uuid,os
from .paths import O,C,write,sha

def main():
 if list(O.glob('host_run_*')):raise ValueError('Prior host exists; inspect rather than retry or duplicate')
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

"""Capture actual host wait return separately from immutable GPU data claims."""
from pathlib import Path
import json,subprocess,sys,time,uuid,hashlib,os
T=Path(__file__).resolve().parents[1];O=T/'outputs/native_visible_expand_initial_v2'
def write(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 if list(O.glob('host_run_*')):raise ValueError('A host observation exists; inspect it rather than duplicate')
 run=O/('host_run_'+uuid.uuid4().hex);run.mkdir(parents=True)
 command=[sys.executable,'-B',str(T/'native_visible_expand_initial_v2/host.py'),'--gpu','5','--run']
 began=time.monotonic()
 with (run/'host.log').open('x')as log:
  p=subprocess.Popen(command,cwd=T,stdout=log,stderr=subprocess.STDOUT)
  ticks=int(Path('/proc/%d/stat'%p.pid).read_text().split(') ',1)[1].split()[19])
  write(run/'launch.json',dict(host_pid=p.pid,host_started_ticks=ticks,command=command,observer_pid=os.getpid(),observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
  print('DEVELOPER_Q0_HOST_STARTED',str(run),p.pid,flush=True)
  code=p.wait()
 write(run/'exit.json',dict(exit_code=code,actual_Popen_wait_return=True,host_pid=p.pid,host_started_ticks=ticks,seconds=time.monotonic()-began))
 print('DEVELOPER_Q0_ACTUAL_HOST_EXIT',code,str(run),flush=True)
 raise SystemExit(code)
if __name__=='__main__':main()

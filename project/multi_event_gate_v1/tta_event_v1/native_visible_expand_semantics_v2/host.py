"""Record actual CPU semantic resume exit independently of output existence."""
from pathlib import Path
import sys,json,subprocess,time,uuid,os
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent;O=T/'outputs/native_visible_expand_semantics_v2'
def write(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 assert not list(O.glob('host_run_*'))and not(O/'protocol.json').exists()
 folder=O/('host_run_'+uuid.uuid4().hex);folder.mkdir();began=time.monotonic();cmd=[sys.executable,'-B',str(C/'cpu_entry.py'),'--','-m','native_visible_expand_semantics_v2.run']
 with(folder/'host.log').open('x')as f:
  p=subprocess.Popen(cmd,cwd=T,stdout=f,stderr=subprocess.STDOUT);ticks=int(Path('/proc/%d/stat'%p.pid).read_text().split(') ',1)[1].split()[19]);write(folder/'launch.json',dict(host_pid=p.pid,host_started_ticks=ticks,observer_pid=os.getpid(),command=cmd));code=p.wait()
 write(folder/'exit.json',dict(exit_code=code,actual_Popen_wait_return=True,host_pid=p.pid,host_started_ticks=ticks,seconds=time.monotonic()-began));raise SystemExit(code)
if __name__=='__main__':main()

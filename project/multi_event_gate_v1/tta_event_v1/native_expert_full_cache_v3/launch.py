"""Record actual host process identity and exit; guarded host owns GPU admission."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,uuid
T=Path(__file__).resolve().parents[1];O=T/'outputs/native_expert_full_cache_v3'
def write(p,value):
 with p.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
def identity(pid):
 s=(Path('/proc')/str(pid)/'stat').read_text();return dict(pid=pid,start_ticks=int(s[s.rfind(')')+2:].split()[19]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
 run=O/('host_run_'+uuid.uuid4().hex);run.mkdir(parents=True);start=time.monotonic()
 cmd=[sys.executable,'-B',str(T/'native_expert_full_cache_v3/host.py'),'--gpu',str(a.gpu),'--run']
 with(run/'console.log').open('x')as f:
  child=subprocess.Popen(cmd,cwd=T,stdout=f,stderr=subprocess.STDOUT)
  write(run/'start.json',dict(wrapper=identity(os.getpid()),host=identity(child.pid),command=cmd,started_UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
  print('ORIGINAL96_HOST_STARTED',run,child.pid,flush=True);code=child.wait()
 write(run/'exit.json',dict(exit_code=code,seconds=time.monotonic()-start,host_pid=child.pid,actual_wait_return=True))
 print('ORIGINAL96_HOST_EXIT',code,run,flush=True);raise SystemExit(code)
if __name__=='__main__':main()

"""Detached, bounded native host supervision; no GPU/resource policy changes."""
from pathlib import Path
import argparse,datetime,fcntl,json,os,subprocess,sys,uuid
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1];O=T/'outputs/native_feedback_confirm_v5'
def protocol():return json.loads((T/'outputs/native_feedback_confirm_v5/protocol.json').read_text())['identity']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def started(pid):return int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker');a=p.parse_args();PROTOCOL=protocol()
 assert json.loads((T/'outputs/native_feedback_confirm_v5/protocol.json').read_text())['identity']==PROTOCOL
 if a.worker:
  assert len(a.worker)==32 and all(c in'0123456789abcdef'for c in a.worker)
  folder=O/('daemon_'+a.worker);lock=(O/'native_feedback_confirm_v5_daemon.lock').open('a')
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  write(folder/'supervisor.json',dict(pid=os.getpid(),started_ticks=started(os.getpid()),session_id=os.getsid(0),UTC=now(),protocol_identity=PROTOCOL))
  command=[sys.executable,'-B',str(T/'native_feedback_confirm_v5/host.py'),'--gpu','6','--run']
  try:
   process=subprocess.Popen(command,cwd=P,stdin=subprocess.DEVNULL)
   write(folder/'host.json',dict(pid=process.pid,started_ticks=started(process.pid),UTC=now(),protocol_identity=PROTOCOL))
   code=process.wait()
  except BaseException as exc:
   write(folder/'supervisor_exception.json',dict(type=type(exc).__name__,reason=str(exc),UTC=now()))
   raise
  write(folder/'exit.json',dict(exit_code=code,UTC=now(),protocol_identity=PROTOCOL,scope='Actual supervised host return code. Native/resource success still requires its own evidence.'))
  raise SystemExit(code)
 O.mkdir(parents=True,exist_ok=True)
 for f in O.glob('daemon_*/supervisor.json'):
  v=json.loads(f.read_text())
  if v['protocol_identity']!=PROTOCOL or(f.parent/'exit.json').exists():continue
  try:live=started(v['pid'])==v['started_ticks']
  except(FileNotFoundError,ProcessLookupError):live=False
  if live:raise RuntimeError('Existing detached supervisor is live; inspect it rather than relaunch')
 run=uuid.uuid4().hex;folder=O/('daemon_'+run);folder.mkdir()
 with (folder/'console.log').open('x')as log:
  process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',run],cwd=P,
    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 write(folder/'launch.json',dict(pid=process.pid,started_ticks=started(process.pid),UTC=now(),protocol_identity=PROTOCOL,
    detached_session=True,unchanged_guard=True,guard_deadline_seconds=7200,only_native_TTA=True))
 print('DETACHED_NATIVE_SUPERVISOR',json.dumps(dict(pid=process.pid,folder=str(folder),protocol_identity=PROTOCOL)),flush=True)
if __name__=='__main__':main()

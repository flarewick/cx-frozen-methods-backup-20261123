"""CPU real fork/received-SIGINT sender and normal exit forwarding proof."""
from pathlib import Path
import sys,os,json,time,subprocess,signal,re,hashlib
from .trace import command
T=Path(__file__).resolve().parents[1];O=T/'outputs/native_signal_origin_v1'
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
def child(root):
 pid=os.fork()
 if pid==0:
  save(root/'ready.json',dict(pid=os.getpid(),uid=os.getuid()))
  try:
   while True:time.sleep(.05)
  except KeyboardInterrupt:
   save(root/'interrupted.json',dict(actual_KeyboardInterrupt=True,signal_not_ignored=True));os._exit(23)
 _,status=os.waitpid(pid,0);os._exit(os.waitstatus_to_exitcode(status))
def main():
 O.mkdir(parents=True,exist_ok=True);root=O/'CPU_fixture';root.mkdir();log=(root/'process.log').open('x')
 p=subprocess.Popen(command([sys.executable,'-u','-B','-m','native_signal_origin_v1.checks','--child',str(root)],root/'received.strace'),cwd=T,stdout=log,stderr=subprocess.STDOUT)
 try:
  start=time.monotonic()
  while not(root/'ready.json').exists():
   if p.poll()is not None:raise RuntimeError('Signal fixture exited: '+(root/'process.log').read_text())
   if time.monotonic()-start>15:raise TimeoutError('No signal fixture ready')
   time.sleep(.02)
  target=json.loads((root/'ready.json').read_text());os.kill(target['pid'],signal.SIGINT);code=p.wait(timeout=15);assert code==23
  assert json.loads((root/'interrupted.json').read_text())['signal_not_ignored'];trace=(root/'received.strace').read_text()
  assert re.search(r'SIGINT.*si_pid='+str(os.getpid())+r'[,}]',trace),trace
  assert re.search(r'SIGINT.*si_uid='+str(os.getuid())+r'[,}]',trace),trace
  normal=subprocess.run(command([sys.executable,'-B','-c','raise SystemExit(0)'],root/'normal.strace'),capture_output=True,text=True,timeout=15);assert normal.returncode==0
  save(O/'CPU_signal_origin_checks.json',dict(passed=True,actual_signal_subtree_wrapper_exit=code,original_KeyboardInterrupt_delivered=True,actual_host_sender_PID=os.getpid(),actual_host_sender_UID=os.getuid(),sender_in_received_signal_trace=True,actual_normal_wrapper_exit=normal.returncode,no_signal_masking_or_ignoring=True,no_syscall_capture=True,no_GPU_calls=True,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in Path(__file__).parent.glob('*.py')}))
  print('CPU_SIGNAL_SENDER_AND_EXIT_FORWARDING_PASS',flush=True)
 finally:
  if p.poll()is None:p.terminate();p.wait(timeout=5)
  log.close()
if __name__=='__main__':
 if len(sys.argv)>1:child(Path(sys.argv[2]))
 else:main()

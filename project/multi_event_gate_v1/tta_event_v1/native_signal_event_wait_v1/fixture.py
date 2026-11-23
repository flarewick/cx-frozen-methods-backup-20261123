"""Real CPU child, original exit wait, deliberately changed sigaction flags."""
import argparse,ctypes,json,os,signal,time,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--traced',action='store_true');p.add_argument('--mode',choices=['normal','interrupt','groupstop'],required=True);a=p.parse_args();root=a.root;root.mkdir(parents=True,exist_ok=True)
 import torch,numpy as np
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 before_threads=len(list(Path('/proc/self/task').iterdir()))
 r,w=os.pipe2(os.O_CLOEXEC);child=os.fork()
 if child==0:
  try:
   os.close(w);assert os.read(r,1)==b'1';os.close(r)
   def handler(signum,frame):
    p=root/'handled.json';p.write_text(json.dumps(dict(signum=signum,once=not p.exists()))+'\n');raise KeyboardInterrupt()
   signal.signal(signal.SIGINT,handler)
   from native_sigint_receipt_v2.receipt import Hook
   hook=Hook(root)
   # Same handler pointer re-registered with libc.signal reproduces missing SA_SIGINFO.
   lib=ctypes.CDLL(None);lib.signal.argtypes=[ctypes.c_int,ctypes.c_void_p];lib.signal.restype=ctypes.c_void_p
   class Action(ctypes.Structure):_fields_=[('fn',ctypes.c_void_p),('flags',ctypes.c_ulong),('restorer',ctypes.c_void_p),('mask',ctypes.c_ulong)]
   act=Action();lib.syscall(ctypes.c_long(13),ctypes.c_int(2),ctypes.c_void_p(0),ctypes.byref(act),ctypes.c_int(8));assert act.flags&4
   assert lib.signal(2,act.fn)==act.fn
   now=Action();lib.syscall(ctypes.c_long(13),ctypes.c_int(2),ctypes.c_void_p(0),ctypes.byref(now),ctypes.c_int(8));assert now.fn==act.fn and not now.flags&4
   (root/'ready.json').write_text(json.dumps(dict(child_pid=os.getpid(),uid=os.getuid(),original_siginfo_flag_lost=True,OS_threads=len(list(Path('/proc/self/task').iterdir())),CUDA_initialized=torch.cuda.is_initialized()))+'\n')
   if a.mode!='normal':
    while True:signal.pause()
   torch.manual_seed(17);np.random.seed(17);t=torch.randn(128,128)
   start=time.monotonic()
   for _ in range(250):out=t@t.T
   arr=np.random.randn(128);status=dict(x.split(':',1)for x in Path('/proc/self/status').read_text().splitlines()if ':'in x)
   (root/'normal.json').write_text(json.dumps(dict(tensor=hashlib.sha256(out.numpy().tobytes()).hexdigest(),array=hashlib.sha256(arr.tobytes()).hexdigest(),RNG=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),threads=len(list(Path('/proc/self/task').iterdir())),voluntary_context_switches=int(status['voluntary_ctxt_switches']),nonvoluntary_context_switches=int(status['nonvoluntary_ctxt_switches']),elapsed=time.monotonic()-start,CUDA_initialized=torch.cuda.is_initialized()))+'\n')
   hook.close();os._exit(0)
  except KeyboardInterrupt:os._exit(130)
  except BaseException:
   import traceback;traceback.print_exc();os._exit(1)
 os.close(r)
 if a.traced:
  from .wait import attach
  attach(child)
 os.write(w,b'1');os.close(w)
 from native_best100_exit_v1.wait import wait_and_record as original
 kwargs=dict(attempt_nonce='CPUfixture',gpu_uuid='CPU_NONE',protocol_identity='CPUfixture',cpu_only_fixture=True)
 if a.traced:
  from .wait import wait_and_record
  code=wait_and_record(child,root/'original_exit.json',original_wait=original,events=root/'events.jsonl',**kwargs)
 else:code=original(child,root/'original_exit.json',**kwargs)
 assert len(list(Path('/proc/self/task').iterdir()))==before_threads and not torch.cuda.is_initialized()
 (root/'parent_exit.json').write_text(json.dumps(dict(child_code=code,no_extra_parent_threads=True,no_GPU_work=True))+'\n')
 raise SystemExit(0 if code in(0,130)else code)
if __name__=='__main__':main()

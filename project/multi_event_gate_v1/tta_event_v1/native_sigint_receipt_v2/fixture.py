"""CPU-only signal and tensor fixture inside the real container namespaces."""
from pathlib import Path
import argparse,hashlib,json,os,signal
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['normal','interrupt','one_argument'],required=True);p.add_argument('--receipt',action='store_true');p.add_argument('--sapien',action='store_true');a=p.parse_args()
 root=a.root;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 assert not torch.cuda.is_initialized()
 def write(name,value):(root/name).write_text(json.dumps(value,indent=2)+'\n')
 def handler(signum,frame):
  write('handled.json',dict(signum=signum,original_interrupt_not_ignored=True))
  raise KeyboardInterrupt('CPU fixture preserves original interrupt')
 signal.signal(signal.SIGINT,handler)
 original=signal.getsignal(signal.SIGINT);threads_before=len(list(Path('/proc/self/task').iterdir()))
 hook=None
 if a.receipt:
  from .receipt import Hook
  hook=Hook(root)
 assert signal.getsignal(signal.SIGINT)is original and len(list(Path('/proc/self/task').iterdir()))==threads_before
 if a.sapien:
  import sapien
  system=sapien.physx.PhysxCpuSystem();scene=sapien.Scene([system]);scene.step()
  write('CPU_physx.json',dict(actual_CPU_system_step=True,no_renderer=True,CUDA_initialized=torch.cuda.is_initialized()))
 status=dict(line.split(':',1)for line in Path('/proc/self/status').read_text().splitlines()if ':'in line)
 write('ready.json',dict(pid=os.getpid(),uid=os.getuid(),seccomp=int(status['Seccomp']),seccomp_filters=int(status.get('Seccomp_filters','0')),CUDA_initialized=torch.cuda.is_initialized(),OS_threads=threads_before))
 if a.mode in('interrupt','one_argument'):
  try:
   if a.mode=='one_argument':
    assert hook is not None
    hook.lib.fixture_saved_one_argument_handler.restype=None
    hook.lib.fixture_saved_one_argument_handler()
    raise AssertionError('Original interrupt handler did not run')
   while True:signal.pause()
  finally:
   if hook is not None:hook.close()
 torch.manual_seed(17);np.random.seed(17)
 x=torch.randn(64,64);y=x@x.T;arr=np.random.randn(64).astype(np.float64)
 if hook is not None:hook.close()
 assert signal.getsignal(signal.SIGINT)is original and len(list(Path('/proc/self/task').iterdir()))==threads_before
 write('normal.json',dict(tensor_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest(),numpy_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),torch_RNG_sha256=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),CUDA_initialized=torch.cuda.is_initialized()))

if __name__=='__main__':main()

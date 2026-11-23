"""CPU-only SAM2, real single-thread fork check, one normally reaped CUDA child."""
from pathlib import Path
import argparse,os,time,traceback,sys
from .common import read,write,verify
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;spec=read(root/'protocol.json');verify(spec)
 selected=os.environ['CUDA_VISIBLE_DEVICES'];os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
 from query_object_v1.stream import video
 from adapter_cache_cpu import state_digest
 with torch.random.fork_rng(devices=[]):model=video.build('cpu')
 digest=state_digest(model);assert not torch.cuda.is_initialized()
 assert len(list(Path('/proc/self/task').iterdir()))==1
 child=os.fork()
 if child==0:
  assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()and int(torch.arange(10).sum())==45;os._exit(0)
 _,status=os.waitpid(child,0);assert os.waitstatus_to_exitcode(status)==0
 barrier=root/'barrier';write(barrier/'ready.json',dict(passed=True,cpu_fork_child_passed=True,os_threads=1,CUDA_initialized=False,model_state_sha256=digest,source_sha256=spec['source_sha256']))
 began=time.monotonic()
 while not(barrier/'permit.json').exists():
  if time.monotonic()-began>1500:raise TimeoutError('CPU-only wait expired')
  time.sleep(.1)
 permit=read(barrier/'permit.json');assert permit['gpu_uuid']==selected and permit['attempt_nonce']==spec['attempt_nonce']and permit['protocol_identity']==spec['identity'];verify(spec)
 child=os.fork()
 if child==0:
  try:
   os.environ['CUDA_VISIBLE_DEVICES']=selected;assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()
   from .runtime import execute
   execute(spec,root,dict(model=model),barrier)
   from native_best100_exit_v1.success import complete_and_exit
   complete_and_exit()
  except BaseException:traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
 from native_best100_exit_v1.wait import wait_and_record
 code=wait_and_record(child,barrier/'normal_child_exit.json',attempt_nonce=spec['attempt_nonce'],gpu_uuid=selected,protocol_identity=spec['identity'],cpu_only_fixture=False)
 assert code==0
 assert not torch.cuda.is_initialized()and state_digest(model)==digest;verify(spec)
 write(root/'CPU_parent_result.json',dict(passed=True,actual_child_exit_code=0,CPU_parent_CUDA_initialized=False,model_source_unchanged=True))
if __name__=='__main__':main()

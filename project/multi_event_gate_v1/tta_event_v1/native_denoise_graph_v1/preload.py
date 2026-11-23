"""Original proven CPU model load/singlethreadfork before one bounded GPU child."""
from pathlib import Path
import argparse,os,time,traceback
from .common import read,write,verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;s=read(root/'protocol.json');verify(s)
 selected=os.environ['CUDA_VISIBLE_DEVICES'];os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 from native_cross_task100_v5.preflight import prepare
 from native_best100_exit_v1.wait import wait_and_record
 from native_best100_exit_v1.success import complete_and_exit
 prepared=prepare(s,root)
 try:
  child=os.fork()
  if child==0:
   assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()and int(torch.arange(10).sum())==45;os._exit(0)
  _,status=os.waitpid(child,0);assert os.waitstatus_to_exitcode(status)==0
  assert prepared['ready']['os_threads']==1 and not prepared['ready']['cuda_initialized'];write(root/'barrier/ready.json',dict(passed=True,source_sha256=s['source_sha256'],actual_original_CPU_preload=prepared['ready'],cpu_fork_child_passed=True));began=time.monotonic()
  while not(root/'barrier/permit.json').exists():
   if time.monotonic()-began>1500:raise TimeoutError('CPU-only admission timeout')
   time.sleep(.1)
  permit=read(root/'barrier/permit.json');assert permit['gpu_uuid']==selected and permit['nonce']==s['attempt_nonce'];verify(s)
  child=os.fork()
  if child==0:
   try:
    os.environ['CUDA_VISIBLE_DEVICES']=selected;assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()
    from .runtime import execute
    execute(s,root,prepared);complete_and_exit()
   except BaseException:traceback.print_exc();__import__('sys').stdout.flush();__import__('sys').stderr.flush();os._exit(1)
  code=wait_and_record(child,root/'barrier/normal_child_exit.json',attempt_nonce=s['attempt_nonce'],gpu_uuid=selected,protocol_identity=s['identity'],cpu_only_fixture=False);assert code==0
  verify(s);assert read(root/'GPU_probe_result.json')['passed'];write(root/'CPU_parent_result.json',dict(passed=True,actual_child_exit_code=0,source_runtime_audit=prepared['storage'].verify_source(),CPU_parent_CUDA_initialized=torch.cuda.is_initialized()))
 finally:prepared['storage'].cleanup()
if __name__=='__main__':main()

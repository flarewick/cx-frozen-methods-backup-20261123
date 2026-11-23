"""Source-bound real CPU preload/fork; exact one GPU child; CPU final audit."""
from pathlib import Path
import argparse,os,sys,time,traceback
from .paths import O,read,write,verify,dataset
from .prepare import prepare as freeze
from .preflight import prepare

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);p.add_argument('--barrier',type=Path);p.add_argument('--nonce');p.add_argument('--cpu-proof',action='store_true');a=p.parse_args()
 selected=os.environ.get('CUDA_VISIBLE_DEVICES','');os.environ['CUDA_VISIBLE_DEVICES']=''
 spec=freeze();verify(spec);root,manifest=dataset(spec,a.gpu_uuid)
 assert not(root/'GPU_episodes_complete.json').exists(),'Preserve prior complete episodes'
 import torch
 prepared=prepare(spec,root)
 try:
  child=os.fork()
  if child==0:
   assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()and int(torch.arange(10).sum())==45;os._exit(0)
  waited,status=os.waitpid(child,0);assert waited==child and os.waitstatus_to_exitcode(status)==0
  ready=dict(prepared['ready'],protocol_identity=spec['identity'],source_sha256=spec['source_sha256'],cpu_fork_child_passed=True)
  if a.cpu_proof:
   write(O/'source_bound_CPU_preload_proof.json',ready);print('SOURCE_BOUND_NATIVE_EPISODE_CPU_PRELOAD_FORK_PASS',flush=True);return
  assert a.barrier is not None and a.nonce;verify(spec);write(a.barrier/'ready.json',ready);began=time.monotonic()
  while not(a.barrier/'permit.json').exists():
   if time.monotonic()-began>240:raise TimeoutError('No fresh source-bound GPU admission permit')
   time.sleep(.05)
  permit=read(a.barrier/'permit.json')
  assert permit['gpu_uuid']==a.gpu_uuid and permit['attempt_nonce']==a.nonce and permit['protocol_identity']==spec['identity']
  child=os.fork()
  if child==0:
   try:
    os.environ['CUDA_VISIBLE_DEVICES']=selected;assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()
    from .runtime import execute
    execute(spec,root,manifest,prepared,a.barrier)
   except BaseException:traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
   os._exit(0)
  from native_nextchunk_v1.child_exit import wait_and_record
  code=wait_and_record(child,a.barrier/'normal_child_exit.json',attempt_nonce=a.nonce,gpu_uuid=a.gpu_uuid,protocol_identity=spec['identity'],cpu_only_fixture=False)
  if code!=0:raise SystemExit(code)
  assert not torch.cuda.is_initialized();verify(spec)
  from .analysis import audit
  check=audit(spec,root)
  write(prepared['session']/'source_storage_audit.json',prepared['storage'].verify_source())
  write(root/'result.json',dict(complete=True,protocol_identity=spec['identity'],manifest_identity=manifest['identity'],CPU_parent_CUDA_initialized=False,CPU_audit=check,actual_developer_native_episodes=20,new_independent_scenes=0,resource_qualification_requires_actual_guard_terminal=True,TTA_utility_proven=False))
 finally:prepared['storage'].cleanup()
if __name__=='__main__':main()

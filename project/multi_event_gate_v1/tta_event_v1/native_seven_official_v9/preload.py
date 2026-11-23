"""Actual CPU model load and one normal CUDA child; no environment changes."""
from pathlib import Path
import argparse,os,sys,time,traceback,signal
from .common import read,write,verify
from native_best100_exit_v1.wait import wait_and_record
from native_best100_exit_v1.success import complete_and_exit

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--barrier',type=Path,required=True);p.add_argument('--nonce',required=True);a=p.parse_args();root=a.batch;spec=read(root/'protocol.json');verify(spec)
 def interrupted(signum,frame):
  from .common import replace
  replace(root/('signal_received_'+str(os.getpid())+'.json'),dict(signal=signum,signal_name=signal.Signals(signum).name,inner_pid=os.getpid(),monotonic_ns=time.monotonic_ns(),stack=traceback.format_stack(frame),sender_unknown=True))
  if signum==signal.SIGINT:raise KeyboardInterrupt('Recorded external SIGINT')
  raise SystemExit(128+signum)
 for sig in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
 selected=os.environ.get('CUDA_VISIBLE_DEVICES','');os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 from .preflight import prepare
 prepared=prepare(spec,root)
 from native_cpu_topp_v1.runtime import prepare as prepare_TOPP,close as close_TOPP
 pool=prepare_TOPP(root,write)
 try:
  child=os.fork()
  if child==0:
   assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()and int(torch.arange(10).sum())==45;os._exit(0)
  _,status=os.waitpid(child,0);assert os.waitstatus_to_exitcode(status)==0
  ready=dict(prepared['ready'],protocol_identity=spec['identity'],source_sha256=spec['source_sha256'],cpu_fork_child_passed=True)
  assert ready['os_threads']==1 and not ready['cuda_initialized'];write(a.barrier/'ready.json',ready);started=time.monotonic()
  while not(a.barrier/'permit.json').exists():
   if time.monotonic()-started>1800:raise TimeoutError('Fresh GPU admission wait expired')
   time.sleep(.1)
  permit=read(a.barrier/'permit.json');assert permit['gpu_uuid']==selected and permit['attempt_nonce']==a.nonce and permit['protocol_identity']==spec['identity'];verify(spec)
  child=os.fork()
  if child==0:
   try:
    os.environ['CUDA_VISIBLE_DEVICES']=selected;assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()
    from native_direct_frames_v1.frames import install as install_direct_stack
    install_direct_stack()
    from native_guide_reuse_v1.typecheck import install as install_guide_typecheck
    install_guide_typecheck()
    from .runtime import execute
    execute(spec,root,dict(gpu_uuid=selected),prepared,a.barrier,exit_success=complete_and_exit)
   except BaseException:traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
   os._exit(0)
  code=wait_and_record(child,a.barrier/'normal_child_exit.json',attempt_nonce=a.nonce,gpu_uuid=selected,protocol_identity=spec['identity'],cpu_only_fixture=False)
  if code:raise SystemExit(code)
  close_TOPP(pool,root,write)
  assert not torch.cuda.is_initialized();verify(spec)
  from .analysis import audit
  checked=audit(spec,root)
  write(prepared['session']/'source_storage_audit.json',prepared['storage'].verify_source());write(root/'result.json',dict(complete=True,protocol_identity=spec['identity'],CPU_audit=checked,actual_native_episodes=checked['actual_native_episodes'],actual_child_exit_code=0,CPU_parent_CUDA_initialized=False,resource_qualification_requires_actual_guard_terminal=True))
 finally:
  pool.close();prepared['storage'].cleanup()
if __name__=='__main__':main()

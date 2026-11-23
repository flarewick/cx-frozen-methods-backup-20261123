"""Actual CPU-only runtime preload and one normal fork child for a short q0 bridge."""
from pathlib import Path
import argparse,os,sys,time,traceback,uuid
from native_expert_smoke_v2.paths import O,read,write,verify,dataset,sha
from native_expert_smoke_v2.prepare import prepare as freeze
def prepare(spec,root):
 import torch
 from native_feedback_confirm_v5.cpu_placement import apply,pages
 from cx_tta_online.native_entry_storage import NativeEntryStorage
 from cx_tta.evaluate_b2 import resolve_tokenizer
 from cx_tta_feedback_confirm5.schedule import instrument
 placement=read(spec['CPU_placement']);apply(placement)
 torch.set_num_threads(1)
 session=root/'sessions'/('process_'+uuid.uuid4().hex[:12]);session.mkdir(parents=True)
 class Storage(NativeEntryStorage):
  def prepare_runtime(self,folder):
   runtime=super().prepare_runtime(folder);file=runtime/'envs/_base_task.py';file.write_text(instrument(file.read_text()));return runtime
 storage=Storage(runtime_source=Path(spec['runtime_source']))
 try:
  runtime=storage.prepare_runtime(session);storage.configure_runtime(session,runtime,resolve_tokenizer())
  from native_expert_smoke_v2.runtime import execute
  from native_expert_smoke_v2.analysis import audit
  assert not torch.cuda.is_initialized()and len(list(Path('/proc/self/task').iterdir()))==1
  return dict(storage=storage,runtime=runtime,session=session,CPU_placement=placement,no_VLA_or_semantic_model_loaded=True)
 except BaseException:storage.cleanup();raise
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);p.add_argument('--barrier',type=Path);p.add_argument('--nonce');p.add_argument('--cpu-proof',action='store_true');a=p.parse_args()
 selected=os.environ.get('CUDA_VISIBLE_DEVICES','');os.environ['CUDA_VISIBLE_DEVICES']=''
 spec=freeze();verify(spec);root,manifest=dataset(spec,a.gpu_uuid)
 if not a.cpu_proof:assert not(root/'GPU_frames_complete.json').exists(),'Existing capture preserved'
 import torch
 prepared=prepare(spec,root)
 try:
  child=os.fork()
  if child==0:
   assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()and int(torch.arange(10).sum())==45
   os._exit(0)
  _,status=os.waitpid(child,0);assert os.waitstatus_to_exitcode(status)==0
  ready=dict(passed=True,protocol_identity=spec['identity'],source_sha256=spec['source_sha256'],
    os_threads=len(list(Path('/proc/self/task').iterdir())),cuda_initialized=torch.cuda.is_initialized(),
    cpu_fork_child_passed=True,CPU_affinity=sorted(os.sched_getaffinity(0)),no_VLA_or_semantic_model_loaded=True)
  if a.cpu_proof:
   write(O/'cpu_preload_proof.json',ready);print('INITIAL_FRAME_ACTUAL_CPU_PREFLIGHT_FORK_PASS',flush=True);return
  assert a.barrier is not None and a.nonce
  write(a.barrier/'ready.json',ready);began=time.monotonic()
  while not(a.barrier/'permit.json').exists():
   if time.monotonic()-began>240:raise TimeoutError('No fresh initial-frame GPU permit')
   time.sleep(.05)
  permit=read(a.barrier/'permit.json')
  assert permit['gpu_uuid']==a.gpu_uuid and permit['attempt_nonce']==a.nonce and permit['protocol_identity']==spec['identity']
  child=os.fork()
  if child==0:
   try:
    os.environ['CUDA_VISIBLE_DEVICES']=selected
    assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()
    from native_expert_smoke_v2.runtime import execute
    execute(spec,root,manifest,prepared,a.barrier)
   except BaseException:traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
   os._exit(0)
  from native_nextchunk_v1.child_exit import wait_and_record
  code=wait_and_record(child,a.barrier/'normal_child_exit.json',attempt_nonce=a.nonce,gpu_uuid=a.gpu_uuid,protocol_identity=spec['identity'],cpu_only_fixture=False)
  if code!=0:raise SystemExit(code)
  assert not torch.cuda.is_initialized()
  from native_expert_smoke_v2.analysis import audit
  check=audit(spec,root)
  write(prepared['session']/'source_storage_audit.json',prepared['storage'].verify_source())
  write(root/'result.json',dict(complete=True,protocol_identity=spec['identity'],manifest_identity=manifest['identity'],
   CPU_parent_CUDA_initialized=False,CPU_audit=check,no_VLA_or_semantic_model_loaded=True,
   short_expert_interface_only=True,long_resource_qualification_claimed=False,new_independent_scenes=0,online_updates=0,TTA_utility_proven=False))
 finally:prepared['storage'].cleanup()
if __name__=='__main__':main()

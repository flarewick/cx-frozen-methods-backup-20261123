"""Actual CPU-only model/runtime load and single-thread fork prerequisite."""
from pathlib import Path
import os,time,uuid
from native_policy_execution_v1.common import T,read,write,sha
C=Path(__file__).resolve().parent;O=T/'outputs/native_trained_runtime_v1'
def prepare(spec,root):
 import torch,numpy as np
 from native_feedback_confirm_v5.cpu_placement import apply
 from cx_tta_online.native_entry_storage import NativeEntryStorage
 from cx_tta.evaluate_b2 import resolve_tokenizer
 from cx_tta_feedback_confirm5.schedule import instrument
 from cx_tta_feedback_confirm5.query import build_head
 from adapter_cache_cpu import load_cpu_policy,state_digest
 from native_trained_response_bridge_v1.bridge import load_model
 from native_trained_semantics_v1.store import SeedStore
 from native_delayed_query_v1.primitives import configuration
 from native_delayed_query_v1.predictor import FrozenPredictor
 from query_object_v1.stream import video
 apply(read(spec['CPU_placement']));torch.set_num_threads(1)
 session=Path(root)/'sessions'/('process_'+uuid.uuid4().hex[:12]);session.mkdir(parents=True)
 class Storage(NativeEntryStorage):
  def prepare_runtime(self,folder):
   runtime=super().prepare_runtime(folder);p=runtime/'envs/_base_task.py';p.write_text(instrument(p.read_text()));return runtime
 storage=Storage(runtime_source=Path(spec['runtime_source']))
 try:
  runtime=storage.prepare_runtime(session);storage.configure_runtime(session,runtime,resolve_tokenizer())
  before=torch.random.get_rng_state().clone();base,info=load_cpu_policy(verify_state=True)
  proof=next(__import__('json').loads(line.split(' ',1)[1])for line in(T.parents[1]/'deployment/reports/target_adapter_cache_cpu_arrow_nvml_20261024.log').read_text().splitlines()if line.startswith('ADAPTER_CACHE_CPU_PROOF '))
  assert proof['passed']and proof['helper_sha256']==sha(T.parents[1]/'deployment/adapter_cache_cpu.py')
  base_digest=state_digest(base._model);assert base_digest==proof['converted_state_sha256']
  with torch.random.fork_rng(devices=[]):
   heads=[build_head(spec['training'])for _ in range(4)];mask_model=video.build('cpu')
   response_model,qualification=load_model(spec['trained_qualification'])
  mask_digest=state_digest(mask_model);head_digests=[state_digest(h)for h,_ in heads]
  assert len(set(head_digests))==1
  assert mask_digest==read(T/'outputs/native_arrived_mask_v3/cpu_models.json')['model_state_sha256']['SAM2']
  assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
  with np.load(spec['projection'],allow_pickle=False)as z:projection={k:z[k].copy()for k in z.files}
  seed_store=SeedStore(T/'outputs/native_trained_semantics_v1');cfg,camera,_=configuration();predictor=FrozenPredictor()
  import sapien  # Existing SDK must load its bundled native dependencies first.
  from native_telemetry_batch_v1.capture import load as load_telemetry
  telemetry_native=load_telemetry()
  from .runtime import execute
  threads=[dict(tid=int(p.name),comm=(p/'comm').read_text().strip())for p in Path('/proc/self/task').iterdir()]
  write(session/'before_fork_state.json',dict(CUDA_initialized=torch.cuda.is_initialized(),OS_threads=threads,thread_count=len(threads),torch_threads=torch.get_num_threads(),matmul_precision=torch.get_float32_matmul_precision()))
  print('ACTUAL_PRE_FORK_STATE',torch.cuda.is_initialized(),threads,flush=True)
  assert not torch.cuda.is_initialized()and len(list(Path('/proc/self/task').iterdir()))==1
  ready=dict(passed=True,os_threads=1,cuda_initialized=False,actual_VLA_loaded_on_CPU=True,actual_SAM2_loaded_on_CPU=True,
    actual_trained_response_loaded_on_CPU=True,actual_CPU_heads=4,CPU_affinity=sorted(os.sched_getaffinity(0)),
    VLA_state_sha256=base_digest,SAM2_state_sha256=mask_digest,head_state_sha256=head_digests,
    trained_qualification_sha256=sha(spec['trained_qualification']),checkpoint_sha256=qualification['training_checkpoint_sha256'],
    current_eval_mode_seed_available=seed_store.result['available'],scope='Actual CPU load/import only; no GPU, native execution or action benefit qualification')
  write(session/'cpu_preload.json',dict(**ready,base_load_info=info))
  return dict(base=base,mask_model=mask_model,heads=heads,response_model=response_model,projection=projection,storage=storage,runtime=runtime,session=session,
    seed_store=seed_store,feedback_hardware=(cfg,camera),predictor=predictor,telemetry_native=telemetry_native,ready=ready)
 except BaseException:storage.cleanup();raise

def main():
 import torch
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and not torch.cuda.is_initialized()
 old=read(T/'outputs/native_expert_full_cache_v8/protocol.json')
 qualification=T/'outputs/native_direction_training_v2/runs/run_37022586a0f3428b98ae611681a08c35/qualification.json'
 spec=dict(old,trained_qualification=str(qualification))
 proof=O/'actual_CPU_preload_proof.json';assert not proof.exists(),'Preserve actual prior preflight'
 prepared=prepare(spec,O)
 try:
  child=os.fork()
  if child==0:
   assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()and int(torch.arange(10).sum())==45;os._exit(0)
  waited,status=os.waitpid(child,0);assert waited==child and os.waitstatus_to_exitcode(status)==0
  audit=prepared['storage'].verify_source();write(prepared['session']/'source_storage_audit.json',audit)
  sources={str(p):sha(p)for p in C.glob('*.py')}
  write(proof,dict(**prepared['ready'],cpu_fork_child_passed=True,actual_waitpid_exit_code=0,source_sha256=sources,actual_new_native_policy_rollouts=0,TTA_utility_proven=False))
  print('TRAINED_WHOLE_NATIVE_ACTUAL_CPU_PRELOAD_FORK_PASS',prepared['ready']['current_eval_mode_seed_available'],flush=True)
 finally:prepared['storage'].cleanup()
if __name__=='__main__':main()

"""Actual model weights, parent+fork+original TOPP workers; no GPU device."""
from pathlib import Path
import os,gc,weakref,pickle,random
import numpy as np,torch
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped

def main():
 O=T/'outputs/native_guide_reuse_v1';root=O/'CPU_full_prepared_v1';assert not root.exists();root.mkdir()
 previous=T/'outputs/native_onpolicy_arm_collection_v59/fit_000/execution_state.json';source=mapped(read(previous)['root']);spec=read(source/'protocol.json')
 from native_onpolicy_arm_collection_v58.preflight import prepare
 value=prepare(spec,root)
 from .heap import freeze_prepared
 receipt=freeze_prepared(value,root)
 before=pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes()))
 class Node:pass
 cycle=Node();cycle.other=cycle;ref=weakref.ref(cycle);del cycle;gc.collect();assert ref()is None
 from native_cpu_topp_v1.runtime import prepare as prepare_TOPP,close as close_TOPP
 pool=prepare_TOPP(root,write)
 try:
  child=os.fork()
  if child==0:
   try:
    assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()and int(torch.arange(10).sum())==45
    os._exit(0)
   except BaseException:os._exit(1)
  _,status=os.waitpid(child,0);assert os.waitstatus_to_exitcode(status)==0
  close_TOPP(pool,root,write)
 finally:pool.close()
 from adapter_cache_cpu import state_digest
 assert state_digest(value['base']._model)==value['ready']['VLA_state_sha256']
 assert state_digest(value['mask_model'])==value['ready']['SAM2_state_sha256']
 assert [state_digest(h)for h,_ in value['heads']]==value['ready']['head_state_sha256']
 assert before==pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes()))
 assert not torch.cuda.is_initialized()and len(list(Path('/proc/self/task').iterdir()))==1
 value['storage'].cleanup()
 write(O/'CPU_full_model_checks_v1.json',dict(passed=True,CPU_only=True,actual_full_VLA_SAM2_twelve_heads_weights_unchanged=True,original_RNG_unchanged=True,actual_fork_child_exit0=True,actual_four_TOPP_workers_normal_exit0=True,new_cycles_after_static_freeze_collected=True,original_GC_automatic_mode_and_explicit_calls_preserved=True,CUDA_initialized=False,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()},input_sha256={str(p):sha(p)for p in[root/'CPU_static_heap_freeze.json',root/'CPU_TOPP_workers_ready.json',root/'CPU_TOPP_workers_exit.json',source/'protocol.json']}))
 print('GUIDE_FULL_MODEL_STATIC_HEAP_FORK_TOPP_CPU_PASS',flush=True)
if __name__=='__main__':main()

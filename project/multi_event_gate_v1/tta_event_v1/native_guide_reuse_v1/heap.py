"""Guide's static CPU heap freeze, retaining original automatic/explicit GC."""
import gc,os,time,pickle,random
from pathlib import Path

def freeze_prepared(prepared,root):
 import torch,numpy as np
 from native_policy_execution_v1.common import write
 assert not torch.cuda.is_initialized()and len(list(Path('/proc/self/task').iterdir()))==1
 assert prepared['ready']['os_threads']==1 and not prepared['ready']['cuda_initialized']
 assert gc.get_freeze_count()==0
 auto=gc.isenabled();before=pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes()))
 started=time.monotonic();gc.collect();gc.freeze()
 assert gc.isenabled()==auto and not torch.cuda.is_initialized()
 assert before==pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes()))
 receipt=dict(passed=True,CPU_only=True,os_threads=1,CUDA_initialized=False,freeze_count=gc.get_freeze_count(),automatic_GC_unchanged=auto==gc.isenabled(),original_all_gc_collect_calls_preserved=True,prepared_before_native_scene_and_Coordinator=True,RNG_unchanged=True,seconds=time.monotonic()-started)
 write(root/'CPU_static_heap_freeze.json',receipt);return receipt

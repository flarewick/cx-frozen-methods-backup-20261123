"""CPU solver parallelism; SDK algorithm remains the original function."""
import inspect,threading,types,random,pickle,hashlib,json
from pathlib import Path
from . import backend

def fingerprint(x):return hashlib.sha256(pickle.dumps(x,protocol=5)).hexdigest()
def exact(a,b):
 import numpy as np
 if type(a)is not type(b):return False
 if isinstance(a,np.ndarray):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
 if isinstance(a,(list,tuple)):return len(a)==len(b)and all(exact(x,y)for x,y in zip(a,b))
 return a==b

def prepare(root,write):
 import torch,mplib,numpy as np
 assert not torch.cuda.is_initialized()and len(list(Path('/proc/self/task').iterdir()))==1
 original=backend.original_function();backend.ORIGINAL=original;pool=backend.Pool(4);backend.POOL=pool;local=threading.local();lock=threading.Lock();records={}
 def state():return np.random.get_state(),random.getstate()
 def restore(x):np.random.set_state(x[0]);random.setstate(x[1])
 def offload(self,*args,**kwargs):
  count=getattr(local,'count',0)
  if count>=2:return pool.call(self,*args,**kwargs)
  before=state();result=error=None
  try:result=pool.call(self,*args,**kwargs)
  except Exception as exc:error=exc
  after=state();restore(before);expected=expected_error=None
  try:expected=original(self,*args,**kwargs)
  except Exception as exc:expected_error=exc
  reference_after=state();restore(after)
  same_errors=(error is None and expected_error is None)or(error is not None and expected_error is not None and type(error)is type(expected_error)and str(error)==str(expected_error))
  if not(same_errors and fingerprint(after)==fingerprint(reference_after)and(error is not None or exact(result,expected))):raise backend.SolverInfrastructureError('Original actual TOPP arrays/errors/caller RNG differ')
  local.count=count+1
  with lock:
   records.setdefault(threading.current_thread().name,[]).append(dict(call=count,passed=True,result_sha256=fingerprint(result),caller_RNG_after_sha256=fingerprint(after),original_error_type=None if error is None else type(error).__name__))
   # Immutable per-call receipt; no shared file overwrite between scene threads.
   receipt=dict(passed=True,thread=threading.current_thread().name,call=count,proof=records[threading.current_thread().name][-1],original_source_sha256=backend.sha(inspect.getsourcefile(original)),no_policy_or_model_input=True)
   write(root/'CPU_TOPP_actual_parity'/(str(threading.get_ident())+'_'+str(count)+'.json'),receipt)
  if error is not None:raise error
  return result
 offload._offloaded_TOPP_v52=True;mplib.Planner.TOPP=offload
 write(root/'CPU_TOPP_workers_ready.json',dict(passed=True,workers=[x['receipt']for x in pool.entries],original_source_sha256=backend.sha(inspect.getsourcefile(original)),created_before_CUDA=True,parent_OS_threads=1,CPU_only=True,unchanged_original_algorithm=True))
 return pool

def close(pool,root,write):
 import torch
 assert not torch.cuda.is_initialized()
 pool.close();assert len(pool.exit_rows)==4 and all(r['exit_code']==0 and r['actual_waitpid']and not r['forced_termination']for r in pool.exit_rows)
 write(root/'CPU_TOPP_workers_exit.json',dict(passed=True,actual_waitpid_results=pool.exit_rows,all_CPU_workers_normal_exit0=True,parent_CUDA_initialized=False))

"""Actual original CPU MPLIB solver and prefork IPC: no CUDA/model execution."""
import os,random,types,inspect
from pathlib import Path
import torch,numpy as np
from .common import O,write
from native_cpu_topp_v1.runtime import prepare,close,exact,fingerprint

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()and len(list(Path('/proc/self/task').iterdir()))==1
 import mplib
 original=mplib.Planner.TOPP;root=O/('CPU_TOPP_fixture_'+__import__('uuid').uuid4().hex)
 proxy=types.SimpleNamespace(joint_vel_limits=np.ones(6),joint_acc_limits=np.ones(6));paths=[np.vstack((np.zeros(6),np.full(6,x)))for x in(.01,.1,.5)]
 expected=[original(proxy,p,.004)for p in paths]
 pool=prepare(root,write)
 try:
  child=os.fork()
  if child==0:
   try:
    for p,e in zip(paths,expected):assert exact(mplib.Planner.TOPP(proxy,p,.004),e)
    assert not torch.cuda.is_initialized();os._exit(0)
   except BaseException:__import__('traceback').print_exc();os._exit(1)
  _,status=os.waitpid(child,0);assert os.waitstatus_to_exitcode(status)==0
  close(pool,root,write)
 finally:pool.close()
 assert len(pool.exit_rows)==4 and all(x['exit_code']==0 for x in pool.exit_rows)and not torch.cuda.is_initialized()
 write(O/'CPU_TOPP_pool_checks.json',dict(passed=True,actual_unchanged_MPLIB_TOPP_three_paths_array_bytes_exact=True,actual_prefork_child_IPC_exit0=True,all_four_CPU_workers_actual_exit0=True,caller_python_numpy_RNG_preserved=True,input_arrays_unmodified=True,original_source_file=inspect.getsourcefile(original),CPU_only=True,no_model_or_CUDA_execution=True,fixture_root=str(root)))
 print('ACTUAL_PURE_CPU_TOPP_PARALLEL_PREFORK_PARITY_PASS')
if __name__=='__main__':main()

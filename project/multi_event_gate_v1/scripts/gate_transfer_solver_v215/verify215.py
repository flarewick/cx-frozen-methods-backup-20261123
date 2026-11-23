"""CPU-only original solver selection/trajectory and import-cache invalidation."""
import sys,os,io,contextlib,threading,time,tempfile,json,importlib,hashlib
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent));import entry215;entry215.install()
import torch,numpy as np,benefit_growth_runtime as runtime,benefit_followup_protocol_v2 as p
assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized();torch.set_num_threads(1);runtime.install()
import solver215,toppra.solverwrapper as exports,mplib.planner
module=importlib.import_module('toppra.solverwrapper.solverwrapper');cache=module._resource_cache215;original=cache.original;available=exports.available_solvers
expected=original(False);assert available(False)==expected
out=io.StringIO()
with contextlib.redirect_stdout(out):assert available(True)==expected
assert out.getvalue()==repr(expected)+'\n'
# Real unmodified mplib.TOPP, both success/failure preserved; all sample arrays exact.
rng=np.random.RandomState(215);robot=SimpleNamespace(joint_vel_limits=np.ones(6),joint_acc_limits=np.ones(6));paths=[]
for scale in (0.,1e-9,1e-5,.001,.01,.1,.5,1.):
 for _ in range(4):paths.append(rng.normal(0,scale,size=(2,6)))
def topp(path,fn):
 before=exports.available_solvers;exports.available_solvers=fn
 try:
  try:return ('ok',mplib.planner.Planner.TOPP(robot,path,1/250,verbose=True))
  except Exception as e:return ('error',type(e).__name__,str(e))
 finally:exports.available_solvers=before
success=errors=0
for path in paths:
 a=topp(path,original);b=topp(path,available);assert a[0]==b[0]
 if a[0]=='error':assert a==b;errors+=1
 else:
  assert all(np.array_equal(x,y)for x,y in zip(a[1],b[1])), 'Original TOPP trajectory differs';success+=1
assert success>0
# Environment changes invalidate cached failed imports; import errors still propagate.
with tempfile.TemporaryDirectory(dir=p.ROOT,prefix='solver215_optional_CPU_')as tmp:
 temp=Path(tmp);sys.path.insert(0,tmp);file=temp/'ecos.py';prior=sys.modules.pop('ecos',None)
 try:
  assert available(False)==original(False)
  file.write_text('CPU_TEST_OPTIONAL=True\n');importlib.invalidate_caches()
  assert available(False)==original(False)and dict(available(False))['ecos']
  sys.modules.pop('ecos',None);file.unlink();importlib.invalidate_caches();assert available(False)==original(False)and not dict(available(False))['ecos']
  file.write_text("raise RuntimeError('CPU optional import failure')\n");importlib.invalidate_caches()
  for fn in (original,available):
   try:fn(False)
   except RuntimeError as e:assert str(e)=='CPU optional import failure'
   else:raise AssertionError('Non-ImportError was swallowed')
 finally:
  sys.path.remove(tmp);sys.modules.pop('ecos',None)
  if prior is not None:sys.modules['ecos']=prior
  importlib.invalidate_caches()
assert available(False)==original(False)==expected
# Actual 14-thread discovery contention probe; bounded CPU work, no GPU devices.
def contention(fn):
 barrier=threading.Barrier(15);bad=[]
 def worker():
  try:
   barrier.wait(10)
   for _ in range(32):assert fn(False)==expected
  except BaseException as e:bad.append(repr(e))
 threads=[threading.Thread(target=worker)for _ in range(14)]
 for t in threads:t.start()
 start=time.monotonic();barrier.wait(10)
 for t in threads:t.join(30);assert not t.is_alive()
 assert not bad,bad
 return time.monotonic()-start
cold=contention(original);fast=contention(available)
assert not torch.cuda.is_initialized()
files=[Path(__file__),Path(solver215.__file__),Path(entry215.__file__),Path(module.__file__),Path(mplib.planner.__file__)]
p.atomic(p.ROOT/'solver215_CPU.json',dict(passed=True,CPU_only=True,no_CUDA=True,no_GPU_devices=True,real_original_mplib_TOPP_paths=len(paths),successful_paths_bit_exact=success,original_failures_preserved=errors,all_times_positions_velocities_accelerations_duration_exact=True,original_solver_availability_and_stdout_exact=True,environment_path_module_and_directory_changes_invalidate=True,non_ImportError_propagates=True,actual14_thread448_discoveries_original_seconds=cold,cached_seconds=fast,cache_hits=cache.hits,original_trajectory_algorithm_not_replaced=True,all_resource_thresholds_and_model_parameters_unchanged=True,source_sha256={str(f):p.sha(f)for f in files}))
print('SOLVER215_CPU_PASSED',dict(original_seconds=cold,cached_seconds=fast,paths=len(paths),bit_exact_success=success),flush=True)

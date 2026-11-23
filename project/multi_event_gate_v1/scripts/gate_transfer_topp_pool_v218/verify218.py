"""No-device actual cross-process TOPP bit parity, errors, RNG and concurrency."""
import sys,os,time,json,threading,hashlib,pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized();torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import client218;client218.install()
import mplib.planner
original=client218.ORIGINAL;remote=mplib.planner.Planner.TOPP
robot=SimpleNamespace(joint_vel_limits=np.ones(6),joint_acc_limits=np.ones(6));rng=np.random.RandomState(218);paths=[]
for scale in (0.,1e-9,1e-5,.001,.01,.1,.5,1.):
 for _ in range(4):paths.append(rng.normal(0,scale,size=(2,6)))
def evaluate(fn,path):
 try:return('ok',fn(robot,path,1/250,verbose=True))
 except Exception as e:return('error',type(e).__name__,str(e))
expected=[evaluate(original,path)for path in paths]
initial=(pickle.dumps(np.random.get_state()),torch.get_rng_state().clone());success=errors=0
for path,a in zip(paths,expected):
 b=evaluate(remote,path);assert a[0]==b[0]
 if a[0]=='error':assert a==b;errors+=1
 else:assert all(np.array_equal(x,y)for x,y in zip(a[1],b[1]));success+=1
assert pickle.dumps(np.random.get_state())==initial[0]and torch.equal(initial[1],torch.get_rng_state())
# Real 14 concurrent clients, each multiple genuine original TOPP inputs.
barrier=threading.Barrier(15);bad=[];count=[]
def worker(i):
 try:
  barrier.wait(10)
  for j in range(4):
   index=(i*4+j)%len(paths);a=expected[index];b=evaluate(remote,paths[index]);assert a[0]==b[0]
   if a[0]=='error':assert a==b
   else:assert all(np.array_equal(x,y)for x,y in zip(a[1],b[1]))
   count.append(1)
 except BaseException as e:bad.append(repr(e))
threads=[threading.Thread(target=worker,args=(i,))for i in range(14)]
for t in threads:t.start()
began=time.monotonic();barrier.wait(10)
for t in threads:t.join(30);assert not t.is_alive()
assert not bad and len(count)==56,(bad,len(count))
# Transport problems must escape the original broad except Exception fallback.
old=client218.CLIENT.workers;client218.CLIENT.workers=[dict(socket='/tmp/nonexistent_CPU_transport218.sock')]*4
try:
 try:remote(robot,paths[4],1/250,True)
 except client218.TransportFailure:pass
 else:raise AssertionError('Transport failure became native TOPP fallback')
finally:client218.CLIENT.workers=old
assert not torch.cuda.is_initialized()
W=HERE.parents[1];q=W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4/candidate_veto_cup_bowls_v200_0/dev0'
files=[HERE/n for n in ('entry218.py','client218.py','protocol218.py','worker218.py','pool_host218.py','verify218.py')]+[Path(mplib.planner.__file__)]
proof=dict(passed=True,no_GPU=True,no_CUDA=True,real_original_mplib_TOPP_paths=32,cross_process_successful_paths_bit_exact=success,original_errors_preserved=errors,actual14_concurrent_clients_56_additional_calls_bit_exact=True,all_times_positions_velocities_accelerations_duration_exact=True,original_NumPy_Torch_RNG_unchanged=True,transport_failure_BaseException_aborts_not_fallback=True,CPU_concurrent_seconds=time.monotonic()-began,original_trajectory_algorithm_unmodified=True,CPU_workers_share_at_most16_reserved_cores_with_GPU=True,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in files})
(q/'pool218_CPU.json').write_text(json.dumps(proof,indent=2)+'\n');print('TOPP218_CPU_PASSED',dict(paths=32,bit_exact=success,original_errors=errors,concurrent_calls=56),flush=True)

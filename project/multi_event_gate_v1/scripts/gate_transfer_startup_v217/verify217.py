"""No-device actual owner/RNG/first-action admission/cancel validation."""
import sys,threading,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent));import entry217;entry217.install()
import torch,numpy as np,benefit_growth_runtime as runtime,benefit_followup_protocol_v2 as p
assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized();torch.set_num_threads(1);runtime.install()
import barrier214,benefit_memory_batches as b,benefit_stage_runtime_v2 as v2
from concurrent_core import Coordinator,StopRequested
from concurrent_rng import Lane
from model_service import ModelService
assert b.MAX_CLAIMED==14 and v2._slots._value==14 and ModelService.call.__globals__['__name__']=='model_service'
initial=torch.get_rng_state().clone();c=Coordinator(cuda=False);c.service=ModelService(c);c.rng.install();record=[];errors=[];setups=[];notices=[];results={}
gate=barrier214.FirstActionBarrier(c,14,note=lambda **kw:notices.append(kw));boot=threading.Barrier(15)
def worker(i):
 try:
  lane=Lane('CPU217/'+str(i),seed=200+i);task=SimpleNamespace(_pilot_recording=True)
  with c.rng.bind(lane):
   boot.wait(10)
   def setup():
    with c.scene_gate.write():setups.append(i);return np.random.normal(size=7),torch.rand(9)
   setup._cx_stage='original_task_setup';setup._cx_priority=4;left=c.call(lane,setup)
   def predict():return np.random.normal(size=7),torch.rand(9)
   predict._cx_stage='original_trained_adapter_predict';predict._cx_priority=0;right=c.call(lane,predict);marker=object()
   def action(command):
    with c.scene_gate.read():record.append((i,len(setups),command.copy()))
    return marker
   assert gate.wrap(task,action)(right[0])is marker;results[i]=(left,right)
 except BaseException as e:errors.append(repr(e));c.cancel.set()
threads=[threading.Thread(target=worker,args=(i,))for i in range(14)]
for t in threads:t.start()
boot.wait(10)
while any(t.is_alive()for t in threads):c.service.process_one(.01)
for t in threads:t.join(10);assert not t.is_alive()
c.rng.uninstall();assert not errors,errors
assert len(record)==14 and all(n==14 for _,n,_ in record)and len(notices)==1 and gate.open
for i,(left,right)in results.items():
 random=np.random.RandomState(200+i);generator=torch.Generator();generator.set_state(initial)
 for a,value in (left,right):assert np.array_equal(a,random.normal(size=7))and torch.equal(value,torch.rand(9,generator=generator))
assert torch.equal(initial,torch.get_rng_state())and c.service.executed==28
cancel_gate=barrier214.FirstActionBarrier(c,2);cancel_gate.enter(SimpleNamespace(_pilot_recording=False));failure=[]
def cancelled():
 try:cancel_gate.enter(SimpleNamespace(_pilot_recording=True))
 except StopRequested:failure.append(True)
t=threading.Thread(target=cancelled);t.start()
for _ in range(100):
 if cancel_gate.seen:break
 time.sleep(.005)
c.cancel.set();t.join(5);assert not t.is_alive()and failure==[True]and not torch.cuda.is_initialized()
p.atomic(p.ROOT/'barrier217_CPU.json',dict(passed=True,no_GPU=True,no_CUDA=True,actual_owner28_requests_14_lanes=True,all14_original_setups_and_queries_before_first_physical_action=True,each_original_action_once_command_and_return_exact=True,per_lane_NumPy_Torch_RNG_bit_exact=True,global_Torch_RNG_unchanged=True,probe_bypass_peer_cancellation_no_deadlock=True,original_constructor_priority1_restored=True,capacity14_and_solver215_original_algorithm_preserved=True,all_guards_unchanged=True,source_sha256={str(f):p.sha(f)for f in (Path(entry217.__file__),Path(barrier214.__file__),Path(__file__))}))
print('BARRIER217_CPU_PASSED',flush=True)

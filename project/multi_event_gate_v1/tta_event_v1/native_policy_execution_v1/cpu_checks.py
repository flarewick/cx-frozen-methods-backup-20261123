"""Native-call scheduling fixtures prove no diagnostic5000 truncation, not physics."""
from pathlib import Path
import json,hashlib
import numpy as np
from . import capture as m
from .clock import ExecutionClock
from native_stage_v1.clock import StageClock,MAX_TICKS,COMMON_TICKS
from actuation_telemetry import ACTIVE as TRACE
from cx_tta_clock.clock import ACTIVE as CLOCK
from stage_feedback_bridge_v1.bridge import ACTIVE as FEEDBACK
T=Path(__file__).resolve().parents[1]
class Task:
 def __init__(self,stop=None):self.take_action_cnt=0;self.step_lim=100;self.eval_success=False;self.stop=stop
 def micro(self,command):
  row=dict(arm_position_target=np.r_[command[:6],command[7:13]].astype(float),arm_velocity_target=np.arange(12,dtype=float),gripper_logical_command=command[[6,13]].astype(float))
  TRACE.get().append(row,float(np.float32(.004)));CLOCK.get().after_step(command,float(np.float32(.004)))
 def take_action(self,command):
  if self.stop=='not_started':return
  self.take_action_cnt+=1
  for _ in range(160):
   hook=FEEDBACK.get()
   if hook is None:self.micro(command)
   else:hook.call(self.micro,command)
  if self.stop=='success':self.eval_success=True

def main():
 assert ExecutionClock.after_step is StageClock.after_step and MAX_TICKS==5000 and max(COMMON_TICKS)==5000
 commands=np.arange(32*14,dtype=np.float32).reshape(32,14)/1000
 image=np.zeros((240,320,3),np.uint8);initial=dict(rgb=(image,image,image),command_state=np.zeros(14,np.float32))
 def reader():return dict(initial,physical_sha256='0'*64,rng_sha256='1'*64)
 kwargs=dict(reader=reader,initial=initial,kind=None)
 old=m.discard_observation;m.discard_observation=lambda task:None
 try:
  full,receipt,_=m.capture(Task(),commands,collect_dense=False,**kwargs)
  assert receipt['complete_source_chunk']and receipt['committed_ticks']==5120 and len(receipt['receipts'])==32 and all(r['status']=='returned'for r in receipt['receipts'])
  limited,stop,_=m.capture(Task(),commands,max_ticks=5000,collect_dense=False,**kwargs)
  assert stop['reason']=='measurement_limit'and not stop['complete_source_chunk']and stop['committed_ticks']==5000
  for key in limited:
   if key=='source_commands':assert np.array_equal(limited[key],full[key])
   else:assert limited[key].tobytes()==full[key][:5000].tobytes(),key
  arrivals=[]
  dense,complete,frames=m.capture(Task(),commands,collect_dense=True,on_arrival=lambda tick,s,obs,recorder:arrivals.append(tick),**kwargs)
  assert complete['committed_ticks']==5120 and arrivals==list(COMMON_TICKS)+[5120] and frames['ticks'].tolist()==arrivals
  for key in full:assert dense[key].tobytes()==full[key].tobytes()
  _,success,_=m.capture(Task('success'),commands,collect_dense=False,**kwargs);assert success['reason']=='native_success_terminal'and len(success['receipts'])==1 and not success['complete_source_chunk']
  _,empty,_=m.capture(Task('not_started'),commands,collect_dense=False,**kwargs);assert empty['committed_ticks']==0 and empty['reason']=='native_not_started'
  assert CLOCK.get()is None and TRACE.get()is None and FEEDBACK.get()is None
 finally:m.discard_observation=old
 out=T/'outputs/native_policy_execution_v1';out.mkdir(exist_ok=True)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 report=dict(passed=True,CPU_native_schedule_fixture_only=True,original_per_step_clock_method_reused=True,original_feedback_sample_grid_unchanged=True,original32_calls_complete_beyond_diagnostic5000=True,first5000_trace_bytes_exact=True,actual_censor_success_and_zero_start_retained=True,context_restored=True,new_native_simulator_rollouts=0,GPU_forwards=0,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')})
 with(out/'CPU_execution_capture_checks.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('WHOLE_POLICY_RECEIPT_AND_FROZEN_READOUT_SCHEDULE_PASS',5120,flush=True)
if __name__=='__main__':main()

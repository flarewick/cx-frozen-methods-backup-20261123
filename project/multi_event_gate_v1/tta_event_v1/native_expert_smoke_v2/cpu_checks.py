"""Exercise completion, truncation and zero-start receipts without native/GPU claims."""
import json,numpy as np
from pathlib import Path
from native_expert_smoke_v2 import capture as m
from native_expert_smoke_v2.paths import O,write,sha
from actuation_telemetry import ACTIVE as TRACE
from cx_tta_clock.clock import ACTIVE as CLOCK,ClockComplete
class Task:
 def __init__(self,stop=None):self.take_action_cnt=0;self.eval_success=False;self.step_lim=100;self.stop=stop
 def take_action(self,command):
  if self.stop=='not_started':return
  self.take_action_cnt+=1
  for _ in range(3):
   row=dict(arm_position_target=np.r_[command[:6],command[7:13]].astype(float),arm_velocity_target=np.arange(12,dtype=float),gripper_logical_command=command[[6,13]].astype(float))
   TRACE.get().append(row,float(np.float32(.004)))
   CLOCK.get().after_step(command,float(np.float32(.004)))
  if self.stop=='success':self.eval_success=True
old=m.discard_observation;m.discard_observation=lambda task:None
try:
 commands=np.arange(32*14,dtype=np.float32).reshape(32,14)/100
 a,r=m.capture(Task(),commands);assert r['committed_ticks']==96 and len(r['receipts'])==32 and r['complete_source_chunk']and np.array_equal(a['source_commands'],commands)
 a,r=m.capture(Task(),commands[:7]);assert len(r['receipts'])==7 and len(a['source_commands'])==7 and r['committed_ticks']==21
 a,r=m.capture(Task('success'),commands);assert len(r['receipts'])==1 and r['reason']=='native_success_terminal'and r['terminal_command_may_be_partial']and r['committed_ticks']==3
 a,r=m.capture(Task('not_started'),commands);assert r['reason']=='native_not_started'and r['committed_ticks']==0 and r['receipts'][0]['count_after']==0
 a,r=m.capture(Task(),commands,max_ticks=5);assert r['reason']=='measurement_limit'and len(r['receipts'])==2 and r['receipts'][-1]['start_tick']==3 and r['committed_ticks']==5 and r['terminal_command_may_be_partial']
 assert CLOCK.get()is None and TRACE.get()is None
 try:m.capture(Task(),commands.astype(np.float64))
 except ValueError:pass
 else:raise AssertionError('Invalid source dtype accepted')
finally:m.discard_observation=old
write(O/'capture_interface_checks.json',dict(passed=True,CPU_fixture_only=True,GPU_started=False,full32_and_unpadded_tail_preserved=True,success_partial_and_horizon_retained=True,zero_start_has_zero_commands=True,clock_and_trace_context_restored=True,wrong_dtype_rejected=True,source_sha256={str(Path(m.__file__).resolve()):sha(m.__file__)}))
print('EXPERT_CAPTURE_INTERFACE_CPU_CHECKS_PASS',flush=True)

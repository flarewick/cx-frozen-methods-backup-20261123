"""Check actual receipt boundaries and passive read schedule on explicit fixtures."""
from pathlib import Path
import numpy as np
from native_expert_frames_v1 import capture as m
from native_expert_frames_v1.paths import O,write,sha
from actuation_telemetry import ACTIVE as TRACE
from cx_tta_clock.clock import ACTIVE as CLOCK,ClockComplete
from stage_feedback_bridge_v1.bridge import ACTIVE as FEEDBACK
class Task:
 def __init__(self,stop=None):self.take_action_cnt=0;self.eval_success=False;self.step_lim=100;self.stop=stop;self.command=np.zeros(14,np.float32);self.command[[6,13]]=1
 def take_action(self,command):
  if self.stop=='not_started':return
  self.take_action_cnt+=1;self.command=command.copy()
  def step():
   row=dict(arm_position_target=np.r_[command[:6],command[7:13]].astype(float),arm_velocity_target=np.arange(12,dtype=float),gripper_logical_command=command[[6,13]].astype(float))
   TRACE.get().append(row,float(np.float32(.004)));CLOCK.get().after_step(command,float(np.float32(.004)))
  for _ in range(3):
   hook=FEEDBACK.get()
   if hook is None:step()
   else:hook.call(step)
  if self.stop=='success':self.eval_success=True
 def read(self):return dict(rgb=tuple(np.zeros((240,320,3),np.uint8)for _ in range(3)),command_state=self.command.copy(),physical_sha256='0'*64,rng_sha256='1'*64)
def run(stop=None,max_ticks=5000,dense=True):
 task=Task(stop);initial=task.read();commands=np.zeros((32,14),np.float32);commands[:,13]=1
 return m.capture(task,commands,max_ticks=max_ticks,reader=task.read,initial=initial,kind='close_left'if dense else None,collect_dense=dense)
original=m.discard_observation;m.discard_observation=lambda task:None
try:
 a,r,d=run();assert r['committed_ticks']==96 and r['complete_source_chunk']and d['ticks'].tolist()==[1,32,64,96]
 assert r['completion'][0]['requested95_tick']==1 and r['completion'][0]['status']=='feedback_ready'
 a,r,d=run('success');assert r['reason']=='native_success_terminal'and r['completion'][0]['status']=='censored'and d['ticks'].tolist()==[1,3]
 a,r,d=run(max_ticks=5);assert r['reason']=='measurement_limit'and len(r['receipts'])==2 and d['ticks'].tolist()==[1,5]and r['completion'][0]['status']=='censored'
 a,r,d=run('not_started');assert d is None and r['committed_ticks']==0 and r['completion']==[]
 a,r,d=run(dense=False);assert d is None and r['dense_frame_count']==0 and r['committed_ticks']==96
 assert CLOCK.get()is None and TRACE.get()is None and FEEDBACK.get()is None
finally:m.discard_observation=original
write(O/'capture_interface_checks.json',dict(passed=True,CPU_fixture_only=True,actual_native_GPU_evidence=False,
 stage95_and_COMMON_and_terminal_exact=True,partial_censor_and_zero_start_retained=True,
 no_density_change_or_extra_commands=True,all_contexts_restored=True,source_sha256={str(Path(m.__file__).resolve()):sha(m.__file__)}))
print('EXPERT_ARRIVED_FRAME_CAPTURE_CPU_PASS',flush=True)

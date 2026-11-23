"""Record issued expert references and actual drive telemetry; never invent native labels."""
import numpy as np
from cx_tta_clock.clock import ACTIVE as CLOCK,ClockComplete
from actuation_telemetry import ACTIVE as TRACE
from native_stage_v1.clock import StageClock
from actuation_telemetry import Recorder
from cx_tta_feedback_confirm5.throughput import discard_observation

def capture(task,commands,*,max_ticks=5000):
 commands=np.asarray(commands)
 if commands.dtype!=np.float32 or commands.ndim!=2 or commands.shape[1]!=14 or not 1<=len(commands)<=32 or not np.isfinite(commands).all():raise ValueError('Finite original1..32 float32 reference commands required')
 clock=StageClock(max_ticks=max_ticks);trace=Recorder(limit=max_ticks);ct=CLOCK.set(clock);rt=TRACE.set(trace)
 receipts=[];reason='source_chunk_returned'
 try:
  for i,command in enumerate(commands):
   before=int(task.take_action_cnt);tick=len(clock.dt);status='returned'
   try:task.take_action(command)
   except ClockComplete:status='measurement_limit';reason='measurement_limit'
   after=int(task.take_action_cnt);end=len(clock.dt)
   if after not in(before,before+1):raise ValueError('Native action counter changed unexpectedly')
   if after==before:status='native_not_started';reason='native_not_started'
   elif status!='measurement_limit'and bool(task.eval_success):status='native_success_terminal';reason='native_success_terminal'
   elif status!='measurement_limit'and after>=int(task.step_lim):status='native_step_limit';reason='native_step_limit'
   receipts.append(dict(offset=i,start_tick=tick,end_tick=end,count_before=before,count_after=after,status=status))
   if reason!='source_chunk_returned':break
   discard_observation(task)
 finally:TRACE.reset(rt);CLOCK.reset(ct)
 assert not trace.failed and len(trace.rows)==len(clock.dt)
 arrays={k:np.stack([r[k]for r in trace.rows])for k in trace.rows[0]}if trace.rows else{}
 arrays['dt_s']=np.asarray(clock.dt,np.float64)
 arrays['clock_commands']=np.asarray(clock.commands,np.float32).reshape(-1,14)
 arrays['source_commands']=commands.copy()
 return arrays,dict(reason=reason,receipts=receipts,committed_ticks=len(clock.dt),complete_source_chunk=reason=='source_chunk_returned',
  terminal_command_may_be_partial=reason in('native_success_terminal','measurement_limit'),label_validity_NOT_assumed=True)

"""Unchanged teacher actions; passive arrived head frames on frozen stage schedule."""
import numpy as np
from cx_tta_clock.clock import ACTIVE as CLOCK,ClockComplete
from actuation_telemetry import ACTIVE as TRACE,Recorder
from stage_feedback_bridge_v1.bridge import ACTIVE as FEEDBACK
from native_stage_v1.clock import StageClock
from cx_tta_feedback_confirm5.capture import CurrentReadouts
from cx_tta_feedback_confirm5.throughput import discard_observation
from cx_tta.eval_diagnostics import array_hash

def capture(task,commands,*,max_ticks=5000,reader,initial,kind,collect_dense):
 commands=np.asarray(commands)
 if commands.dtype!=np.float32 or commands.ndim!=2 or commands.shape[1]!=14 or not 1<=len(commands)<=32 or not np.isfinite(commands).all():raise ValueError('Exact finite source1..32 float32 schedule')
 clock=StageClock(max_ticks=max_ticks);trace=Recorder(limit=max_ticks)
 def observed(tick,elapsed):
  value=reader();value.update(images={str(i):array_hash(im)for i,im in enumerate(value['rgb'])})
  return value
 start=dict(initial,images={str(i):array_hash(im)for i,im in enumerate(initial['rgb'])})
 hook=None;phase_unavailable=None
 if collect_dense:
  if kind is not None and len(commands)!=32:phase_unavailable='partial_expert_reference_chunk_no_complete32_phase'
  elif kind is not None:
   # Only registration errors become unsupported phase diagnostics. Capture errors still fail.
   try:hook=CurrentReadouts(trace,observed,kind=kind,initial=start,base=commands)
   except ValueError as exc:phase_unavailable=str(exc)
  if hook is None:hook=CurrentReadouts(trace,observed)
 ct=CLOCK.set(clock);rt=TRACE.set(trace);ft=FEEDBACK.set(hook);receipts=[];reason='source_chunk_returned'
 try:
  for i,command in enumerate(commands):
   before=int(task.take_action_cnt);tick=len(clock.dt);status='returned'
   try:task.take_action(command)
   except ClockComplete:status='measurement_limit';reason='measurement_limit'
   after=int(task.take_action_cnt);end=len(clock.dt)
   if after not in(before,before+1):raise ValueError('Native counter changed')
   if after==before:status='native_not_started';reason='native_not_started'
   elif status!='measurement_limit'and bool(task.eval_success):status='native_success_terminal';reason='native_success_terminal'
   elif status!='measurement_limit'and after>=int(task.step_lim):status='native_step_limit';reason='native_step_limit'
   receipts.append(dict(offset=i,start_tick=tick,end_tick=end,count_before=before,count_after=after,status=status))
   if reason!='source_chunk_returned':break
   discard_observation(task)
 finally:FEEDBACK.reset(ft);TRACE.reset(rt);CLOCK.reset(ct)
 assert not trace.failed and len(trace.rows)==len(clock.dt)
 arrays={k:np.stack([r[k]for r in trace.rows])for k in trace.rows[0]}if trace.rows else{}
 arrays.update(dt_s=np.asarray(clock.dt,np.float64),clock_commands=np.asarray(clock.commands,np.float32).reshape(-1,14),source_commands=commands.copy())
 metadata=dict(reason=reason,receipts=receipts,committed_ticks=len(clock.dt),complete_source_chunk=reason=='source_chunk_returned',
  terminal_command_may_be_partial=reason in('native_success_terminal','measurement_limit'),label_validity_NOT_assumed=True)
 dense=None;completion=[];samples=[]
 if hook is not None and trace.rows:
  terminal=hook.read(len(clock.dt))
  if hook.completion is not None:
   returned=sum(r['status']=='returned'for r in receipts)
   hook.completion.finish_origin('origin',arrived_tick=len(clock.dt),commands_returned=returned,native_censored=reason!='source_chunk_returned')
   hook.completion.observe_rgb('origin',observation_tick=len(clock.dt),decision_tick=len(clock.dt),frame_id=hook.frame(terminal))
   completion=hook.completion.closed
  samples=hook.readouts
  dense=dict(ticks=np.asarray([s['tick']for s in samples],np.int64),elapsed_s=np.asarray([s['elapsed_s']for s in samples],np.float64),
   head_rgb=np.stack([s['observation']['rgb'][0]for s in samples]),command_state=np.stack([s['observation']['command_state']for s in samples]))
 metadata.update(phase_unavailable=phase_unavailable,dense_frame_count=len(samples),completion=completion,
  readouts=[dict(tick=s['tick'],elapsed_s=s['elapsed_s'],common_time_readout=s['common_time_readout'],requested95_readout=s['requested95_readout'],
   head_RGB_sha256=array_hash(s['observation']['rgb'][0]),physics_sha256=s['observation']['physical_sha256'],rng_sha256=s['observation']['rng_sha256'])for s in samples],
  phase_plan_source='actual_executed_expert32_not_offline_VLA_base')
 return arrays,metadata,dense

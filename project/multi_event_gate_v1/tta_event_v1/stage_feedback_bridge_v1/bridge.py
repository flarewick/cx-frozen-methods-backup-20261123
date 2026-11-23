"""After-committed-actuation adapter; no physics/control calls of its own."""
from contextvars import ContextVar
ACTIVE=ContextVar('cx_arrived_stage_feedback_bridge_v1',default=None)

class AfterActuation:
 def __init__(self,recorder,feedback_queue,chunk_id,passive_read,*,common_ticks,expected_stop):
  self.recorder=recorder;self.queue=feedback_queue;self.chunk_id=chunk_id;self.passive_read=passive_read;self.common_ticks=frozenset(common_ticks);self.expected_stop=expected_stop
  self.start_count=len(recorder.rows);self.start_tick=feedback_queue.tick;self.failed=False;self.readouts=[]
 def call(self,delegate,*args,**kwargs):
  if self.failed:raise ValueError('Failed native bridge cannot resume partial branch')
  before=len(self.recorder.rows)
  if self.queue.tick!=self.start_tick+before-self.start_count:raise ValueError('Own recorder and feedback clock shifted')
  try:result=delegate(*args,**kwargs)
  except self.expected_stop:
   self._after(before);raise
  except BaseException:
   self.failed=True;raise
  self._after(before);return result
 def _after(self,before):
  try:
   if self.recorder.failed or len(self.recorder.rows)!=before+1 or len(self.recorder.dt)!=before+1:raise ValueError('Exactly one actually committed native step required')
   row=self.recorder.rows[-1];dt=self.recorder.dt[-1];tick=self.start_tick+len(self.recorder.rows)-self.start_count
   self.queue.step(self.chunk_id,row['gripper_logical_command'],dt,arrived_tick=tick)
   if tick in self.common_ticks or self.queue.due():
    observation=self.passive_read(tick,self.queue.seconds)
    if not isinstance(observation,dict) or not isinstance(observation.get('frame_id'),str):raise ValueError('Passive reader must return current RGB evidence identity')
    requests=self.queue.observe(tick,decision_tick=tick,frame_id=observation['frame_id'])
    self.readouts.append(dict(tick=tick,elapsed_s=self.queue.seconds,common_time_readout=tick in self.common_ticks,stage_requests=requests,observation=observation))
  except BaseException:
   self.failed=True;raise
 def end_origin_chunk(self):
  if self.failed:raise ValueError('Failed native branch never exports')
  self.queue.end_chunk(self.chunk_id,arrived_tick=self.queue.tick)

def trace_step(task):
 # Existing frozen source remains untouched. Old500clock is still authoritative
 # when this optional adapter is used in a separately prepared runtime copy.
 from cx_tta_cover3.schedule import trace_step as original
 active=ACTIVE.get()
 return original(task) if active is None else active.call(original,task)

def instrument(source):
 from cx_tta_cover3.schedule import instrument as original
 text=original(source);needle='from cx_tta_cover3.schedule import trace_step as _cx_diag_step\n'
 if text.count(needle)!=1:raise ValueError('Frozen native wrapper import changed')
 return text.replace(needle,'from stage_feedback_bridge_v1.bridge import trace_step as _cx_diag_step\n')

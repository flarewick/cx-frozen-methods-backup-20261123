"""Align endpoint feedback with the preserved native post-command observation."""
import hashlib,numpy as np
from cx_tta_feedback_confirm5.capture import CurrentReadouts as Original

def rgb_sha(obs):return hashlib.sha256(np.ascontiguousarray(obs['rgb'][0]).tobytes()).hexdigest()

class CurrentReadouts(Original):
 def __init__(self,*args,on_arrival=None,**kwargs):
  self._consumer=on_arrival;self._pending=None;self.endpoint_readback=None
  super().__init__(*args,on_arrival=self._buffer,**kwargs)
 def _buffer(self,tick,elapsed,observed,recorder):
  if self._pending is not None:raise ValueError('An arrived frame was not dispatched before the next read')
  self._pending=(tick,elapsed,observed,recorder)
 def flush(self):
  if self._pending is None:return
  args=self._pending;self._pending=None
  if self._consumer is not None:self._consumer(*args)
 def after(self,before):
  if self.recorder.failed or len(self.recorder.rows)!=before+1:return super().after(before)
  # One later committed step may exist in recorder; SourceStream slices exactly
  # [:captured_tick], so no future command/readout is included in the label.
  self.flush()
  return super().after(before)
 def terminal_read(self,tick,*,complete_native_chunk):
  cached=bool(self.readouts and self.readouts[-1]['tick']==tick)
  if cached and complete_native_chunk:
   if self._pending is None or self._pending[0]!=tick:raise ValueError('Endpoint already dispatched; do not overwrite queue history')
   old=self.readouts[-1]['observation'];elapsed=float(sum(self.recorder.dt))
   fresh=self.reader(tick,elapsed)
   if not np.array_equal(old['command_state'],fresh['command_state']):raise ValueError('Endpoint own commands changed without a committed step')
   if old['physical_sha256']!=fresh['physical_sha256']:raise ValueError('Endpoint diagnostic physics changed without a committed step')
   self.endpoint_readback=dict(tick=tick,elapsed_s=elapsed,old_head_RGB_sha256=rgb_sha(old),new_head_RGB_sha256=rgb_sha(fresh),RGB_changed=rgb_sha(old)!=rgb_sha(fresh),native_observation_cadence_preserved=True,physical_and_own_command_exact=True,callback_not_previously_dispatched=True,only_one_feedback_frame_dispatched=True)
   self.readouts[-1]['observation']=fresh
   self._pending=(tick,elapsed,fresh,self.recorder)
   terminal=fresh
  else:
   # A terminal read that is not on the grid already occurs after discard in
   # the original implementation; censored cached endpoints are not refreshed.
   if not cached:self.flush()
   terminal=super().read(tick)
  self.flush()
  return terminal

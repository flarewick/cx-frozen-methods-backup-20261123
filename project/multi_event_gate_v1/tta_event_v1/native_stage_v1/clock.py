"""Separate bounded measurement clock; never mutates old clock.TICKS."""
import numpy as np
from cx_tta_clock.clock import ClockComplete
from actuation_telemetry import Recorder
LEGACY_TICKS=(32,64,125,187,250,375,500)
MAX_TICKS=5000
COMMON_TICKS=tuple(sorted(set(LEGACY_TICKS)|{500+round(k*62.5) for k in range(1,73)}))
class StageClock:
 def __init__(self,max_ticks=MAX_TICKS):
  if type(max_ticks)is not int or not 1<=max_ticks<=MAX_TICKS:raise ValueError('Bounded physical measurement')
  self.max_ticks=max_ticks;self.commands=[];self.dt=[];self.samples=[]
 def after_step(self,command,dt):
  command=np.asarray(command)
  if command.shape!=(14,) or not np.isfinite(command).all() or not np.isfinite(dt) or dt<=0:raise ValueError('Finite executed command and actualdt required')
  if len(self.dt)>=self.max_ticks:raise ValueError('No steps after explicit measurement stop')
  if self.dt and dt!=self.dt[0]:raise ValueError('Native dt changed')
  if abs(float(dt)-.004)>1e-8:raise ValueError('Original250Hz nativecontract required')
  self.commands.append(command.astype(np.float32).copy());self.dt.append(float(dt))
  if len(self.dt)==self.max_ticks:raise ClockComplete()

class OriginRecorder(Recorder):
 def __init__(self,max_ticks=MAX_TICKS):
  if type(max_ticks)is not int or not 1<=max_ticks<=MAX_TICKS:raise ValueError('Bounded original measurement limit')
  super().__init__(limit=max_ticks);self.sealed=None
 def append(self,row,dt):
  if self.sealed is not None:raise ValueError('Cannot append after nativeorigin or captureend')
  super().append(row,dt)
 def seal(self,reason,*,commands_returned,expected_commands=32):
  if self.failed or self.sealed is not None:raise ValueError('Failed or duplicate seal')
  if type(commands_returned)is not int or expected_commands!=32 or not 0<=commands_returned<=32:raise ValueError('Original32 command schedule required')
  if reason not in ('origin_chunk_completed','measurement_horizon_end','native_command_censored'):raise ValueError('Explicit termination required')
  if reason=='origin_chunk_completed' and commands_returned!=32:raise ValueError('All32originalcommands mustreturn')
  if reason=='measurement_horizon_end' and len(self.rows)!=self.limit:raise ValueError('No fabricated timeout')
  self.sealed=dict(reason=reason,commands_returned=commands_returned,expected_commands=32,committed_ticks=len(self.rows),complete_origin=reason=='origin_chunk_completed')
 def packet(self):
  if self.failed or self.sealed is None or not self.rows:raise ValueError('Only sealed actual measurements export; zerostepcapture explicitunavailable')
  value={key:np.stack([r[key] for r in self.rows]) for key in self.rows[0]};value['dt_s']=np.array(self.dt,np.float64)
  return value

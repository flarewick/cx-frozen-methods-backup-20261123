"""Whole-policy execution guard; original per-step validation and sampling stay fixed."""
from native_stage_v1.clock import StageClock
MAX_EXECUTION_TICKS=100000
class ExecutionClock(StageClock):
 def __init__(self,max_ticks=MAX_EXECUTION_TICKS):
  if type(max_ticks)is not int or not 1<=max_ticks<=MAX_EXECUTION_TICKS:raise ValueError('Explicit bounded whole-policy execution')
  self.max_ticks=max_ticks;self.commands=[];self.dt=[];self.samples=[]

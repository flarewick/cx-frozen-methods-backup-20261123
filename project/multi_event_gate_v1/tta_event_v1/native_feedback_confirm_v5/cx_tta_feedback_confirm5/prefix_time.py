"""Actual stepcounts returned by existing nativecontrol, plus originalslowpath."""
from contextvars import ContextVar
import numpy as np
from .paths import read
ACTIVE=ContextVar('cx_native_prefix_clock',default=None)
class Counter:
 def __init__(self,dt):
  if not np.isfinite(dt) or dt<=0:raise ValueError('Realpositive scene timestep required')
  self.dt=float(dt);self.ticks=0
 def __enter__(self):self.token=ACTIVE.set(self);return self
 def __exit__(self,*args):ACTIVE.reset(self.token)
 def append(self,ticks,dt):
  if type(ticks)is not int or ticks<0 or float(dt)!=self.dt:raise ValueError('Actualcounts/constantreal dt required')
  self.ticks+=ticks

def install_native(module=None):
 if module is None:
  import native_generic_physics
  module=native_generic_physics.load_native()
 original=module.control
 def wrapped(*args,**kwargs):
  result=original(*args,**kwargs);counter=ACTIVE.get()
  if counter is not None:counter.append(int(result[3]),counter.dt)
  return result
 module.control=wrapped
 return lambda:setattr(module,'control',original)

def verify_prefix_clock(folder):
 value=read(folder/'prefix_clock.json');discovery=read(folder/'discovery.json')
 ticks=value['query_observation_ticks'];rows=value['executed_chunks']
 if len(ticks)!=discovery['sampler_calls'] or not ticks or ticks[0]!=0 or any(type(v)is not int or v<0 for v in ticks):raise ValueError('Queryclock missing/invalid')
 if len(rows)!=len(ticks)-1:raise ValueError('Completeprefix requires actualspan perobservedinterval')
 for i,row in enumerate(rows):
  if row['query']!=i or row['start_tick']!=ticks[i] or row['end_tick']!=ticks[i+1] or row['end_tick']<row['start_tick'] or row['censored']:raise ValueError('Noncausal/missing trueprefixspan')
 if not value['native_returned_counts_and_actual_slow_steps'] or not value['physical_interval_duration_available'] or not 0<value['dt_s']<.1:raise ValueError('No fabricated/unknown clock')
 return True

def checks():
 from types import SimpleNamespace
 result=(False,0,0,17,0.,0.);native=SimpleNamespace(control=lambda *a,**k:result);old=native.control
 restore=install_native(native)
 with Counter(.004) as a:
  assert native.control() is result and a.ticks==17
  with Counter(.004) as b:assert native.control() is result and b.ticks==17
  assert a.ticks==17
  a.append(3,.004);assert a.ticks==20
 restore();assert native.control is old and ACTIVE.get() is None
 return dict(native_return_tuple_identity_and_counts=True,prefix_counter_thread_context_isolated=True,native_function_restored=True)

"""Passive initial observation with the original expert's explicit step_lim=None."""
import numpy as np

def physical(task):
 from cx_tta.attribution_physics import physical_snapshot
 from cx_tta.eval_diagnostics import json_hash
 value=physical_snapshot(task)
 if task.step_lim is None:
  errors=value['coverage']['errors'];expected=[r for r in errors if r['field']=='control.step_lim']
  assert len(expected)==1 and value['control']['step_lim']is None
  # None is the native expert mode, not a missing limit in a policy action loop.
  value['coverage']['errors']=[r for r in errors if r['field']!='control.step_lim']
  value['usable']=not value['coverage']['errors']
 if not value['usable']:raise ValueError('Incomplete actual native physical state')
 return json_hash(value)

def observe(task):
 from cx_tta.deploy import unpack_observation
 from cx_tta_clock.clock import passive_observation
 from cx_tta.recovery_contract import rng_capture,rng_fingerprint
 before=physical(task);count=int(task.take_action_cnt);rng=rng_fingerprint(rng_capture())
 def actual():
  images,state=unpack_observation(task.get_obs())
  return dict(rgb=tuple(np.array(im,copy=True)for im in images),command_state=state.copy())
 value=passive_observation(task,actual)
 assert physical(task)==before and int(task.take_action_cnt)==count and rng_fingerprint(rng_capture())==rng
 value.update(physical_sha256=before,rng_sha256=rng,native_action_count=count)
 return value

from contextlib import contextmanager
@contextmanager
def unrecorded(task,original_get_obs):
 recorded=task.get_obs;task.get_obs=original_get_obs
 try:yield
 finally:task.get_obs=recorded

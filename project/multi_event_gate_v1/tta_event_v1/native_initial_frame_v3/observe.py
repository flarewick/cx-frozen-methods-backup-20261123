"""Same passive native get_obs/RGB/state path; no encoder, actor, future or action."""
import numpy as np
def physical(task):
 from cx_tta.attribution_physics import physical_snapshot
 from cx_tta.eval_diagnostics import json_hash
 value=physical_snapshot(task)
 if not value['usable']:raise ValueError('Unreadable initialization physics')
 return json_hash(value)
def observe(task):
 from cx_tta.deploy import unpack_observation
 from cx_tta_clock.clock import passive_observation
 from cx_tta.recovery_contract import rng_capture,rng_fingerprint
 before=physical(task);count=task.take_action_cnt
 def actual():
  images,state=unpack_observation(task.get_obs())
  return dict(rgb=tuple(np.array(im,copy=True)for im in images),command_state=state.copy())
 value=passive_observation(task,actual)
 assert physical(task)==before and task.take_action_cnt==count==0
 value.update(initial_physical_sha256=before,rng_sha256=rng_fingerprint(rng_capture()),policy_actions=0)
 return value

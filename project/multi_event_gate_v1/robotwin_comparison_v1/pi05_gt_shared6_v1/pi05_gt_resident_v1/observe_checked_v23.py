"""One passive native observation, retaining raw signature before policy float32 conversion."""
import numpy as np
from cx_tta.eval_diagnostics import array_hash

def raw_signature(observation):
 return dict(images={name:array_hash(observation['observation'][name]['rgb'])for name in ('head_camera','right_camera','left_camera')},command_state=array_hash(observation['joint_action']['vector']))

def observe(task):
 from cx_tta.deploy import unpack_observation
 from cx_tta_clock.clock import passive_observation
 from cx_tta.recovery_contract import rng_capture,rng_fingerprint
 from native_initial_frame_v3.observe import physical
 before=physical(task);count=int(task.take_action_cnt);rng=rng_fingerprint(rng_capture())
 def actual():
  observation=task.get_obs();signature=raw_signature(observation);images,state=unpack_observation(observation)
  return dict(rgb=tuple(np.array(im,copy=True)for im in images),command_state=state.copy(),raw_observation_signature=signature)
 value=passive_observation(task,actual)
 physical_ok=physical(task)==before
 action_ok=int(task.take_action_cnt)==count
 rng_ok=rng_fingerprint(rng_capture())==rng
 assert physical_ok and action_ok and rng_ok,('Passive observation changed state',type(task).__name__,count,physical_ok,action_ok,rng_ok)
 value.update(physical_sha256=before,rng_sha256=rng,native_action_count=count)
 return value

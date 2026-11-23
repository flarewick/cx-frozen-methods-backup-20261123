"""Passive RGB and commanded state at actual native chunk boundaries."""
import numpy as np
def observe(task):
 from cx_tta.deploy import unpack_observation
 from cx_tta_clock.clock import passive_observation
 from cx_tta.recovery_contract import rng_capture,rng_fingerprint
 from native_initial_frame_v3.observe import physical
 before=physical(task);count=int(task.take_action_cnt);rng=rng_fingerprint(rng_capture())
 def actual():
  images,state=unpack_observation(task.get_obs())
  return dict(rgb=tuple(np.array(im,copy=True)for im in images),command_state=state.copy())
 value=passive_observation(task,actual)
 assert physical(task)==before and int(task.take_action_cnt)==count and rng_fingerprint(rng_capture())==rng
 value.update(physical_sha256=before,rng_sha256=rng,native_action_count=count)
 return value

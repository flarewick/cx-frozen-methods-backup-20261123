"""Twelve same-capture raw RGBA checks, at most six per task type; no extra capture."""
import numpy as np

def configure(overlap):
 assert not hasattr(overlap,'image_shadow')
 overlap.image_shadow=dict(observations=0,cameras=0,task_types=set(),per_task={})
def compare(overlap,task,cameras,values):
 state=overlap.image_shadow;name=type(task).__module__+'.'+type(task).__name__
 if state['observations']>=12 or state['per_task'].get(name,0)>=6:return
 assert len(cameras)==len(values)>0
 for camera,value in zip(cameras,values,strict=True):
  original=camera.get_picture('Color')
  assert value.dtype==original.dtype==np.float32 and value.shape==original.shape and value.strides==original.strides
  assert value.tobytes()==original.tobytes(),'Original same-capture float RGBA differs'
  state['cameras']+=1
 state['observations']+=1;state['task_types'].add(name);state['per_task'][name]=state['per_task'].get(name,0)+1
def report(overlap):
 state=overlap.image_shadow;assert overlap.closed and state['observations']==12 and state['cameras']>=36 and len(state['task_types'])>=2
 return dict(passed=True,same_capture_observations=12,raw_camera_arrays=state['cameras'],task_types=sorted(state['task_types']),per_task=dict(state['per_task']),float32_RGBA_dtype_shape_stride_bytes_exact=True,no_extra_camera_capture_or_policy_query=True,worker_closed=True)

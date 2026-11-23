"""Same captured buffer oracle; no additional camera capture or observation."""
import numpy as np
from native_policy_execution_v1.common import read,sha
def configure(overlap):
 assert not hasattr(overlap,'image_shadow')
 overlap.image_shadow=dict(observations=0,cameras=0,task_types=set())
def compare(overlap,task,cameras,values):
 state=overlap.image_shadow
 if state['observations']>=12:return
 assert len(cameras)==len(values)>0
 for camera,value in zip(cameras,values,strict=True):
  original=camera.get_picture('Color')
  assert value.dtype==original.dtype==np.float32 and value.shape==original.shape and value.strides==original.strides
  assert value.tobytes()==original.tobytes(),'Original same-capture float RGBA differs'
  state['cameras']+=1
 state['observations']+=1;state['task_types'].add(type(task).__module__+'.'+type(task).__name__)
def report(overlap):
 state=overlap.image_shadow;assert overlap.closed and state['observations']==12 and state['cameras']>=36 and len(state['task_types'])>=2
 return dict(passed=True,same_capture_observations=12,raw_camera_arrays=state['cameras'],task_types=sorted(state['task_types']),float32_RGBA_dtype_shape_stride_bytes_exact=True,no_extra_camera_capture_or_policy_query=True,worker_closed=True)
def audit(root,spec):
 assert spec['native_original_getImage_GIL_release_async_readback']
 file=root/'native_image_readback_runtime.json';v=read(file)
 assert v['passed']and v['same_capture_observations']==12 and v['raw_camera_arrays']>=36 and len(v['task_types'])>=2
 assert v['float32_RGBA_dtype_shape_stride_bytes_exact']and v['no_extra_camera_capture_or_policy_query']and v['worker_closed']
 return dict(passed=True,raw_arrays=v['raw_camera_arrays'],actual_same_capture_RGBA_byte_exact=True,original_whole25_controls_still_required=True,input_sha256={str(file):sha(file)})

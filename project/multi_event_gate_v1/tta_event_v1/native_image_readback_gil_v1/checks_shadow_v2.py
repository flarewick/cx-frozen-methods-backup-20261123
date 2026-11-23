import ast
from pathlib import Path
import numpy as np
from native_policy_execution_v1.common import T,write,sha

def main():
 import torch
 from . import readback_shadow_v2 as readback
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();saved=readback.batch_read;old_install=readback.install
 class CPUOverlap:
  def __init__(self):self.image_shadow=dict(observations=12,cameras=36,task_types=set())
  def run(self,fn,args):return fn(*args)
 readback.install=lambda task,enabled:old_install(task,enabled,CPUOverlap())
 source=Path('/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin/envs/camera/camera.py');node=next(n for n in ast.walk(ast.parse(source.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='get_rgba');namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
 class Buffer:
  def __init__(self,n):self.value=(np.arange(48,dtype=np.float32).reshape(3,4,4)+n)/255.;self.reads=0
  def get_picture(self,name):assert name=='Color';self.reads+=1;return self.value.copy()
 class Cameras:
  collect_wrist_camera=True;collect_head_camera=True;get_rgba=namespace['get_rgba']
  def __init__(self):self.left_camera=Buffer(0);self.right_camera=Buffer(10);self.static_camera_list=[Buffer(20),Buffer(30)];self.static_camera_name=['head_camera','front_camera']
 class Task:
  render_freq=0;eval_video_path=None;data_type={}
 task=Task();task.cameras=Cameras();c=task.cameras;calls=[]
 readback.batch_read=lambda cameras:(calls.append([id(cam)for cam in cameras]),[cam.value.copy()for cam in cameras])[1]
 try:
  readback.install(task,True);getter=c.get_rgba;readback.install(task,True);assert c.get_rgba is getter
  for i in range(3):
   expected=Cameras.get_rgba(c);count=sum(x.reads for x in[c.left_camera,c.right_camera,*c.static_camera_list]);actual=c.get_rgba()
   assert actual.keys()==expected.keys()
   for k in expected:assert actual[k]['rgba'].tobytes()==expected[k]['rgba'].tobytes()
   assert count==sum(x.reads for x in[c.left_camera,c.right_camera,*c.static_camera_list])
   c.left_camera.value+=np.float32(1/255.)
  c.static_camera_list=c.static_camera_list[:1];c.static_camera_name=['head_camera'];c.get_rgba();assert len(calls[-1])==3
  readback.install(task,False);assert 'get_rgba'not in c.__dict__ and not c._cx_native_CPU_RGBA
  task.data_type={'depth':True}
  try:readback.install(task,True)
  except AssertionError:pass
  else:raise AssertionError('Unsupported modality accepted')
 finally:readback.batch_read=saved;readback.install=old_install
 from .shadow import configure,compare
 test=CPUOverlap();del test.image_shadow;configure(test)
 cameras=[c.left_camera,c.right_camera,*c.static_camera_list]
 values=[cam.value.copy()for cam in cameras]
 for _ in range(12):compare(test,task,cameras,values)
 assert test.image_shadow['observations']==12 and test.image_shadow['cameras']==36
 total=sum(cam.reads for cam in cameras);compare(test,task,cameras,values);assert sum(cam.reads for cam in cameras)==total
 bad=CPUOverlap();del bad.image_shadow;configure(bad);broken=[v.copy()for v in values];broken[0].flat[0]+=np.float32(1.)
 try:compare(bad,task,cameras,broken)
 except AssertionError:pass
 else:raise AssertionError('Raw float difference accepted')
 assert not torch.cuda.is_initialized()
 import sapien
 from .loader_validated import load
 import sapien.render as sr
 assert load().registered_camera_type()is sr.RenderCameraComponent
 C=Path(__file__).parent;write(T/'outputs/native_image_readback_gil_v1/CPU_shadow_pixel_checks_v2.json',dict(passed=True,CPU_only=True,actual_original_get_rgba_AST_pixel_dictionary_preserved=True,original_CPP_readback_fixture_no_extra_capture=True,new_frames_and_filtered_camera_order_exact=True,reference_restores_original_getter=True,unsupported_modes_refused=True,requires_actual_original_whole25_controls_new_getter_not_GPU_qualified=True,source_sha256={str(p):sha(p)for p in C.glob('*')if p.is_file()},input_sha256={str(source):sha(source)}))
 print('CPU_ORIGINAL_CAMERA_RGBA_FAST_PRIMARY_REFERENCE_RESTORE_PASS',flush=True)
if __name__=='__main__':main()

"""Keep all original returned policy/feedback channels; omit only unused front RGB."""
import sys
def configure(task,enabled):task._cx_only_used_static_camera=bool(enabled)
def build(original):
 def observe(task):
  if not getattr(task,'_cx_only_used_static_camera',False):return original(task)
  assert task.render_freq==0 and getattr(task,'eval_video_path',None)is None
  assert not any(task.data_type.get(k,False)for k in ['third_view','depth','mesh_segmentation','actor_segmentation','pointcloud'])
  c=task.cameras;old_cameras=c.static_camera_list;old_names=c.static_camera_name
  assert c.collect_head_camera and c.collect_wrist_camera
  assert len(old_names)==len(old_cameras)and len(set(old_names))==len(old_names)
  assert set(old_names)=={'head_camera','front_camera'},('Unvalidated static camera configuration',old_names)
  index=old_names.index('head_camera');c.static_camera_list=[old_cameras[index]];c.static_camera_name=['head_camera']
  try:return original(task)
  finally:c.static_camera_list=old_cameras;c.static_camera_name=old_names
 return observe
def install():
 from native_expert_smoke_v2 import observe as module
 original=module.observe;candidate=build(original);changed=[]
 for name,mod in list(sys.modules.items()):
  if mod is None or not(name.startswith('native_')or name.startswith('cx_tta')):continue
  if vars(mod).get('observe')is original:vars(mod)['observe']=candidate;changed.append(name)
 assert 'native_expert_smoke_v2.observe'in changed
 return changed

import ast,copy,inspect,textwrap,weakref
from native_image_readback_gil_v1.loader_validated import read as batch_read

def install(task,enabled,overlap):
 c=task.cameras
 if not enabled:
  if getattr(c,'_cx_native_CPU_RGBA',False):c.__dict__.pop('get_rgba');c._cx_native_CPU_RGBA=False
  return
 if getattr(c,'_cx_native_CPU_RGBA',False):return
 assert task.render_freq==0 and getattr(task,'eval_video_path',None)is None
 assert not any(task.data_type.get(k,False)for k in['third_view','depth','mesh_segmentation','actor_segmentation','pointcloud'])
 assert 'get_rgba'not in c.__dict__
 original=type(c).get_rgba;tree=ast.parse(textwrap.dedent(inspect.getsource(original)));before=copy.deepcopy(tree);hits=[]
 class Change(ast.NodeTransformer):
  def visit_Call(self,n):
   if ast.unparse(n)=="camera.get_picture('Color')":hits.append(copy.deepcopy(n));return ast.copy_location(ast.parse('cached(camera)',mode='eval').body,n)
   return self.generic_visit(n)
 tree=Change().visit(tree);assert len(hits)==1
 class Undo(ast.NodeTransformer):
  def visit_Call(self,n):return copy.deepcopy(hits[0])if ast.unparse(n)=='cached(camera)'else self.generic_visit(n)
 assert ast.dump(Undo().visit(copy.deepcopy(tree)))==ast.dump(before)
 box={};namespace=dict(original.__globals__,cached=lambda camera:box[id(camera)])
 exec(compile(ast.fix_missing_locations(tree),'<original-native-CPU-RGBA-pixel-conversion>','exec'),namespace);publish=namespace['get_rgba'];ref=weakref.ref(c)
 def getter():
  cameras=ref();assert cameras is not None
  chosen=([cameras.left_camera,cameras.right_camera]if cameras.collect_wrist_camera else[])+[cam for cam,name in zip(cameras.static_camera_list,cameras.static_camera_name,strict=True)if name!='head_camera'or cameras.collect_head_camera]
  values=overlap.run(batch_read,(chosen,));assert len(values)==len(chosen)
  from native_seven_plain_early_v1.pixel_shadow import compare
  compare(overlap,task,chosen,values)
  box.update({id(cam):value for cam,value in zip(chosen,values,strict=True)})
  try:return publish(cameras)
  finally:box.clear()
 c.get_rgba=getter;c._cx_native_CPU_RGBA=True

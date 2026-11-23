"""Coalesce only unobserved render pose uploads, flush before every capture.

All physics, light random draws, wrist-camera pose updates and picture captures
remain in their original order. GPU image parity is a separate mandatory gate.
"""
import ast
import hashlib
import inspect
import textwrap
import time
import weakref


def contract(scene_source,camera_source):
    def method(source,name):
        nodes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(nodes)!=1:raise ValueError('Ambiguous render method '+name)
        return nodes[0]
    scene=method(scene_source,'update_render')
    expected=ast.parse('def update_render(self):\n    self.render_system.step()').body[0]
    if ast.dump(scene,include_attributes=False)!=ast.dump(expected,include_attributes=False):
        raise ValueError('Scene.update_render is no longer a pure render-system pose upload')
    camera=method(camera_source,'update_picture')
    expected=ast.parse('def update_picture(self):\n    if self.collect_wrist_camera:\n        self.left_camera.take_picture()\n        self.right_camera.take_picture()\n    for camera in self.static_camera_list:\n        camera.take_picture()').body[0]
    if ast.dump(camera,include_attributes=False)!=ast.dump(expected,include_attributes=False):
        raise ValueError('Original camera capture order changed')
    return dict(passed=True,scene_ast_sha256=hashlib.sha256(ast.dump(scene,include_attributes=False).encode()).hexdigest(),
                camera_ast_sha256=hashlib.sha256(ast.dump(camera,include_attributes=False).encode()).hexdigest())


def install(task,gate=None,native=None):
    if task.render_freq or getattr(task,'eval_video_path',None) is not None:
        raise ValueError('Deferred pose upload requires headless no-video control')
    if any(task.data_type.get(k,False) for k in ('third_view','depth','mesh_segmentation','actor_segmentation','pointcloud')):
        raise ValueError('Unvalidated direct render readout')
    scene,cameras=task.scene,task.cameras
    if getattr(cameras,'_cx_pose_upload',False):return cameras._cx_pose_upload_stats
    contract(textwrap.dedent(inspect.getsource(type(scene).update_render)),
             textwrap.dedent(inspect.getsource(type(cameras).update_picture)))
    scene_ref,camera_ref=weakref.ref(scene),weakref.ref(cameras)
    dirty=[False]
    stats=dict(logical_uploads=0,physical_uploads=0,captures=0,upload_seconds=0.,capture_seconds=0.)
    # Unbound functions avoid reference cycles retaining old Vulkan scenes.
    original_upload=type(scene).update_render
    original_capture=type(cameras).update_picture
    def mark():dirty[0]=True;stats['logical_uploads']+=1
    def capture():
        s,c=scene_ref(),camera_ref()
        if s is None or c is None:raise RuntimeError('Deferred render owner already released')
        if dirty[0]:
            began=time.monotonic()
            if native is None:original_upload(s)
            else:
                with gate.read():native.render_step(s.render_system)
            stats['upload_seconds']+=time.monotonic()-began;stats['physical_uploads']+=1;dirty[0]=False
        began=time.monotonic()
        if native is None:original_capture(c)
        else:
            selected=([c.left_camera,c.right_camera] if c.collect_wrist_camera else [])+list(c.static_camera_list)
            with gate.read():native.take_pictures(selected)
        stats['capture_seconds']+=time.monotonic()-began;stats['captures']+=1
    scene.update_render=mark;cameras.update_picture=capture
    cameras._cx_pose_upload=True;cameras._cx_pose_upload_stats=stats
    return stats

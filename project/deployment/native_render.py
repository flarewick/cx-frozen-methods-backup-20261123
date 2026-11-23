"""Call every original SAPIEN render/camera operation with Python's GIL released.

One native mutex serializes renderer calls. Per-scene ownership and SceneGate
prevent teardown/global-cache races; no frame, pose update or readback is skipped.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time
import weakref

HERE=Path(__file__).resolve().parent

def install_task(task,scene_gate,native,note=lambda *a,**k:None):
    scene=task.scene;cameras=task.cameras
    if getattr(cameras,'_cx_native_capture',False):return
    source=Path(inspect.getsourcefile(type(cameras)))
    if source.name!='camera.py':raise RuntimeError('Unexpected camera runtime')
    proof=json.loads((HERE/'reports/target_native_render_cpu_20261023.json').read_text())
    if not proof.get('passed')or proof['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()or proof['original_camera_source_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest():raise RuntimeError('Native camera CPU contract proof stale')
    install(cameras,scene,scene_gate,native,note)

def install(cameras,scene,scene_gate,native,note=lambda *a,**k:None):
    expected=ast.parse("def update_render(self):\n    self.render_system.step()")
    actual=ast.parse(textwrap.dedent(inspect.getsource(scene.update_render.__func__)))
    if ast.dump(actual,include_attributes=False)!=ast.dump(expected,include_attributes=False):raise RuntimeError('Original scene render API changed')
    # Instance methods must not retain their owning scene/camera through closures.
    # Old Vulkan scenes should die immediately on replacement, without cyclic GC.
    scene_ref=weakref.ref(scene);cameras_ref=weakref.ref(cameras)
    stats=dict(captures=0,camera_calls=0,render_seconds=0.,capture_seconds=0.)
    def update_render():
        current=scene_ref()
        if current is None:raise RuntimeError('Render scene already released')
        began=time.monotonic()
        with scene_gate.read():native.render_step(current.render_system)
        stats['render_seconds']+=time.monotonic()-began
    def update_picture():
        current=cameras_ref()
        if current is None:raise RuntimeError('Capture cameras already released')
        selected=[]
        if current.collect_wrist_camera:selected.extend([current.left_camera,current.right_camera])
        selected.extend(current.static_camera_list)
        began=time.monotonic()
        with scene_gate.read():native.take_pictures(selected)
        stats['capture_seconds']+=time.monotonic()-began
        stats['captures']+=1;stats['camera_calls']+=len(selected)
        if stats['captures']%128==0:note('native_capture_profile',**stats)
    scene.update_render=update_render;cameras.update_picture=update_picture
    cameras._cx_native_capture=True;cameras._cx_native_capture_stats=stats

"""Defer physics-only visual sync; preserve every original observation/capture.

The original update is still called by physics control each tick. Its random
light prefix is retained. The two visual-sync statements are flushed by every
get_obs, including internal early-success observations. Rendering is never read
by the six original success predicates.
"""
import ast,functools,hashlib,inspect,json
from pathlib import Path
from render_sync_materialization import prefix_method
D=Path(__file__).resolve().parent

def install(task,note=lambda *a,**k:None,verify_proof=True):
    if getattr(task,'_cx_shared6_visual_sync',False):return
    if task.render_freq or task.eval_video_path is not None:raise RuntimeError('Visual sync deferral requires original headless/no-video mode')
    if verify_proof:
        proof=json.loads((D/'reports/target_shared6_render_sync_cpu_20261024.json').read_text())
        if not proof.get('passed')or proof['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()or proof['prefix_helper_sha256']!=hashlib.sha256((D/'render_sync_materialization.py').read_bytes()).hexdigest():raise RuntimeError('Original visual-sync CPU proof missing/stale')
        name=type(task).__name__;path=Path(inspect.getsourcefile(type(task)))
        if name not in proof['task_source_sha256']or hashlib.sha256(path.read_bytes()).hexdigest()!=proof['task_source_sha256'][name]:raise RuntimeError('Unverified task visual dependencies')
    original_update=task._update_render;original_action=task.take_action;original_obs=task.get_obs
    prefix=prefix_method(inspect.unwrap(original_update))
    inside=materializing=deferred=actions=0
    @functools.wraps(original_action)
    def action(*args,**kwargs):
        nonlocal inside,actions
        inside+=1
        try:return original_action(*args,**kwargs)
        finally:
            inside-=1;actions+=1
            if actions%128==0:note('shared6_physics_visual_sync_deferred',actions=actions,physics_only_syncs_deferred=deferred)
    @functools.wraps(original_obs)
    def observation(*args,**kwargs):
        nonlocal materializing
        materializing+=1
        try:return original_obs(*args,**kwargs)
        finally:materializing-=1
    @functools.wraps(original_update)
    def update(*args,**kwargs):
        nonlocal deferred
        if inside and not materializing:
            deferred+=1;return prefix(*args,**kwargs)
        return original_update(*args,**kwargs)
    task.take_action=action;task.get_obs=observation;task._update_render=update
    task._cx_shared6_visual_sync=True

"""Experimental defer of visual pose synchronization during physics substeps.

All light updates/RNG and every original camera capture remain. Flush the full
original render update before each observation, including success-time capture.
"""
import ast
import functools
import inspect
import textwrap
import types
from observation_materialization import wrap as materialize


def prefix_method(method):
    fn=method.__func__
    tree=ast.parse(textwrap.dedent(inspect.getsource(fn)))
    function=tree.body[0]
    if not isinstance(function,ast.FunctionDef) or function.name!='_update_render':
        raise ValueError('Unknown render update method')
    tails=[ast.unparse(n) for n in function.body[-2:]]
    expected=['self.cameras.update_wrist_camera(self.robot.left_camera.get_pose(), self.robot.right_camera.get_pose())','self.scene.update_render()']
    if tails!=expected:raise ValueError('Frozen render update structure changed')
    function.body=function.body[:-2]
    namespace={}
    exec(compile(tree,fn.__code__.co_filename,'exec'),fn.__globals__,namespace)
    return types.MethodType(namespace['_update_render'],method.__self__)


def wrap(step,note=lambda *a,**k:None):
    optimized=materialize(step,note)
    @functools.wraps(step)
    def run(task,policy,*args,**kwargs):
        if getattr(task,'render_freq',0):raise ValueError('Requires interactive rendering disabled')
        original_update=task._update_render;original_action=task.take_action;original_obs=task.get_obs
        prefix=prefix_method(original_update)
        inside=materializing=skipped=0
        restored=[]
        def replace(name,fn):
            restored.append((name,name in vars(task),vars(task).get(name)));setattr(task,name,fn)
        def action(*a,**k):
            nonlocal inside
            inside+=1
            try:return original_action(*a,**k)
            finally:inside-=1
        def observation(*a,**k):
            nonlocal materializing
            materializing+=1
            try:return original_obs(*a,**k)
            finally:materializing-=1
        def update(*a,**k):
            nonlocal skipped
            if inside and not materializing:
                skipped+=1;return prefix(*a,**k)
            return original_update(*a,**k)
        replace('_update_render',update);replace('take_action',action);replace('get_obs',observation)
        try:
            result=optimized(task,policy,*args,**kwargs)
            note('deferred_visual_pose_sync',physics_only_syncs_deferred=skipped)
            return result
        finally:
            for name,owned,old in reversed(restored):
                if owned:setattr(task,name,old)
                else:delattr(task,name)
    return run

"""Experimental deferred CPU observation materialization; all renders retained.

Not enabled in any evaluation entry until protocol/parity validation completes.
"""
import functools


def wrap(step, note=lambda *a, **k: None):
    @functools.wraps(step)
    def optimized(task, policy, *args, **kwargs):
        if getattr(task,'eval_video_path',None) is not None or getattr(task,'eval_video_ffmpeg',None) is not None:
            raise ValueError('Deferred materialization requires video disabled')
        if any(task.data_type.get(k,False) for k in ('third_view','depth','mesh_segmentation','actor_segmentation','pointcloud')):
            raise ValueError('Unvalidated observation modality')
        restored=[]
        total=post=inside=deferred=0
        def replace(obj,name,fn):
            restored.append((obj,name,name in vars(obj),vars(obj).get(name)))
            setattr(obj,name,fn)
        original_predict=policy.predict
        original_action=task.take_action
        original_obs=task.get_obs
        def predict(*a,**k):
            nonlocal total
            result=original_predict(*a,**k)
            # Match the unchanged official zip(actions[:step_size],delta[:...]).
            size=kwargs.get('step_size',args[3] if len(args)>3 else 32)
            total=min(len(result[0][:size]),len(result[1][:size]))
            return result
        def action(*a,**k):
            nonlocal inside
            inside+=1
            try:return original_action(*a,**k)
            finally:inside-=1
        def observation(*a,**k):
            nonlocal post,deferred
            if inside:
                return original_obs(*a,**k)  # Success-time internal observation.
            post+=1
            if not total or post>=total:
                return original_obs(*a,**k)
            # Retain every light/RNG update, pose sync and actual camera render.
            # Only unused intermediate CPU readback/conversion/getters are deferred.
            task._update_render()
            task.cameras.update_picture()
            deferred+=1
            return None  # Only the final value is returned by the official step.
        replace(policy,'predict',predict);replace(task,'take_action',action);replace(task,'get_obs',observation)
        try:
            result=step(task,policy,*args,**kwargs)
            if post!=total:raise RuntimeError('Official observation call contract changed')
            note('deferred_observation_materialization',post_action_calls=post,cpu_readbacks_deferred=deferred)
            return result
        finally:
            for obj,name,owned,old in reversed(restored):
                if owned:setattr(obj,name,old)
                else:delattr(obj,name)
    return optimized

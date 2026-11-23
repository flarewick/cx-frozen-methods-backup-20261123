"""Current-RGB actor query. No anchor/frame argument may replace observed RGB."""
import numpy as np
def query_current(base,head,robots,projection,current,instruction,previous,*,decision_tick):
    from cx_tta.eval_diagnostics import array_hash
    from native_stage_v1.cx_tta_stage.query import query
    if (type(decision_tick)is not int or decision_tick<0
            or current.get('observation_tick')!=decision_tick):
        raise ValueError('Exactly the already-arrived current observation required')
    rgb=current['rgb']
    if len(rgb)!=3 or any(not isinstance(im,np.ndarray) or im.dtype!=np.uint8 or im.ndim!=3 or im.shape[-1]!=3 for im in rgb):
        raise ValueError('Actual three-camera current RGB required')
    hashes={str(i):array_hash(im) for i,im in enumerate(rgb)}
    if current['images']!=hashes:
        raise ValueError('Actor RGB differs from current observation hash')
    images=tuple(im.copy() for im in rgb)
    result=query(base,head,robots,projection,current,images,instruction,previous)
    result.update(actor_images=hashes,actor_observation_tick=decision_tick,
                  actor_image_source='actual_current_RGB_not_initial_anchor')
    return result

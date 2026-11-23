"""Truthful current observer AND actor RGB, actual query-clock and once base."""
import numpy as np
from .paths import write
def save_history(folder,rows):
    with (folder/'observed_history.npz').open('xb') as f:
        np.savez_compressed(f,queries=np.arange(len(rows),dtype=np.int64),
            states=np.stack([r['command_state'] for r in rows]),
            base_chunks=np.stack([r['base'] for r in rows]),raw_deltas=np.stack([r['raw_delta'] for r in rows]),
            **{'legal_'+k:np.concatenate([r['legal'][k] for r in rows],axis=0) for k in rows[0]['legal']},
            **{'cam'+str(i):np.stack([r['rgb'][i] for r in rows]) for i in range(3)})
    write(folder/'actor_history.json',dict(queries=[{k:r[k] for k in ('query','actor_images','actor_observation_tick','actor_image_source')} for r in rows],
        currentRGB_is_actual_actor_input=True,no_legacy_anchor_substitution=True))
def save_query(folder,name,result,meta):
    with (folder/(name+'.npz')).open('xb') as f:
        np.savez_compressed(f,base=result['base'],raw_delta=result['raw_delta'],
           **{'legal_'+k:v for k,v in result['legal'].items()})
    write(folder/(name+'.json'),dict(meta,actor_images=result['actor_images'],
         actor_observation_tick=result['actor_observation_tick'],actor_image_source=result['actor_image_source']))

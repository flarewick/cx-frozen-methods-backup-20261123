"""Audited cache reader. Whole source scenes, not forks, are split groups."""
from pathlib import Path
import numpy as np
from .cache_recovery import read,verify_hashes


def load_fixed_cache(run, *, scope):
    if scope!='developer_same_task_scene_holdout':raise ValueError('This cache is not independent confirmation or task-transfer evidence')
    run=Path(run);a=read(run/'audit.json');p=read(run/'protocol.json')
    if not a['passed'] or a['schema']!='fixed_recovery_context_v1_state_div_pi_real_fork_history':raise ValueError('Audited causal cache required')
    verify_hashes(a['source_sha256']);verify_hashes(p['source_sha256'])
    with np.load(run/'data.npz',allow_pickle=False) as z:
        inputs={k:z[k].copy() for k in ('semantic','coarse','base')}
        labels={k:z[k].copy() for k in ('target','valid')}
    metadata=[]
    for row in p['rows']:
        # Existing trainers split on `group`; expose scene identity at that key.
        metadata.append(dict(task=row['task'],group=row['scene_group'],scene_seed=row['scene_seed'],
            branch_group=row['group'],source_point=row['source_point'],frame=row['native_count'],
            kind='recovery',pilot_split='recovery_fit',role=row['role'],source=row['source']))
    if len(metadata)!=a['training_ready_rows'] or sorted({r['group'] for r in metadata})!=a['independent_scenes']:
        raise ValueError('Scene grouping mismatch')
    return inputs,labels,metadata


def verify_scene_disjoint(train_rows, held_rows):
    train={(r['task'],r['scene_seed']) for r in train_rows};held={(r['task'],r['scene_seed']) for r in held_rows}
    if not train or not held or train & held:raise ValueError('Whole-scene nonempty disjoint split required; forks cannot cross splits')
    return True

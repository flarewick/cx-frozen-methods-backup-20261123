"""Offline effect supervision contract; privileged fields never feed deployed policy."""
import numpy as np


def articulation_state(task):
    result = {}
    for name, value in vars(task).items():
        if name.startswith('_'):
            continue
        objects = list(enumerate(value)) if isinstance(value, (list, tuple)) else [(None, value)]
        for index, obj in objects:
            if not (hasattr(obj, 'actor') and hasattr(obj, 'config') and callable(getattr(obj, 'get_qpos', None))):
                continue
            qpos = np.asarray(obj.get_qpos(), dtype=np.float32)
            if not qpos.size:
                continue
            limits = np.asarray(obj.get_qlimits(), dtype=np.float32)
            if qpos.ndim != 1 or limits.shape != (len(qpos), 2) or not np.isfinite(qpos).all() or not np.isfinite(limits).all():
                raise ValueError('Invalid articulated object state')
            key = name if index is None else f'{name}_{index}'
            result[key] = {'qpos': qpos, 'qlimits': limits}
    return result


def timeline(ticks, dense_ticks, dense_dt):
    ticks, dense_ticks, dense_dt = map(np.asarray, (ticks, dense_ticks, dense_dt))
    if (ticks.ndim != 1 or len(ticks) < 2 or ticks.dtype.kind not in 'iu'
        or dense_ticks.ndim != 1 or not len(dense_ticks) or dense_ticks.dtype.kind not in 'iu'
        or dense_dt.shape != dense_ticks.shape or not np.isfinite(dense_dt).all()
        or np.any(dense_dt <= 0) or np.any(np.diff(ticks) < 0)
        or np.any(np.diff(dense_ticks) != 1)
        or ticks[0] < dense_ticks[0] or ticks[-1] > dense_ticks[-1] + 1):
        raise ValueError('Invalid effect timeline or dense coverage')
    indexes = ticks - dense_ticks[0]
    clock = np.concatenate(([0.], np.cumsum(dense_dt, dtype=np.float64)))
    time = clock[indexes]
    return {'time_s': time, 'transition_dt_s': np.diff(time),
            'transition_valid': np.diff(ticks) > 0,
            'dense_start': indexes[:-1], 'dense_stop': indexes[1:]}


def add_timeline(h):
    values = timeline(h['cx_measurement/physics_tick'][:], h['cx_dense/tick_before'][:], h['cx_dense/dt_s'][:])
    for key, value in values.items():
        h.create_dataset('cx_effect/' + key, data=value)
    h['cx_effect'].attrs['time_origin'] = 'first recorded dense tick; integrate actual per-step dt'
    h['cx_effect'].attrs['usage'] = 'offline supervision only; zero-dt transitions excluded from effect learning'


def validate_effect(h):
    expected = timeline(h['cx_measurement/physics_tick'][:], h['cx_dense/tick_before'][:], h['cx_dense/dt_s'][:])
    for key, value in expected.items():
        stored = h['cx_effect/' + key][:]
        if stored.dtype != value.dtype or not np.array_equal(stored, value):
            raise ValueError('Effect timeline differs from dense trace: ' + key)
    count = len(expected['time_s'])
    actors = []
    if 'cx_effect/articulation' in h:
        for name, group in h['cx_effect/articulation'].items():
            q, limits = group['qpos'][:], group['qlimits'][:]
            if (q.ndim != 2 or q.shape[0] != count or not q.shape[1]
                or limits.shape != (*q.shape, 2) or not np.isfinite(q).all() or not np.isfinite(limits).all()):
                raise ValueError('Invalid stored articulation: ' + name)
            actors.append(name)
    return {'articulations': actors, 'positive_time_transitions': int(expected['transition_valid'].sum()),
            'zero_time_transitions': int((~expected['transition_valid']).sum()),
            'effect_training_ready': False}

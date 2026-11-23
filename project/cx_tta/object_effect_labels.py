"""Offline entity-aligned labels only; never concatenate these to policy inputs."""
import numpy as np


def rotation_delta(reference, candidate):
    """Shortest world-frame rotation vector, SAPIEN wxyz quaternion convention."""
    a, b = np.asarray(reference, float), np.asarray(candidate, float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(np.r_[a, b]).all():
        raise ValueError('Invalid quaternion')
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-12:
        raise ValueError('Zero quaternion')
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    w = b[0] * a[0] + b[1:] @ a[1:]
    v = -b[0] * a[1:] + a[0] * b[1:] - np.cross(b[1:], a[1:])
    if w < 0:
        w, v = -w, -v
    length = np.linalg.norm(v)
    return (2 * v if length < 1e-12 else 2 * np.arctan2(length, w) * v / length)


def entity_effects(reference, candidate, limits):
    """Keep every entity separately, including static/zero-effect entities."""
    out = {'rigid': {}, 'articulated': {}}
    for group in ('object_poses', 'articulation_qpos'):
        if set(reference[group]) != set(candidate[group]):
            raise ValueError('Entity association changed')
    for name, pose in reference['object_poses'].items():
        a, b = np.asarray(pose), np.asarray(candidate['object_poses'][name])
        if a.shape != (7,) or b.shape != (7,) or not np.isfinite(np.r_[a, b]).all():
            raise ValueError('Invalid pose')
        out['rigid'][name] = {'translation_world_m': (b[:3] - a[:3]).tolist(),
                              'rotation_world_rad': rotation_delta(a[3:], b[3:]).tolist()}
    for name, value in reference['articulation_qpos'].items():
        a, b = np.asarray(value), np.asarray(candidate['articulation_qpos'][name])
        lim = np.asarray(limits[name])
        if (a.shape != b.shape or a.ndim != 1 or lim.shape != (len(a), 2)
                or not np.isfinite(np.r_[a, b, lim.ravel()]).all()):
            raise ValueError('Invalid articulation')
        span = lim[:, 1] - lim[:, 0]
        valid = span > 1e-8
        normalized = np.zeros_like(a, dtype=float)
        normalized[valid] = (b - a)[valid] / span[valid]
        out['articulated'][name] = {'delta_native': (b - a).tolist(),
                                   'delta_range_fraction': normalized.tolist(),
                                   'range_valid_mask': valid.tolist()}
    return out


def labels_for_run(result):
    base = result['branches']['base']
    branches = {}
    for name, branch in result['branches'].items():
        limits = branch['articulation_limits_offline_only']
        branches[name] = {
            'from_start': entity_effects(branch['before']['offline_only'], branch['after']['offline_only'], limits),
            'versus_base': entity_effects(base['after']['offline_only'], branch['after']['offline_only'], limits),
            'executed': branch['executed'], 'duration_s': branch['transition']['duration_s'],
            'full_horizon': branch['transition']['full_horizon'],
            'success_after_offline_only': branch['success_after']}
    return {'before_offline_only': base['before']['offline_only'], 'branches': branches,
            'scope': 'entity names associate labels only, not model inputs; no success-improvement direction labels'}

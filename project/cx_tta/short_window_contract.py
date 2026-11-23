"""Passive short-window observations and separate simulator-only visibility labels."""
import numpy as np

from .audit import digest
from .collect_interventions import observation_signature
from .deploy import CAMERA_KEYS, unpack_observation
from .eval_diagnostics import array_hash, state_snapshot


def entity_ids(task):
    """Match recording.object_poses keys, including every articulation link."""
    scene_ids = {int(e.per_scene_id) for e in task.scene.entities}
    mapping = {}
    for name, value in vars(task).items():
        if name.startswith('_'):
            continue
        objects = list(enumerate(value)) if isinstance(value, (list, tuple)) else [(None, value)]
        for index, obj in objects:
            if not (hasattr(obj, 'actor') and hasattr(obj, 'config') and callable(getattr(obj, 'get_pose', None))):
                continue
            actor = obj.actor
            links = actor.get_links() if callable(getattr(actor, 'get_links', None)) else [actor]
            ids = sorted({int(getattr(link, 'entity', link).per_scene_id) for link in links})
            if not ids or any(i <= 0 or i not in scene_ids for i in ids):
                raise ValueError('Unknown entity/link segmentation ID: ' + name)
            key = name if index is None else f'{name}_{index}'
            mapping[key] = ids
    if set(mapping) != set(state_snapshot(task)['offline_only']['object_poses']):
        raise ValueError('Entity mapping differs from recorded physical labels')
    return mapping, scene_ids


def visibility_labels(actor_maps, mapping):
    maps = np.asarray(actor_maps)
    if maps.ndim != 3 or maps.dtype.kind not in 'ui' or maps.min() < 0:
        raise ValueError('Raw nonnegative integer segmentation required')
    result = {}
    for name, ids in mapping.items():
        if not ids or any(int(i) <= 0 for i in ids):
            raise ValueError('Unknown ID mapping must not become invisible')
        views, mask_hashes = [], []
        for actor_map in maps:
            mask = np.isin(actor_map, ids)
            y, x = np.nonzero(mask)
            count = len(x)
            views.append({'pixels': count, 'any_rendered_pixel': bool(count),
                          'pixel_support_ge16': count >= 16,
                          'bbox_xyxy_exclusive': [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if count else None})
            mask_hashes.append(array_hash(mask))
        result[name] = {'entity_ids': list(ids), 'views': views,
                        'visible_in_any_camera': any(v['any_rendered_pixel'] for v in views),
                        'semantic_mask_sha256': mask_hashes}
    return result


def validate_calibration(k, e, m):
    k, e, m = np.asarray(k), np.asarray(e), np.asarray(m)
    if e.shape == (3, 4):
        e = np.vstack([e, [0., 0., 0., 1.]])
    if (k.shape != (3, 3) or e.shape != (4, 4) or m.shape != (4, 4)
            or not np.isfinite(np.r_[k.ravel(), e.ravel(), m.ravel()]).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or abs(np.linalg.det(e[:3, :3])) < 1e-6
            or not np.allclose(e[3], [0, 0, 0, 1])):
        raise ValueError('Invalid camera calibration')
    return k.copy(), e.copy(), m.copy()


def capture_frame(task, observation, run, branch, offset, mapping, scene_ids):
    """Read existing Color/Segmentation buffers only: no render, step or get_obs."""
    before_tick = getattr(task, '_cx_intervention_ticks', 0)
    before_count = task.take_action_cnt
    images, command = unpack_observation(observation)
    cameras = {name: camera for name, camera in zip(task.cameras.static_camera_name, task.cameras.static_camera_list)}
    cameras.update(right_camera=task.cameras.right_camera, left_camera=task.cameras.left_camera)
    maps, intrinsics, extrinsics, model_matrices = [], [], [], []
    for name, image in zip(CAMERA_KEYS, images):
        camera = cameras[name]
        color = (camera.get_picture('Color') * 255).clip(0, 255).astype(np.uint8)[..., :3]
        if not np.array_equal(color, image):
            raise ValueError('Render buffer is not aligned with observation: ' + name)
        raw = np.asarray(camera.get_picture('Segmentation'))
        if raw.ndim != 3 or raw.shape[:2] != image.shape[:2] or raw.shape[2] < 2 or raw.dtype.kind not in 'ui':
            raise ValueError('Unsupported segmentation buffer')
        actor_map = raw[..., 1].copy()
        if not set(map(int, np.unique(actor_map))).issubset(scene_ids | {0}):
            raise ValueError('Rendered actor IDs do not match this scene')
        maps.append(actor_map.astype(np.uint32))  # Never truncate IDs to uint8.
        cfg = observation['observation'][name]
        k, e, m = validate_calibration(cfg['intrinsic_cv'], cfg['extrinsic_cv'], cfg['cam2world_gl'])
        intrinsics.append(k)
        extrinsics.append(e)
        model_matrices.append(m)
    actor_maps = np.stack(maps)
    label = visibility_labels(actor_maps, mapping)
    physical = state_snapshot(task)
    observed_path = run / f'{branch}_frame_{offset:+04d}_observed.npz'
    offline_path = run / f'{branch}_frame_{offset:+04d}_offline.npz'
    np.savez_compressed(observed_path, head_rgb=images[0], right_rgb=images[1], left_rgb=images[2],
                        command_state=command, command_state_raw=np.asarray(observation['joint_action']['vector']),
                        arm_qpos=np.asarray(physical['arm_qpos']), gripper_qpos=np.asarray(physical['gripper_qpos']),
                        intrinsic_cv=np.stack(intrinsics),
                        extrinsic_cv=np.stack(extrinsics), cam2world_gl=np.stack(model_matrices))
    np.savez_compressed(offline_path, actor_ids=actor_maps)
    if getattr(task, '_cx_intervention_ticks', 0) != before_tick or task.take_action_cnt != before_count:
        raise ValueError('Recorder advanced physics/actions')
    return {'offset': offset, 'action_count': before_count, 'physics_tick': before_tick,
            'physics_time_s': getattr(task, '_cx_intervention_time', 0.),
            'observation_signature': observation_signature(observation),
            'observed_file': observed_path.name, 'observed_sha256': digest(observed_path),
            'offline_file': offline_path.name, 'offline_sha256': digest(offline_path),
            'physical_offline_only': physical, 'visibility_offline_only': label}


def make_windows(frames, executed_mask, dt, ticks, stride):
    mask, dt, ticks = np.asarray(executed_mask), np.asarray(dt), np.asarray(ticks)
    if (mask.shape != (32,) or mask.dtype != np.bool_ or dt.shape != (32,) or ticks.shape != (32,)
            or ticks.dtype.kind not in 'iu' or not np.isfinite(dt).all() or np.any(dt < 0) or np.any(ticks < 0)
            or np.any(dt[~mask] != 0) or np.any(ticks[~mask] != 0)
            or np.any((dt == 0) != (ticks == 0)) or np.any(np.diff(mask.astype(int)) > 0)):
        raise ValueError('Invalid short-window timeline')
    if stride not in (4, 8):
        raise ValueError('Unsupported stride')
    offsets = [f['offset'] for f in frames]
    expected = {0, *range(stride, 33, stride)}
    if (stride not in (4, 8) or not expected.issubset(offsets) or offsets != sorted(set(offsets))
            or offsets[-1] != 32 or any(o < 0 and o != -stride for o in offsets)):
        raise ValueError('Missing/duplicate snapshot boundaries')
    out = []
    for a, b in zip(frames, frames[1:]):
        lo, hi = a['offset'], b['offset']
        if lo < 0:
            continue  # Historical prefix is input context, not seven new samples.
        count, duration, steps = int(mask[lo:hi].sum()), float(dt[lo:hi].sum()), int(ticks[lo:hi].sum())
        if (b['action_count'] - a['action_count'] != count or b['physics_tick'] - a['physics_tick'] != steps
                or not np.isclose(b['physics_time_s'] - a['physics_time_s'], duration, atol=1e-9)):
            raise ValueError('Snapshot and action clocks disagree')
        out.append({'start': lo, 'stop': hi, 'executed': count, 'duration_s': duration,
                    'positive_time': duration > 0, 'full_stride': hi - lo == stride and count == stride,
                    'same_initial_state_across_branches': lo == 0,
                    'label_scope': 'same-start intervention' if lo == 0 else 'branch-specific temporal transition'})
    return out

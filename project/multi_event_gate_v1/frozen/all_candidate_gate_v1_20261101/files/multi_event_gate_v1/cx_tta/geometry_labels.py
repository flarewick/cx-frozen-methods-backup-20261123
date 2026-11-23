"""Pure offline geometry contracts; actor IDs and truth never enter model inputs."""
import ast
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .probe_candidates import read
from .robot_kinematics import transform

ROLE_ACTORS = {
    'beat_block_hammer': ('hammer', 'block'), 'place_shoe': ('shoe', 'target_block'),
    'place_empty_cup': ('cup', 'coaster'), 'turn_switch': ('switch', None),
}


def poses7(values):
    values = np.asarray(values, float)
    if values.shape[-1] != 7 or not np.isfinite(values).all():
        raise ValueError('Expected finite xyz/wxyz poses')
    out = np.broadcast_to(np.eye(4), (*values.shape[:-1], 4, 4)).copy()
    out[..., :3, 3] = values[..., :3]
    q = values[..., 3:][..., [1, 2, 3, 0]]
    out[..., :3, :3] = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)
    return out


def local_points(config, key):
    values = np.asarray(config.get(key, []), float)
    if values.size == 0:
        return np.empty((0, 4, 4))
    if values.ndim != 3 or values.shape[1:] != (4, 4) or not np.isfinite(values).all():
        raise ValueError('Invalid point matrix')
    if not np.allclose(values[:, 3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('Invalid point homogeneous row')
    out = values.copy()
    out[:, :3, 3] *= np.asarray(config['scale'])
    return out


def box_points(runtime, task, actor):
    """Read literal box size and function-point definitions from actual task source."""
    code = ast.parse((runtime / 'envs' / (task + '.py')).read_text())
    call = next(n.value for n in ast.walk(code) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == actor for t in n.targets)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                and n.value.func.id == 'create_box')
    size = ast.literal_eval(next(v.value for v in call.keywords if v.arg == 'half_size'))
    source = ast.parse((runtime / 'envs/utils/create_actor.py').read_text())
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'create_box')
    data = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'data' for t in n.targets)
                and isinstance(n.value, ast.Dict))
    config = {'scale': size}
    for key in ('contact_points_pose', 'functional_matrix'):
        value = next(v for k, v in zip(data.keys, data.values) if isinstance(k, ast.Constant) and k.value == key)
        config[key] = ast.literal_eval(value)
    for n in ast.walk(code):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) and t.value.attr == 'config'
                        and isinstance(t.value.value, ast.Attribute) and t.value.value.attr == actor
                        and isinstance(t.slice, ast.Constant) and t.slice.value == 'functional_matrix'):
                    config['functional_matrix'] = ast.literal_eval(n.value)
    return config


def fixed_link_transform(urdf, link, scale):
    """A missing moving-joint pose cannot be substituted with q=0."""
    tree = ET.parse(urdf).getroot()
    joints = {j.find('child').get('link'): j for j in tree.findall('joint')}
    out = np.eye(4); chain = []; seen = set(); node = link
    while node in joints:
        if node in seen:
            raise ValueError('Cyclic actor URDF')
        seen.add(node); joint = joints[node]
        if joint.get('type') != 'fixed':
            return None
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        chain.append(transform(xyz * scale, rpy)); node = joint.find('parent').get('link')
    for t in reversed(chain):
        out = out @ t
    return out


def switch_assets(folder):
    return sorted((p for p in folder.iterdir() if p.is_dir() and p.name != 'visual'), key=lambda p: int(p.name))


def switch_points(folder):
    config = read(folder / 'model_data.json'); points = []; files = [folder / 'model_data.json', folder / 'mobility.urdf']
    for item in config['contact_points']:
        base = fixed_link_transform(folder / 'mobility.urdf', item['base'], config['scale'])
        if base is None:
            points.append(None)
        else:
            local = np.asarray(item['matrix'], float).copy(); local[:3, 3] *= config['scale']
            points.append(base @ local)
    return points, files


def episode_definition(runtime, task, info):
    """Task/asset mapping is an offline label annotation, not a policy/router."""
    files = [runtime / 'envs' / (task + '.py'), runtime / 'envs/utils/actor_utils.py', runtime / 'envs/utils/create_actor.py']
    assets = runtime / 'assets/objects'
    actors = ROLE_ACTORS[task]; definitions = []
    for slot, actor in enumerate(actors):
        if actor is None:
            definitions.append(None); continue
        if actor in ('block', 'target_block'):
            config = box_points(runtime, task, actor)
        else:
            key = '{B}' if actor == 'coaster' else '{A}'
            name, base = info[key].split('/'); index = int(base.removeprefix('base'))
            if actor == 'switch':
                folder = switch_assets(assets / name)[index]
                points, deps = switch_points(folder); files += deps
                # Nominal official contact, not articulated success/progress.
                definitions.append(dict(actor=actor, contact_local=points,
                                        functional_local=points[0], semantic='nominal_official_contact_on_fixed_link',
                                        asset=str(folder), config=read(folder / 'model_data.json'),
                                        articulation_label_valid=False))
                continue
            path = assets / name / ('model_data' + str(index) + '.json'); config = read(path); files.append(path)
        cp = local_points(config, 'contact_points_pose'); fp = local_points(config, 'functional_matrix')
        functional_id = 1 if actor == 'block' else 0
        definitions.append(dict(actor=actor, contact_local=list(cp), functional_local=fp[functional_id] if len(fp) > functional_id else None,
                                semantic='functional_point_' + str(functional_id), config=config,
                                articulation_label_valid=False))
    return definitions, files


def labels_from_poses(root, definitions):
    """Two semantic slots: manipulated item/tool and destination. SI units."""
    n = len(root); output = dict(root_world=root, root_valid=np.zeros((n, 2), bool),
        functional_world=np.broadcast_to(np.eye(4), (n, 2, 4, 4)).copy(), functional_valid=np.zeros((n, 2), bool),
        contact_world=np.broadcast_to(np.eye(4), (n, 2, 4, 4, 4)).copy(), contact_valid=np.zeros((n, 2, 4), bool))
    for slot, definition in enumerate(definitions):
        if definition is None:
            continue
        output['root_valid'][:, slot] = True
        if definition['functional_local'] is not None:
            output['functional_world'][:, slot] = root[:, slot] @ definition['functional_local']
            output['functional_valid'][:, slot] = True
        if len(definition['contact_local']) > 4:
            raise ValueError('Schema capacity exceeded; no truncation allowed')
        for j, local in enumerate(definition['contact_local']):
            if local is not None:
                output['contact_world'][:, slot, j] = root[:, slot] @ local
                output['contact_valid'][:, slot, j] = True
    a, b = output['functional_world'][:, 0], output['functional_world'][:, 1]
    output['relation_valid'] = output['functional_valid'].all(1)
    output['relation_world_m'] = b[:, :3, 3] - a[:, :3, 3]
    output['relation_source_frame_m'] = np.einsum('ni,nij->nj', output['relation_world_m'], a[:, :3, :3])
    output['relation_rotation'] = a[:, :3, :3].swapaxes(-1, -2) @ b[:, :3, :3]
    output['articulation_valid'] = np.zeros(n, bool)
    return output


def regression_labels(labels):
    """World geometry decoded once; wrist-relative geometry uses analytic transforms."""
    r = labels['root_world']; f = labels['functional_world']; rv = labels['root_valid']; fv = labels['functional_valid']
    y = np.concatenate([r[..., :3, 3].reshape(len(r), 6) / .1,
                        f[..., :3, 3].reshape(len(r), 6) / .1,
                        r[..., :3, :2].reshape(len(r), 12)], 1).astype(np.float32)
    valid = np.concatenate([np.repeat(rv, 3, axis=1), np.repeat(fv, 3, axis=1), np.repeat(rv, 6, axis=1)], 1)
    y[~valid] = 0.
    return y, valid


def relative_points(world, ee):
    """Row-vector inverse rigid transform, preserving the chosen command-FK frame."""
    return np.einsum('...i,...ij->...j', world - ee[..., :3, 3], ee[..., :3, :3])

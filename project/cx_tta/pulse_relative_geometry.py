"""Current/history privileged 3D geometry control for the frozen schema-4 dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .audit import digest
from .paths import new_run, write_json
from .robot_kinematics import RobotFK


def relative_features(object_pose, tips, joint=None, limits=None):
    """Object origin and orientation in each FK control-link frame (not surface gap)."""
    value = np.asarray(object_pose)
    rotation = Rotation.from_quat(value[3:][[1, 2, 3, 0]]).as_matrix()
    features = []
    for tip in tips:
        features.extend(tip[:3, :3].T @ (value[:3] - tip[:3, 3]))
        features.extend((tip[:3, :3].T @ rotation)[:, :2].ravel())
    if joint is None:
        features.extend([0., 0.])
    else:
        if len(joint) != 1 or len(limits) != 1:
            raise ValueError('Only single-DOF teacher state supported')
        lo, hi = limits[0]
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError('Invalid joint range')
        features.extend([(joint[0] - lo) / (hi - lo), 1.])
    return np.asarray(features, dtype=np.float32)


def build(dataset):
    index = json.loads((dataset / 'index.json').read_text())
    aggregate_path = Path(index['dataset']) / 'index.json'
    if digest(aggregate_path) != index['dataset_index_sha256']:
        raise ValueError('Aggregate changed')
    aggregate = json.loads(aggregate_path.read_text())
    for file, key in [('inputs_oracle.npz', 'inputs_sha256'), ('labels_offline.npz', 'labels_sha256')]:
        if digest(dataset / file) != index[key]:
            raise ValueError('Dataset changed')
    states, provenance = {}, []
    for sid, source in enumerate(aggregate['sources']):
        run = Path(source['run'])
        if digest(run / 'result.json') != source['result_sha256'] or digest(run / 'manifest.json') != source['manifest_sha256']:
            raise ValueError('Source changed')
        result = json.loads((run / 'result.json').read_text())
        manifest = json.loads((run / 'manifest.json').read_text())
        config_path = Path(manifest['source']) / 'resolved_config.json'
        if digest(config_path) != manifest['source_config_sha256']:
            raise ValueError('Robot config changed')
        robots = [RobotFK(json.loads(config_path.read_text()), side) for side in ('left', 'right')]
        branch = result['branches']['base']
        frames = {f['offset']: f for f in branch['frames']}
        start = manifest['pulse_start']
        objects = {}
        for offset in (start, start - 4):
            frame = frames[offset]
            physical = frame['physical_offline_only']
            truth = physical['offline_only']
            tips = [robot.pose(q) for robot, q in zip(robots, physical['arm_qpos'])]
            for entity, pose in truth['object_poses'].items():
                objects.setdefault(entity, []).append(relative_features(pose, tips, truth['articulation_qpos'].get(entity), branch['articulation_limits_offline_only'].get(entity)))
        for entity, values in objects.items():
            if len(values) != 2:
                raise ValueError('Missing historical entity')
            states[sid, entity] = np.concatenate(values)
        if frames[start]['physics_time_s'] <= frames[start - 4]['physics_time_s']:
            raise ValueError('Nonpositive history interval')
        provenance.append(dict(source_index=sid, result_sha256=source['result_sha256'], offsets=[start, start - 4], history_dt_s=frames[start]['physics_time_s'] - frames[start - 4]['physics_time_s'], config_sha256=digest(config_path), urdf_sha256=digest(robots[0].urdf)))
    with np.load(dataset / 'inputs_oracle.npz') as z:
        data = dict(z)
    data['geometry'][:, :84] = 0
    for i, row in enumerate(index['records']):
        if row['split'] != 'train':
            raise ValueError('Original validation prohibited')
        data['geometry'][i, :40] = states[row['source_index'], row['entity']]
    if not all(np.isfinite(x).all() for x in data.values()):
        raise ValueError('Nonfinite features')
    run = new_run('pulse_relative_geometry', group=('effect_learning',))
    np.savez_compressed(run / 'inputs_oracle.npz', **data)
    (run / 'labels_offline.npz').write_bytes((dataset / 'labels_offline.npz').read_bytes())
    write_json(run / 'index.json', dict(index, inputs_sha256=digest(run / 'inputs_oracle.npz'), parent_dataset=str(dataset), parent_index_sha256=digest(dataset / 'index.json'), geometry_provenance=provenance, scope='PRIVILEGED diagnostic: replace only 84D image relation block by 40D current/history object poses in each measured control-link frame plus zero padding. Same vision, actions, robot state, future known base plan, labels and model width. No future realized state, task IDs or contact labels. Not deployable.'))
    with np.load(dataset / 'inputs_oracle.npz') as z:
        assert np.array_equal(data['vision'], z['vision']) and np.array_equal(data['action'], z['action'])
        assert np.array_equal(data['geometry'][:, 84:], z['geometry'][:, 84:])
    assert digest(run / 'labels_offline.npz') == index['labels_sha256']
    write_json(run / 'interface_audit.json', dict(status='passed', sources=len(provenance), rows=len(index['records']), offsets=[8, 4], unchanged=['vision', 'action', 'geometry[84:]', 'labels', 'records'], modified='geometry[:84] only', code_sha256=digest(Path(__file__))))
    return run


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    a = p.parse_args()
    print(build(a.dataset), flush=True)

if __name__ == '__main__':
    main()

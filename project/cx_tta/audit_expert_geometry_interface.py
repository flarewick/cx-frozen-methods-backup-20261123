"""S2g: read-only geometry/observation alignment on the frozen 32 expert scenes.

This audit does not generate functional-goal labels or train a policy. Actor root
poses and measured joints remain offline diagnostics, separate from command state.
"""
import argparse
import ast
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np

from .audit import digest
from .audit_pose_capacity import angles, numpy_fk
from .paths import ROOT, output_path, write_json
from .probe_candidates import read, verify
from .robot_kinematics import RobotFK

FEATURES = ROOT / 'outputs/feature_caches/joint_prefix_20260930_141244_699126'
CAMERAS = ('head_camera', 'right_camera', 'left_camera')
ARMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value, message):
    require(np.isfinite(value).all(), message)
    return value


def stats(value):
    return dict(mean=float(np.mean(value)), p95=float(np.percentile(value, 95)),
                maximum=float(np.max(value)))


def audit(run):
    run = output_path(run)
    protocol = read(run / 'protocol.json')
    verify(protocol)
    inventory = read(run / 'expert_geometry_inventory.json')
    verify(inventory)
    experts = protocol['fit_experts']
    groups = {r['group'] for r in experts}
    require(len(experts) == 572 and len(groups) == 32, 'Frozen expert scope changed')
    require(groups == {r['group'] for r in inventory['records']}, 'Inventory scope changed')
    records = {r['source']['group']: r for r in read(FEATURES / 'result.json')['records']
               if r['source']['kind'] == 'expert' and r['source']['group'] in groups}
    require(set(records) == groups, 'Missing expert feature source')
    sources = {str(run / 'protocol.json'): digest(run / 'protocol.json'),
               str(run / 'expert_geometry_inventory.json'): digest(run / 'expert_geometry_inventory.json'),
               str(FEATURES / 'result.json'): digest(FEATURES / 'result.json'),
               str(Path(__file__)): digest(Path(__file__)),
               str(ROOT / 'cx_tta/robot_kinematics.py'): digest(ROOT / 'cx_tta/robot_kinematics.py'),
               str(ROOT / 'cx_tta/audit_pose_capacity.py'): digest(ROOT / 'cx_tta/audit_pose_capacity.py')}
    output, all_position, all_angle = [], [], []
    for item in inventory['records']:
        record = records[item['group']]
        row = record['source']
        require(row['split'] == 'train' and row['pilot_split'] == 'fit', 'Non-training expert opened')
        require(row['source'] == item['source'] and row['source_sha256'] == item['sha256'], 'HDF5 lineage mismatch')
        paths = {row['source']: row['source_sha256'], row['cache']: row['cache_sha256'],
                 row['label']: row['label_sha256'], record['file']: record['sha256']}
        collection = Path(row['collection'])
        config_path = collection / 'resolved_config.json'
        paths[str(config_path)] = row['config_sha256']
        instrumentation = read(collection / 'instrumentation.json')
        sources[str(collection / 'instrumentation.json')] = digest(collection / 'instrumentation.json')
        runtime_base = collection / 'robotwin/envs/_base_task.py'
        paths[str(runtime_base)] = instrumentation['instrumented_base_sha256']
        # The helper evolved after schema-2 collection. Do not conflate its
        # present source with the historic wrapper, or silently bypass data hashes.
        current_recording = ROOT / 'cx_tta/recording.py'
        sources[str(current_recording)] = digest(current_recording)
        historical_wrapper_available = digest(current_recording) == instrumentation['recording_sha256']
        for name, sha in paths.items():
            require(digest(Path(name)) == sha, 'Changed geometry source: ' + name)
        sources.update(paths)
        tree = ast.parse(runtime_base.read_text())
        observed_calls = {ast.unparse(n.func) for n in ast.walk(next(n for n in ast.walk(tree)
                          if isinstance(n, ast.FunctionDef) and n.name == 'get_obs')) if isinstance(n, ast.Call)}
        require('self.scene.step' not in observed_calls and '_cx_record_step' not in observed_calls,
                'get_obs directly advances physics')
        bots = [RobotFK(read(config_path), side) for side in ('left', 'right')]
        for bot in bots:
            require(digest(bot.urdf) == row['urdf_sha256'], 'URDF changed')
            sources[str(bot.urdf)] = digest(bot.urdf)
        with np.load(row['cache']) as z:
            frames = z['frame_index'].copy()
            context = z['context'][:, :14].copy()
        with np.load(row['label']) as z:
            require(np.array_equal(frames, z['source_frame_index']), 'Label frame alignment changed')
            measured_label = z['current_arm_qpos'].copy()
        with np.load(record['file']) as z:
            require(np.array_equal(frames, z['frames']), 'Feature frame alignment changed')
        selected = np.array([r['frame'] for r in experts if r['group'] == item['group']], dtype=np.int64)
        require(np.array_equal(selected, frames), 'Selected frozen training rows changed')
        with h5py.File(row['source'], 'r') as h:
            require(str(h.attrs['task']) == row['task'] and int(h.attrs['seed']) == row['seed'], 'Scene identity mismatch')
            require(str(h.attrs['instruction']) == row['instruction'], 'Instruction mismatch')
            commands = finite(h['joint_action/vector'][:], 'Nonfinite command state')
            n = len(commands)
            require(frames.ndim == 1 and np.all((frames >= 0) & (frames < n)), 'Invalid frame indices')
            measured = finite(h['cx_measurement/arm_qpos'][:], 'Nonfinite measured joints')
            require(measured.shape == (n, 2, 6), 'Measured joint shape mismatch')
            require(np.array_equal(measured[frames], measured_label), 'Measured label alignment mismatch')
            scaled = commands[frames].astype(np.float32)
            scaled[:, ARMS] /= np.pi
            require(np.array_equal(scaled, context), 'Legal command-state alignment mismatch')
            ticks = h['cx_measurement/physics_tick'][:]
            dense_ticks = h['cx_dense/tick_before'][:]
            require(len(ticks) == n and np.all(np.diff(ticks) >= 0), 'Invalid observation clock')
            require(np.all(np.diff(dense_ticks) == 1) and dense_ticks[0] <= ticks[0]
                    and dense_ticks[-1] + 1 >= ticks[-1], 'Dense timeline does not cover observations')
            require(np.all(h['cx_measurement/physics_dt_s'][:] > 0)
                    and np.all(h['cx_dense/dt_s'][:] > 0), 'Invalid time steps')
            pose_error, quaternion_error, entities = 0., 0., {}
            for name, ds in h['cx_privileged'].items():
                poses = finite(ds[:], 'Nonfinite actor pose')
                require(poses.shape == (n, 7), 'Actor pose shape mismatch')
                error = float(np.max(np.abs(np.linalg.norm(poses[:, 3:], axis=1) - 1.)))
                require(error < 2e-6, 'Non-unit actor quaternion')
                quaternion_error = max(quaternion_error, error)
                entities[name] = dict(initial_root_world_xyz_m=poses[0, :3].tolist(),
                                     cached_root_world_xyz_min_m=poses[frames, :3].min(0).tolist(),
                                     cached_root_world_xyz_max_m=poses[frames, :3].max(0).tolist())
            images = 0
            for camera in CAMERAS:
                pre = 'observation/' + camera + '/'
                require(len(h[pre + 'rgb']) == n, 'RGB frame length mismatch')
                k = finite(h[pre + 'intrinsic_cv'][:], 'Nonfinite intrinsic')
                e = finite(h[pre + 'extrinsic_cv'][:], 'Nonfinite extrinsic')
                c = finite(h[pre + 'cam2world_gl'][:], 'Nonfinite camera pose')
                require(k.shape == (n, 3, 3) and e.shape == (n, 3, 4) and c.shape == (n, 4, 4), 'Calibration shape mismatch')
                require(np.all(k[:, 0, 0] > 0) and np.all(k[:, 1, 1] > 0), 'Invalid focal length')
                expected = (np.diag([1., -1., -1., 1.]) @ np.linalg.inv(c.astype(float)))[:, :3]
                error = float(np.max(np.abs(expected - e)))
                require(error < 2e-6, 'GL/CV calibration convention mismatch')
                pose_error = max(pose_error, error)
                for f in {int(frames[0]), int(frames[len(frames) // 2]), int(frames[-1])}:
                    image = cv2.imdecode(np.asarray(h[pre + 'rgb'][f], np.uint8), cv2.IMREAD_COLOR)
                    require(image is not None and image.shape == (240, 320, 3), 'Invalid stored JPEG dimensions')
                    images += 1
            command_pose = np.stack([numpy_fk(b, commands[frames, start:start + 6].astype(float))
                                     for b, start in zip(bots, (0, 7))], axis=1)
            measured_pose = np.stack([numpy_fk(b, measured[frames, arm].astype(float))
                                      for arm, b in enumerate(bots)], axis=1)
            position = np.linalg.norm(command_pose[..., :3, 3] - measured_pose[..., :3, 3], axis=-1) * 100.
            angle = angles(command_pose, measured_pose)
            all_position.append(position); all_angle.append(angle)
            output.append(dict(group=item['group'], task=row['task'], cached_rows=len(frames),
                               total_frames=n, decoded_images=images, actor_roots=entities,
                               camera_convention_max_error=pose_error, quaternion_norm_max_error=quaternion_error,
                               zero_tick_transitions=int(np.sum(np.diff(ticks) == 0)),
                               command_vs_measured_fk_position_cm=stats(position),
                               command_vs_measured_fk_angle_deg=stats(angle),
                               direct_get_obs_physics_step=False,
                               collection_recording_sha256=instrumentation['recording_sha256'],
                               historical_wrapper_available=historical_wrapper_available))
    require(sum(r['cached_rows'] for r in output) == 572, 'Incomplete selected rows')
    verify(dict(source_sha256=sources))
    report = dict(status='complete_readonly_index_and_geometry_audit', passed=True,
                  complete_historical_capture_implementation_parity=all(r['historical_wrapper_available'] for r in output),
                  expert_scenes=32, cached_rows=572,
                  task_scenes=dict(Counter(r['task'] for r in output)), records=output,
                  command_vs_measured_fk_position_cm=stats(np.concatenate(all_position)),
                  command_vs_measured_fk_angle_deg=stats(np.concatenate(all_angle)),
                  source_sha256=sources,
                  scope='Only original gate-fit training experts; no held calibration/confirmation scenes, new cache, or training.',
                  limitations=['Frame IDs, physics ticks, source hashes and historical non-stepping get_obs are checked; this is not full capture-code replay.',
                               'The current recording helper evolved after collection. Historical helper hashes are recorded; its exact source snapshot is unavailable here.',
                               'RGB decodes sample first/middle/last cached frames per camera; hashes cover entire sources.',
                               'Actor root is not an object contact point, articulated joint state, functional task goal, or recovery-success label.',
                               'Command FK is the frozen observable contract; measured FK only quantifies tracking differences offline.',
                               'Multiple frames in a scene do not add independent geometric placements.'])
    write_json(run / 'expert_geometry_alignment.json', report)
    print('EXPERT_GEOMETRY_AUDIT_PASSED', report['task_scenes'], 'rows', report['cached_rows'], flush=True)
    print('COMMAND_VS_MEASURED_FK', report['command_vs_measured_fk_position_cm'],
          report['command_vs_measured_fk_angle_deg'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    audit(parser.parse_args().run.resolve())

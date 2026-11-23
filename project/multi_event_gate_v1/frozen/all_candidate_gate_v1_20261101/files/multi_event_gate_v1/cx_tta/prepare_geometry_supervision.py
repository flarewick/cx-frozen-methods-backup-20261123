"""S2h: audited offline geometry from existing 32 expert scenes and 20 recovery rows."""
from pathlib import Path
import numpy as np
import h5py
from threadpoolctl import threadpool_limits
from .paths import ROOT, new_run, write_json
from .audit import digest
from .probe_candidates import read, verify
from .robot_kinematics import RobotFK
from .structured_features import pack_observable, load_structured
from .geometry_labels import (ROLE_ACTORS, poses7, episode_definition, labels_from_poses,
                              regression_labels, relative_points, switch_assets, switch_points)

SOURCE = ROOT / 'outputs/feature_caches/structured_readout_20261003_065755_140725'
FEATURES = ROOT / 'outputs/feature_caches/joint_prefix_20260930_141244_699126'
CONDITIONS = ('recovery_only_omit', 'expert_plus_omit', 'expert_plus_include')


def background_weights(rows):
    weights = np.zeros(len(rows), float)
    tasks = sorted({r['task'] for r in rows})
    for task in tasks:
        scenes = {r['scene_group'] for r in rows if r['task'] == task}
        for scene in scenes:
            ids = [i for i, r in enumerate(rows) if r['scene_group'] == scene]
            weights[ids] = 1 / (len(tasks) * len(scenes) * len(ids))
    return weights


def prepare():
    p, recovery_inputs = load_structured(SOURCE)
    verify(read(SOURCE / 'completion.json'))
    audit = read(SOURCE / 'expert_geometry_alignment.json'); verify(audit)
    if not audit['passed']:
        raise ValueError('Expert geometry audit required')
    experts = p['fit_experts']; wanted = {r['group'] for r in experts}
    records = [r for r in read(FEATURES / 'result.json')['records']
               if r['source']['kind'] == 'expert' and r['source']['group'] in wanted]
    require = lambda value, msg: None if value else (_ for _ in ()).throw(ValueError(msg))
    require(len(experts) == 572 and len(wanted) == 32 and len(records) == 32, 'Expert scope')
    run = new_run('geometry_supervision', group=('feature_caches',)); print('RUN', run, flush=True)
    modules = ('geometry_labels.py', 'prepare_geometry_supervision.py', 'geometry_readout.py',
               'train_geometry_readout.py', 'audit_geometry_readout.py')
    files = [ROOT / 'cx_tta' / f for f in modules]
    files += [SOURCE / f for f in ('protocol.json', 'projection.npz', 'probe_labels.npz', 'probe_protocol.json',
              'expert_geometry_alignment.json', 'completion.json')]
    files += [FEATURES / 'result.json', ROOT / 'scripts/train_geometry_readout.sh']
    sources = {str(f): digest(f) for f in files}
    write_json(run / 'design.json', dict(status='frozen_before_geometry_build',
        scope='32 original training expert scenes/572 rows plus 20 cached switch recovery rows/3 developer scenes only.',
        schema='Two semantic slots: manipulated item/tool and destination; SI world roots, official functional/nominal contact, all contact candidates, explicit validity, relative geometry.',
        coordinates='Recorded root already includes actor creation transforms; scale only local translations once. Articulated points require known fixed URDF chain; moving-link points without joints invalid. Command FK is distinct from measured joints.',
        recovery='Existing teacher nominal-contact reconstruction is the recovery label; asset candidate matching is a consistency diagnostic, not recorded asset-ID truth. No inferred asset ID enters model.',
        inputs='Frozen expert-only PCA32, current fused prefix and native command state/FK. Experts have no cached exact 32-native-action past: has_history=0, past fields zero. Recovery keeps its actual past. No task/asset/GT/phase input.',
        experiment='Same conditional geometry head and true initial weights, 600 updates/AdamW .0003/decay .01/3 seeds. Fixed expert and scene/branch/row recovery weights; only selected expert/held-recovery loss weights change, never redistribute.',
        conditions=list(CONDITIONS),
        claims='Offline geometry readout only. New common target/coordinate contract cannot be compared to S2g as a single-factor expert-data effect; source-weight contrast within S2h provides that control.',
        source_sha256=sources))
    with np.load(SOURCE / 'projection.npz') as z:
        pre = {k: z[k].copy() for k in z.files}
    input_blocks = []; label_blocks = []; camera_blocks = []; rows = []; definitions = []; original_states = []; ee_blocks = []
    for record in records:
        row = record['source']; collection = Path(row['collection']); runtime = collection / 'robotwin'
        sources.update({record['file']: record['sha256'], row['source']: row['source_sha256']})
        for f in (collection / 'attempts.json', collection / 'resolved_config.json', collection / 'instrumentation.json'):
            sources[str(f)] = digest(f)
        attempts = read(collection / 'attempts.json'); attempt = next(x for x in attempts if x['seed'] == row['seed'])
        require(attempt['replay_success'] and attempt['format_valid'] and attempt['split'] == 'train', 'Unverified expert')
        require(Path(attempt['hdf5']) == Path(row['source']), 'Expert scene identity')
        dd, deps = episode_definition(runtime, row['task'], attempt['scene_info']['info'])
        sources.update({str(f): digest(f) for f in deps})
        with np.load(record['file']) as z:
            frames = z['frames'].copy(); joint = z['joint'].copy()
        expected = [r['frame'] for r in experts if r['group'] == row['group']]
        require(expected == frames.tolist(), 'Expert frame order')
        bots = [RobotFK(read(collection / 'resolved_config.json'), side) for side in ('left', 'right')]
        with h5py.File(row['source'], 'r') as h:
            state = h['joint_action/vector'][:][frames].astype(np.float32)
            root = np.broadcast_to(np.eye(4), (len(frames), 2, 4, 4)).copy()
            for slot, actor in enumerate(ROLE_ACTORS[row['task']]):
                if actor:
                    root[:, slot] = poses7(h['cx_privileged/' + actor][:][frames])
            labels = labels_from_poses(root, dd)
            cam = np.stack([h['observation/' + c + '/extrinsic_cv'][:][frames]
                            for c in ('head_camera', 'right_camera', 'left_camera')], 1)
            roots_cam = np.einsum('ncki,nsi->ncsk', cam, np.concatenate([root[..., :3, 3], np.ones((len(frames), 2, 1))], -1))
            funcs_cam = np.einsum('ncki,nsi->ncsk', cam, np.concatenate([labels['functional_world'][..., :3, 3], np.ones((len(frames), 2, 1))], -1))
            camera_blocks.append(dict(root_camera_cv_m=roots_cam, functional_camera_cv_m=funcs_cam,
                                      calibration_valid=np.ones((len(frames), 3), bool)))
        x = pack_observable(joint, state, np.zeros_like(joint), np.zeros_like(state), np.zeros(len(state), bool), pre, bots)
        input_blocks.append(x); label_blocks.append(labels); original_states.append(state)
        from .audit_pose_capacity import numpy_fk
        ee = np.stack([numpy_fk(b, state[:, start:start + 6].astype(float)) for b, start in zip(bots, (0, 7))], 1)
        ee_blocks.append(ee)
        for f in frames:
            rows.append(dict(kind='expert', task=row['task'], scene_group=row['group'], group=row['group'],
                             frame=int(f), source=row['source'], instruction=row['instruction']))
        definitions.append(dict(group=row['group'], task=row['task'], info=attempt['scene_info']['info'],
            runtime=str(runtime),
            actors=[None if d is None else dict(actor=d['actor'], semantic=d['semantic'],
                contact_candidates=len(d['contact_local']), functional_valid=d['functional_local'] is not None,
                config=d['config'], asset=d.get('asset'),
                articulation_label_valid=d['articulation_label_valid']) for d in dd]))
    require(len(rows) == 572, 'Incomplete experts')
    ii = np.array([i for i, r in enumerate(p['rows']) if r['task'] == 'turn_switch'])
    rec_rows = [p['rows'][i] for i in ii]
    with np.load(SOURCE / 'probe_labels.npz') as z:
        rec = {k: z[k].copy() for k in z.files}
    require(np.array_equal(ii, rec['indices']), 'Recovery row order')
    diagnostic = Path(p['diagnostic']); diag = read(diagnostic / 'result.json'); inv = read(diagnostic / 'invariants.json')
    sources.update({str(diagnostic / f): digest(diagnostic / f) for f in ('result.json', 'invariants.json', 'audit.json')})
    roots = np.broadcast_to(np.eye(4), (20, 2, 4, 4)).copy(); candidates = []
    for j, row in enumerate(rec_rows):
        branch = next(b for b in diag['branches'] if b['run'] == row['source'])
        root = poses7(branch['switch_root_pose']); roots[j, 0] = root
        # Validate the nominal point against all available fixed-link asset definitions.
        source = Path(read(Path(row['source']) / 'manifest.json')['source']); asset = source / 'robotwin/assets/objects/056_switch'
        options = []
        for index, folder in enumerate(switch_assets(asset)):
            points, deps = switch_points(folder); sources.update({str(f): digest(f) for f in deps})
            if points[0] is not None:
                difference = float(np.linalg.norm((root @ points[0])[:3, 3] - rec['contacts_world'][j]))
                options.append(dict(index=index, folder=folder.name, error_m=difference))
        matches = [v for v in options if v['error_m'] < 1e-4]
        require(len(matches) == 1, 'Recovery nominal contact has no unique asset consistency match')
        # IDs inferred from label geometry are diagnostic only; do not use them to generate labels or inputs.
        candidates.append(dict(row=j, scene=row['scene_group'], nominal_point_matches=matches,
                               recorded_asset_id_available=False, all_candidates=options))
        sources[str(Path(row['source']) / 'fork.json')] = digest(Path(row['source']) / 'fork.json')
    rec_labels = labels_from_poses(roots, [dict(functional_local=None, contact_local=[]), None])
    rec_labels['functional_world'][:, 0] = roots[:, 0]
    rec_labels['functional_world'][:, 0, :3, 3] = rec['contacts_world']
    rec_labels['functional_valid'][:, 0] = True
    rec_labels['contact_world'][:, 0, 0] = rec_labels['functional_world'][:, 0]
    rec_labels['contact_valid'][:, 0, 0] = True
    # Recovery point orientation is unknown: root orientation is a bookkeeping carrier, NOT a contact-frame label.
    input_blocks.append({k: v[ii] for k, v in recovery_inputs.items()}); label_blocks.append(rec_labels)
    camera_blocks.append(dict(root_camera_cv_m=np.zeros((20, 3, 2, 3)), functional_camera_cv_m=np.zeros((20, 3, 2, 3)),
                              calibration_valid=np.zeros((20, 3), bool)))
    ee_blocks.append(None)
    paired = read(Path(p['paired']) / 'protocol.json'); bots = [RobotFK(read(Path(paired['config'])), s) for s in ('left', 'right')]
    cp = read(Path(p['cache']) / 'protocol.json')
    with np.load(Path(p['cache']) / 'features.npz') as z:
        state = z['state'][np.array(cp['feature_current_indices'])[ii]].copy()
    ee_blocks[-1] = np.stack([numpy_fk(b, state[:, start:start + 6].astype(float)) for b, start in zip(bots, (0, 7))], 1)
    original_states.append(state)
    rows += [dict(r, kind='recovery') for r in rec_rows]
    x = {k: np.concatenate([b[k] for b in input_blocks], 0) for k in input_blocks[0]}
    labels = {k: np.concatenate([b[k] for b in label_blocks], 0) for k in label_blocks[0]}
    labels.update({k: np.concatenate([b[k] for b in camera_blocks], 0) for k in camera_blocks[0]})
    labels['functional_orientation_valid'] = labels['functional_valid'].copy(); labels['functional_orientation_valid'][572:, 0] = False
    labels['contact_orientation_valid'] = labels['contact_valid'].copy(); labels['contact_orientation_valid'][572:, 0, 0] = False
    labels['command_ee_world'] = np.concatenate(ee_blocks); labels['command_state'] = np.concatenate(original_states)
    labels['root_relative_command_m'] = relative_points(labels['root_world'][..., :3, 3][:, None], labels['command_ee_world'][:, :, None])
    labels['functional_relative_command_m'] = relative_points(labels['functional_world'][..., :3, 3][:, None], labels['command_ee_world'][:, :, None])
    labels['regression'], labels['regression_valid'] = regression_labels(labels)
    require(all(np.isfinite(a).all() for a in x.values()) and all(np.isfinite(a).all() for a in labels.values()), 'Nonfinite arrays')
    for key in ('past_visual', 'past_text', 'past_state', 'past_ee_pose', 'has_history'):
        require(not np.any(x[key][:572]), 'Invented expert history')
    np.savez_compressed(run / 'inputs.npz', **x); np.savez_compressed(run / 'labels_offline_only.npz', **labels)
    verify(dict(source_sha256=sources)); verify(p)
    sources.update({str(run / f): digest(run / f) for f in ('design.json', 'inputs.npz', 'labels_offline_only.npz')})
    write_json(run / 'protocol.json', dict(status='complete_geometry_cache', rows=rows, expert_indices=list(range(572)),
        recovery_indices=list(range(572, 592)), forward_keys=list(x), source=str(SOURCE), projection=str(SOURCE / 'projection.npz'),
        channel_dim=32, expert_definitions=definitions, recovery_asset_consistency=candidates,
        source_sha256=sources, historical_capture_code_parity=False,
        semantics='World root xyz/orientation; official functional/nominal point xyz; point-set grasp candidates; object-to-destination relation. No articulated-progress labels.',
        orientation_limits='Recovery nominal point orientation unavailable, explicitly masked; root orientation is available. Contact candidates have no teacher-selected grasp ID.',
        no_action_policy_updated=True))
    print('GEOMETRY_CACHE_READY', run, 'rows', len(rows), flush=True)


if __name__ == '__main__':
    threadpool_limits(2); prepare()

"""Prepare a paired, observable-only action/geometry development cache on CPU.

Preserve the exact S2e 611-row training order and action/source weights.  S2h
geometry is auxiliary supervision only; 19 extra action rows have no fabricated
geometry labels.  No image, full fused-prefix, simulator or backbone copy/run.
"""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from .audit import digest
from .paths import ROOT, new_run, write_json
from .probe_candidates import read, verify
from .robot_kinematics import RobotFK
from .structured_features import ARMS, pack_observable


GEOMETRY = ROOT / 'outputs/feature_caches/geometry_supervision_20261003_075610_949206'
PAIRED = ROOT / 'outputs/effect_learning/paired_fixed_20261003_040014_081930'
STRUCTURED = ROOT / 'outputs/feature_caches/structured_readout_20261003_065755_140725'
PREFIX = ROOT / 'outputs/feature_caches/joint_prefix_20260930_141244_699126'
OLD_RECOVERY = ROOT / 'outputs/feature_caches/recovery_20261002_072911_916687'
FORWARD_KEYS = ('visual', 'text', 'state', 'past_visual', 'past_text',
                'past_state', 'has_history', 'ee_pose', 'past_ee_pose')
CAPS = np.array([.1] * 6 + [.2] + [.1] * 6 + [.2], np.float32)


def require(value, message):
    if not value:
        raise ValueError(message)


def row_key(row):
    """Whole-scene identity, and branch identity for multiple recovery forks."""
    kind = row['kind']
    scene = row.get('scene_group', row['group'])
    branch = row.get('branch_group', row['group']) if kind == 'recovery' else ''
    return kind, row['task'], scene, branch, int(row['frame'])


def fk_contract(robot):
    """Only the robot kinematic structure, never object/teacher state."""
    return dict(names=list(robot.names), tip=robot.tip, root=robot.root.tolist(),
                joints={key: [parent, name, kind, origin.tolist(), axis.tolist()]
                        for key, (parent, name, kind, origin, axis) in sorted(robot.joints.items())})


def cap_diagnostic(base, target, valid):
    delta = target - base
    over = (np.abs(delta) > CAPS) & valid[..., None]
    count = int(valid.sum())
    require(count > 0, 'No valid actions')
    clipped = base + np.clip(delta / CAPS, -1, 1) * CAPS
    return dict(valid_steps=count, entirely_supported_steps=int((~over.any(-1) & valid).sum()),
                entirely_supported_fraction=float((~over.any(-1) & valid).sum() / count),
                exceeding_arm_coordinate_fraction=float(over[..., ARMS].sum() / (count * 12)),
                exceeding_gripper_coordinate_fraction=float(over[..., [6, 13]].sum() / (count * 2)),
                max_abs_raw_residual=float(np.max(np.abs(delta[valid]))),
                irreducible_capped_native_rmse=float(np.sqrt(np.mean((clipped[valid] - target[valid]) ** 2))))


def prepare(source=GEOMETRY):
    source = Path(source).resolve()
    run = new_run('structured_action', group=('effect_learning',))
    print('RUN', run, flush=True)
    try:
        return _prepare(source, run)
    except BaseException as error:
        write_json(run / 'failure.json', dict(status='failed_before_training', error_type=type(error).__name__,
                                             error=str(error), traceback=traceback.format_exc()))
        raise


def _prepare(source, run):
    gp = read(source / 'protocol.json'); pp = read(PAIRED / 'protocol.json')
    sp = read(STRUCTURED / 'protocol.json'); rp = read(OLD_RECOVERY / 'protocol.json')
    for directory, names in ((source, ('label_audit.json',)),
                             (PAIRED, ('preflight_audit.json', 'candidate_audit.json')),
                             (STRUCTURED, ('interface_audit.json',)),
                             (OLD_RECOVERY, ('audit.json',))):
        for name in names:
            require(read(directory / name)['passed'], 'Unpassed parent audit: ' + str(directory / name))
    for p in (gp, pp, sp):
        verify(p)
    # The user previously deleted the old L0 directories.  The audited S2e
    # derivatives still seal all six recovery action rows and prefix arrays,
    # while this earlier protocol retains two unavailable provenance references.
    # Record the gap rather than asserting historical replay or silently ignoring
    # any missing data/weights/configuration used by the present preparation.
    missing_provenance = []
    for name, sha in rp['source_sha256'].items():
        path = Path(name)
        if not path.is_file():
            require(path.is_relative_to(ROOT / 'outputs/evaluations') and path.name == 'provenance.json',
                    'Missing consumed recovery dependency: ' + name)
            missing_provenance.append(dict(path=name, original_sha256=sha,
                                           reason='User-deleted original L0 provenance; derived data independently sealed by S2e'))
        else:
            require(digest(path) == sha, 'Changed recovery dependency: ' + name)
    with np.load(source / 'inputs.npz') as z:
        geometry_inputs = {k: z[k].copy() for k in z.files}
    with np.load(source / 'labels_offline_only.npz') as z:
        geometry = z['regression'].copy(); geometry_valid = z['regression_valid'].copy()
    require(set(geometry_inputs) == set(FORWARD_KEYS), 'Unexpected forward field')
    require(len(gp['rows']) == 592 and geometry.shape == (592, 24), 'Unexpected S2h scope')
    with np.load(PAIRED / 'data.npz') as z:
        action = {k: z[k].copy() for k in ('base', 'target', 'valid')}
    with np.load(PAIRED / 'schedule.npz') as z:
        train = z['train_indices'].copy(); weights = z['row_weights'][0].copy()
    require(train.tolist() == pp['train_indices'] and len(train) == 611, 'S2e order changed')
    require(set(train).isdisjoint(pp['calibration_indices']), 'Calibration entered training')
    rows = pp['rows']; index = {row_key(r): i for i, r in enumerate(rows)}
    require(len(index) == len(rows), 'Nonunique S2e identities')
    mapped = [index[row_key(r)] for r in gp['rows']]
    require(len(set(mapped)) == 592 and set(mapped) <= set(train), 'Geometry/action mapping not bijective')
    local = {int(i): j for j, i in enumerate(train)}
    projected = {k: np.zeros((611,) + v.shape[1:], v.dtype) for k, v in geometry_inputs.items()}
    packed = np.zeros(611, bool); gy = np.zeros((611, 24), np.float32); gv = np.zeros((611, 24), bool)
    metadata = [dict(rows[int(i)], original_index=int(i), training_row=j) for j, i in enumerate(train)]
    deps = {**gp['source_sha256'], **pp['source_sha256'], **sp['source_sha256'], **rp['source_sha256']}
    for missing in missing_provenance:
        deps.pop(missing['path'])

    def add_dependency(path, expected=None):
        path = Path(path); actual = digest(path)
        if expected is not None:
            require(actual == expected, 'Changed direct source: ' + str(path))
        deps[str(path)] = actual
        return path

    for directory, names in ((source, ('protocol.json', 'inputs.npz', 'labels_offline_only.npz', 'label_audit.json')),
                             (PAIRED, ('protocol.json', 'data.npz', 'schedule.npz')),
                             (STRUCTURED, ('protocol.json', 'inputs.npz', 'interface_audit.json')),
                             (OLD_RECOVERY, ('protocol.json', 'data.npz', 'features.npz', 'audit.json'))):
        for name in names:
            add_dependency(directory / name)
    projection = add_dependency(Path(gp['projection']))
    with np.load(projection) as z:
        pre = {k: z[k].copy() for k in z.files}
    config_path = add_dependency(Path(pp['config'])); config = read(config_path)
    robots = [RobotFK(config, side) for side in ('left', 'right')]
    contracts = [fk_contract(b) for b in robots]
    for bot in robots:
        add_dependency(bot.urdf)
    checked_configs = []

    def check_config(path):
        path = Path(path)
        if any(entry['path'] == str(path) for entry in checked_configs):
            return
        add_dependency(path)
        candidate = [RobotFK(read(path), side) for side in ('left', 'right')]
        require([fk_contract(b) for b in candidate] == contracts, 'Robot FK contract mismatch: ' + str(path))
        for bot in candidate:
            add_dependency(bot.urdf)
        checked_configs.append(dict(path=str(path), robot_fk_structure_exact=True))

    def assign(original, block, n, meta):
        j = local[original]
        require(not packed[j] and set(block) == set(FORWARD_KEYS), 'Duplicate or privileged packing')
        for key in FORWARD_KEYS:
            projected[key][j] = block[key][n]
        packed[j] = True
        metadata[j].update(meta, fk_config=str(config_path))

    for gi, original in enumerate(mapped):
        r = gp['rows'][gi]
        if r['kind'] == 'recovery':
            require(rows[original]['source'] == r['source'] and rows[original]['source_point'] == r['source_point'],
                    'Recovery branch/source disagreement')
            query = dict(kind='npz', path=r['observation'], native_count=r['native_count'])
            instruction = r['instruction']
            manifest = read(add_dependency(Path(r['source']) / 'manifest.json'))
            cfg = Path(manifest['source']) / 'resolved_config.json'
            check_config(cfg)
        else:
            query = dict(kind='expert_hdf5', path=r['source'], frame=r['frame'], native_count=r['frame'])
            instruction = r['instruction']
        assign(original, geometry_inputs, gi, dict(geometry_row=gi, instruction=instruction,
               source_query=query, geometry_available=True, history_origin='frozen_S2h'))
        gy[local[original]] = geometry[gi]; gv[local[original]] = geometry_valid[gi]

    prefix_result = read(add_dependency(PREFIX / 'result.json'))
    records = prefix_result['records']
    expert_groups = {r['group'] for r in gp['rows'] if r['kind'] == 'expert'}
    for record in records:
        s = record['source']
        if s['kind'] == 'expert' and s['group'] in expert_groups:
            check_config(Path(s['collection']) / 'resolved_config.json')
    with np.load(STRUCTURED / 'inputs.npz') as z:
        shoe_inputs = {k: z[k].copy() for k in z.files}
    for i, r in enumerate(sp['rows']):
        if r['task'] != 'place_shoe':
            continue
        k = ('recovery', r['task'], r['scene_group'], r['group'], int(r['frame']))
        original = index[k]
        manifest = read(add_dependency(Path(r['source']) / 'manifest.json'))
        check_config(Path(manifest['source']) / 'resolved_config.json')
        assign(original, shoe_inputs, i, dict(geometry_row=None, instruction=r['instruction'],
               source_query=dict(kind='npz', path=r['observation'], native_count=r['native_count']),
               geometry_available=False, history_origin='frozen_S2g_exact32'))

    with np.load(OLD_RECOVERY / 'features.npz') as z:
        joint = z['joint'].copy(); state = z['state'].copy()
    with np.load(OLD_RECOVERY / 'data.npz') as z:
        old_actions = {k: z[k].copy() for k in action}
    old_history = np.array(rp['history_indices'], np.int64)
    previous = np.maximum(old_history, 0)
    for i, h in enumerate(old_history):
        if h >= 0:
            current, past = rp['rows'][i], rp['rows'][h]
            require(h < i and current['source'] == past['source'] and current['group'] == past['group']
                    and current['instruction'] == past['instruction']
                    and current['native_count'] - past['native_count'] == 32, 'Illegal old recovery history')
    block = pack_observable(joint, state, joint[previous], state[previous], old_history >= 0, pre, robots)
    for i, r in enumerate(rp['rows']):
        k = ('recovery', r['task'], r['group'], r['group'], int(r['native_count']))
        original = index[k]
        require(all(np.array_equal(action[key][original], old_actions[key][i]) for key in action),
                'Old recovery label mismatch')
        assign(original, block, i, dict(geometry_row=None, instruction=r['instruction'],
               source_query=dict(kind='npz', path=r['observation'], native_count=r['native_count']),
               geometry_available=False, history_origin='same_teacher_source_exact32' if old_history[i] >= 0 else 'absent'))

    def pack_anchor(original):
        row = rows[original]
        found = [r for r in records if r['source']['kind'] == 'anchor'
                 and r['source']['group'] == row['group'] and r['source']['chunk'] == row['frame']]
        require(len(found) == 1, 'Anchor prefix not unique')
        record = found[0]; s = record['source']
        add_dependency(record['file'], record['sha256']); checkpoint = add_dependency(s['checkpoint'], s['checkpoint_sha256'])
        manifest = read(add_dependency(Path(s['run']) / 'manifest.json'))
        add_dependency(Path(s['run']) / 'result.json', s['result_sha256'])
        with np.load(record['file']) as z:
            j = z['joint'].copy()
            require(z['frames'].tolist() == [row['frame']], 'Anchor feature query mismatch')
        with np.load(checkpoint) as z:
            st = z['command_state'][None].copy()
            require(np.array_equal(z['base_actions'], action['base'][original]), 'Anchor base mismatch')
        require(manifest['instruction'] == s['instruction'] and manifest['chunk'] == row['frame'], 'Anchor source identity')
        require(np.array_equal(action['base'][original], action['target'][original]), 'Anchor not identity target')
        # The deleted L0 trace is unavailable.  A chunk number alone does not
        # establish its exact native action count, so no anchor past is invented.
        x = pack_observable(j, st, np.zeros_like(j), np.zeros_like(st), np.array([False]), pre, robots)
        meta = dict(geometry_row=None, instruction=s['instruction'], geometry_available=False,
                    source_query=dict(kind='npz', path=str(checkpoint), native_count=None, chunk=row['frame']),
                    history_origin='absent_missing_exact_native_count', prefix_source=record['file'],
                    source_scene=s['source'], historical_raw_trace_available=Path(s['source']).is_dir(),
                    fk_config=str(config_path))
        return x, meta

    for original in train:
        if rows[int(original)]['kind'] == 'anchor':
            block, meta = pack_anchor(int(original)); assign(int(original), block, 0, meta)
    require(packed.all(), 'Missing action input rows')
    counts = Counter(r['kind'] for r in metadata)
    require(counts == {'expert': 572, 'anchor': 10, 'recovery': 29}, 'Unexpected training components')
    require(len(expert_groups) == 32, 'Expert training scene scope changed')
    for r in metadata:
        scene = r.get('scene_group', r['group'])
        seed = int(scene.split(':')[1])
        require(not 100380 <= seed <= 100399, 'Development evaluation seed entered training')
    require(sum(r['geometry_available'] for r in metadata) == 592 and not gv[~np.array([r['geometry_available'] for r in metadata])].any(),
            'Fabricated auxiliary labels')
    for key, array in projected.items():
        require(len(array) == 611 and np.isfinite(array).all(), 'Invalid observable: ' + key)
    ww = weights[train].copy()
    require(np.all(ww > 0), 'Unexpected include_all zero weight')
    denominator = np.sum(ww[:, None] * gv, axis=0, dtype=np.float64)
    require(np.all(denominator > 0), 'Unsupported geometry prior coordinate')
    aux_mean = (np.sum(ww[:, None] * gv * gy, axis=0, dtype=np.float64) / denominator).astype(np.float32)
    labels = dict(base=action['base'][train], target=action['target'][train], valid=action['valid'][train],
                  geometry=gy, geometry_valid=gv, row_weights=ww, aux_mean=aux_mean)

    development = [int(i) for i in pp['calibration_indices'] if rows[int(i)]['kind'] == 'anchor']
    require(len(development) == 4 and {rows[i]['group'] for i in development} == {'place_empty_cup:320002'},
            'Unexpected development anchor scope')
    dx = {k: [] for k in FORWARD_KEYS}; dm = []
    for original in development:
        block, meta = pack_anchor(original)
        for key in FORWARD_KEYS:
            dx[key].append(block[key][0])
        dm.append(dict(rows[original], original_index=original, diagnostic_only=True, **meta))
    dx = {k: np.stack(v) for k, v in dx.items()}
    dl = dict(base=action['base'][development], target=action['target'][development], valid=action['valid'][development],
              geometry=np.zeros((4, 24), np.float32), geometry_valid=np.zeros((4, 24), bool))
    np.savez_compressed(run / 'inputs.npz', **projected)
    np.savez_compressed(run / 'labels.npz', **labels)
    np.savez_compressed(run / 'development_inputs.npz', **dx)
    np.savez_compressed(run / 'development_labels.npz', **dl)
    snapshot = run / 'source_snapshot'; snapshot.mkdir()
    shutil.copyfile(Path(__file__), snapshot / Path(__file__).name)
    for path in (Path(__file__), snapshot / Path(__file__).name, ROOT / 'cx_tta/structured_features.py',
                 ROOT / 'cx_tta/robot_kinematics.py'):
        add_dependency(path)
    for name in ('inputs.npz', 'labels.npz', 'development_inputs.npz', 'development_labels.npz'):
        add_dependency(run / name)
    diagnostics = {kind: cap_diagnostic(labels['base'][ids], labels['target'][ids], labels['valid'][ids])
                   for kind in counts for ids in [[i for i, r in enumerate(metadata) if r['kind'] == kind]]}
    protocol = dict(status='complete_before_model_initialization', rows=metadata, development_rows=dm,
        source_geometry=str(source), source_paired=str(PAIRED), source_structured=str(STRUCTURED), source_old_recovery=str(OLD_RECOVERY),
        original_train_indices=train.tolist(), original_calibration_indices=pp['calibration_indices'],
        geometry_to_original_indices=mapped, geometry_to_training_indices=[local[i] for i in mapped],
        config=str(config_path), projection=str(projection), channel_dim=32, forward_keys=list(FORWARD_KEYS),
        label_keys=list(labels), training_rows=611, development_rows_count=4,
        component_counts=dict(counts), expert_scenes=32, source_weights_condition='S2e_include_all_row_weights[0]',
        component_weight_sums={kind: float(ww[[i for i, r in enumerate(metadata) if r['kind'] == kind]].sum()) for kind in counts},
        preprocessing='Exact S2g expert572/32-scene PCA32 bytes; no fit/refit. Existing S2h592 and S2g shoe inputs copied numerically unchanged.',
        auxiliary_prior='Columnwise original action-row-weighted valid geometry mean from the592 training rows only; no development labels.',
        original_caps=CAPS.tolist(), original_cap_diagnostic=diagnostics,
        supervision='Raw native target retained; trainer uses clip((target-base)/original_caps,-1,1). No raw teacher success or geometry-readout metric substitutes for capped policy success.',
        historical_noise='Expert base_cache_20260926 uses explicit NumPy32x32 Gaussian noise; old/new recovery use official compiled internal noise. Training distribution differs; live paired evaluation must use official internal noise.',
        history='Existing S2h experts have no past; new recovery has genuine32-native-action past. Old recovery past checked same source/group/instruction/native_count difference32. Anchor chunk IDs cannot verify native counts after raw L0 deletion, so all anchors explicitly have no past.',
        robot_fk_contracts=contracts, robot_config_checks=checked_configs,
        missing_inherited_provenance=missing_provenance, historical_raw_L0_replay_verified=False,
        inference='Exactly9 observable fields plus current VLA base chunk. Geometry/validity/task/scene/seed/asset/teacher phase are supervision or metadata only.',
        scope='Development action-candidate cache; 100380..100399 excluded from every training source. Four cup anchor rows are diagnostic only. No GPU, backbone sampling, simulator, new collection or policy success claim.',
        source_sha256=deps)
    verify(dict(source_sha256=deps))
    write_json(run / 'protocol.json', protocol)
    write_json(run / 'preparation_audit.json', dict(passed=True, training_rows=611, development_rows=4,
        training_order_exact=True, original_action_labels_exact=True, original_row_weights_exact=True,
        geometry_bijection_rows=592, geometry_input_copy_exact=True, unlabelled_geometry_rows=19,
        development_not_training=True, evaluation_seeds_not_training=True, robot_fk_contract_exact=True,
        forward_whitelist=True, no_raw_data_copies=True, gpu_used=False,
        source_sha256={str(run / 'protocol.json'): digest(run / 'protocol.json')}))
    print('STRUCTURED_ACTION_DATA_READY', run, 'training611 development4 geometry592', flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, default=GEOMETRY)
    args = parser.parse_args()
    threadpool_limits(2)
    prepare(args.source)


if __name__ == '__main__':
    main()

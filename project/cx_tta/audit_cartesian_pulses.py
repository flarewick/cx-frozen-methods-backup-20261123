"""Validate schema-4 short-window observations/actions before building training data."""
import argparse
import json
from pathlib import Path

import numpy as np

from .short_window_contract import make_windows, visibility_labels, validate_calibration
from .audit import digest
from .deploy import CAMERA_KEYS
from .eval_diagnostics import array_hash
from .intervention_contract import validate_transition
from .cartesian_pulse import BRANCHES, candidates, pulse_windows
from .robot_kinematics import RobotFK
from .paths import output_path
from .probe_action_replay import compare_physical


def audit_run(run):
    run = output_path(Path(run))
    manifest = json.loads((run / 'manifest.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    if ((run / 'failure.json').exists() or result['status'] != 'complete'
            or result.get('schema_version') != 4 or manifest.get('schema_version') != 4
            or not result['valid_same_state_intervention']
            or set(result['branches']) != set(BRANCHES)):
        raise ValueError('Incomplete or unsupported intervention run')
    source = Path(manifest['source'])
    seed, chunk = manifest['source_seed'], manifest['chunk']
    trace = source / 'traces' / f'r0_seed{seed}.jsonl'
    if (digest(trace) != manifest['source_trace_sha256']
            or digest(source / 'manifest.json') != manifest['source_manifest_sha256']):
        raise ValueError('Source provenance changed')
    if digest(source / 'resolved_config.json') != manifest['source_config_sha256']:
        raise ValueError('Source configuration changed')
    reference = [json.loads(line) for line in trace.read_text().splitlines()][chunk]
    with np.load(run / 'checkpoint.npz', allow_pickle=False) as data:
        checkpoint = dict(data)
    base = checkpoint['base_actions']
    if not np.array_equal(base, np.asarray(reference['base_executed'], np.float32)):
        raise ValueError('Saved base actions differ from source')
    specification = json.loads((run / 'proposal_spec.json').read_text())
    config = json.loads((source / 'resolved_config.json').read_text())
    robots = [RobotFK(config,side) for side in ('left','right')]
    proposals, expected_spec, cartesian = candidates(base, robots, manifest['pulse_start'])
    if expected_spec != specification:
        raise ValueError('Cartesian specification mismatch')
    summaries = {}
    stride = manifest['record_stride']
    if result['record_stride'] != stride:
        raise ValueError('Stride mismatch')
    prefix = [a for row in [json.loads(line) for line in trace.read_text().splitlines()][:chunk]
              for a in row['base_executed']]
    history = np.asarray(prefix[-stride:], np.float32).reshape(-1, 14)
    initial_masks = None
    first = result['branches']['base']
    for name in BRANCHES:
        branch = result['branches'][name]
        if (not compare_physical(reference['before'], branch['before'], manifest['tolerance'])['matched']
                or not compare_physical(first['before'], branch['before'], manifest['tolerance'])['matched']
                or branch['observation_signature'] != first['observation_signature']):
            raise ValueError('Branch start mismatch: ' + name)
        with np.load(run / f'transition_{name}.npz', allow_pickle=False) as data:
            arrays = dict(data)
        if not np.array_equal(arrays['planned_cartesian_residual'],cartesian[name]):
            raise ValueError('Planned Cartesian residual mismatch')
        summary = validate_transition(arrays)
        if summary != branch['transition'] or summary['executed'] != branch['executed']:
            raise ValueError('Transition summary mismatch: ' + name)
        if not np.allclose(arrays['proposed_actions'], proposals[name], atol=1e-7, rtol=0):
            raise ValueError('Proposal mismatch: ' + name)
        if not np.array_equal(arrays['residual_actions'], arrays['proposed_actions'] - base):
            raise ValueError('Residual mismatch: ' + name)
        if not np.array_equal(arrays['proposed_actions'][arrays['executed_mask']],
                              np.asarray(branch['executed_actions'], np.float32).reshape(-1, 14)):
            raise ValueError('Executed actions mismatch: ' + name)
        for short, camera in zip(('head', 'right', 'left'), CAMERA_KEYS):
            if (array_hash(arrays[short + '_rgb_after'])
                    != branch['after_observation_signature']['images'][camera]):
                raise ValueError('Endpoint image mismatch: ' + name)
        if not np.array_equal(arrays['history_actions'], history):
            raise ValueError('History actions mismatch')
        frames = branch['frames']
        windows = pulse_windows(frames, arrays['executed_mask'], arrays['physics_dt_s'], arrays['physics_steps'], manifest['pulse_start'])
        if windows != branch['windows']:
            raise ValueError('Window metadata mismatch')
        if ([f['offset'] for f in frames if f['offset'] < 0] != ([-stride] if chunk else [])
                or branch['history_observation_available'] != bool(chunk)):
            raise ValueError('Missing prefix history')
        for frame in frames:
            contact_label=frame['contacts_offline_only']
            if not contact_label['scope'].startswith('instantaneous snapshot'):
                raise ValueError('Invalid contact scope')
            for pair in contact_label['pairs']:
                if len(pair['entity_ids'])!=2 or len(pair['entity_names'])!=2:
                    raise ValueError('Invalid contact pair')
                for position in pair['positions_world_m']:
                    if np.asarray(position).shape!=(3,) or not np.isfinite(position).all():
                        raise ValueError('Invalid contact position')
            if 0 <= frame['offset'] <= manifest['pulse_start']:
                other = next(f for f in first['frames'] if f['offset']==frame['offset'])
                if (frame['observation_signature'] != other['observation_signature'] or
                    not compare_physical(other['physical_offline_only'],frame['physical_offline_only'],manifest['tolerance'])['matched'] or
                    frame['visibility_offline_only'] != other['visibility_offline_only']):
                    raise ValueError('Common pre-pulse state mismatch')
            observed_path, offline_path = run / frame['observed_file'], run / frame['offline_file']
            if (observed_path.parent != run or offline_path.parent != run
                    or digest(observed_path) != frame['observed_sha256']
                    or digest(offline_path) != frame['offline_sha256']):
                raise ValueError('Frame file hash/path mismatch')
            with np.load(observed_path, allow_pickle=False) as data:
                observed = dict(data)
            with np.load(offline_path, allow_pickle=False) as data:
                labels = visibility_labels(data['actor_ids'], branch['entity_id_map_offline'])
            if set(labels) != set(frame['physical_offline_only']['offline_only']['object_poses']):
                raise ValueError('Entity mapping and physical label keys differ')
            if labels != frame['visibility_offline_only']:
                raise ValueError('Visibility labels mismatch')
            signature = {'images': {camera: array_hash(observed[short + '_rgb'])
                         for short, camera in zip(('head', 'right', 'left'), CAMERA_KEYS)},
                         'command_state': array_hash(observed['command_state_raw'])}
            if signature != frame['observation_signature']:
                raise ValueError('Frame observation mismatch')
            if not np.array_equal(observed['command_state'], observed['command_state_raw'].astype(np.float32)):
                raise ValueError('Command conversion mismatch')
            for key in ('arm_qpos', 'gripper_qpos'):
                if not np.array_equal(observed[key], frame['physical_offline_only'][key]):
                    raise ValueError('Measured state mismatch')
            for i in range(3):
                validate_calibration(observed['intrinsic_cv'][i], observed['extrinsic_cv'][i], observed['cam2world_gl'][i])
            if frame['offset'] in (0, 32):
                endpoint = 'before' if frame['offset'] == 0 else 'after'
                sig_key = 'observation_signature' if frame['offset'] == 0 else 'after_observation_signature'
                if (signature != branch[sig_key] or not compare_physical(
                        branch[endpoint], frame['physical_offline_only'], manifest['tolerance'])['matched']):
                    raise ValueError('Frame endpoint mismatch')
            if frame['offset'] == 0:
                masks = {k: v['semantic_mask_sha256'] for k, v in labels.items()}
                if initial_masks is None:
                    initial_masks = masks
                elif masks != initial_masks:
                    raise ValueError('Initial branch semantic masks differ')
        summaries[name] = dict(summary, windows=len(windows),
                              positive_time_windows=sum(w['positive_time'] for w in windows))
    if (first['executed'] != 32
            or not compare_physical(reference['after'], first['after'], manifest['tolerance'])['matched']):
        raise ValueError('Base endpoint not reproduced')
    return {'schema_version': 4, 'status': 'passed', 'run': str(run), 'branches': summaries,
            'source_observation_exact_match': result['source_observation_exact_match'],
            'scope': 'data contract only; not evidence of recovery or M1 training readiness'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run), indent=2))


if __name__ == '__main__':
    main()

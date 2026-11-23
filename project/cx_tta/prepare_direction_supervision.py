"""Reuse same-state expert actions as directional auxiliary labels, NOT recovery goals."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import h5py
import numpy as np
from .audit import digest
from .dataset import load_cache, expert_chunk
from .effect_contract import timeline
from .paths import ROOT, new_run, write_json
from .robot_kinematics import RobotFK, cartesian_residual

OFFSETS = np.array([0, 3, 7, 15, 31], dtype=int)


def direction_targets(base, expert, valid, robots):
    """FK of recorded *commands*: no claim of realized motion or optimal object goal."""
    base, expert, valid = np.asarray(base), np.asarray(expert), np.asarray(valid)
    if base.shape != expert.shape or base.shape != (32, 14) or valid.shape != (32,) or valid.dtype != bool:
        raise ValueError('Expected one aligned 32-action example')
    safe = expert[OFFSETS].copy()
    mask = valid[OFFSETS]
    safe[~mask] = base[OFFSETS][~mask]
    target = cartesian_residual(base[OFFSETS], safe, robots).astype(np.float32)
    target[~mask] = 0
    return target, mask


def audit_preferences():
    reports, rows = [], []
    for family in ('intervention_batches', 'short_window_batches', 'cartesian_pulse_batches'):
        count = Counter()
        scenes = set()
        for path in sorted((ROOT/'outputs'/family).glob('*/*/index.json')):
            index = json.loads(path.read_text())
            if index['status'] != 'complete':
                raise ValueError('Incomplete intervention batch')
            for row in index['records']:
                if row['split'] != 'train':
                    continue  # Never open original held-out branch outcome labels.
                run = Path(row['run'])
                result = json.loads((run/'result.json').read_text())
                manifest = json.loads((run/'manifest.json').read_text())
                if (manifest['task'], manifest['source_seed'], manifest['chunk']) != (row['task'], row['seed'], row['chunk']):
                    raise ValueError('Intervention identity mismatch')
                actual_hash = digest(run/'result.json')
                if 'result_sha256' in row and actual_hash != row['result_sha256']:
                    raise ValueError('Intervention changed')
                outcomes = {k:bool(v['success_after']) for k,v in result['branches'].items()}
                mixed = len(set(outcomes.values())) > 1
                count['points'] += 1
                count['branches'] += len(outcomes)
                count['successful_endpoints'] += sum(outcomes.values())
                count['points_with_success_preference'] += mixed
                scenes.add((row['task'], row['seed']))
                rows.append(dict(family=family, task=row['task'], seed=row['seed'], chunk=row['chunk'], run=str(run), result_sha256=actual_hash, endpoint_success=outcomes, has_rankable_success_preference=mixed))
        reports.append(dict(family=family, independent_scenes=len(scenes), **dict(count)))
    return dict(summary=reports, records=rows, scope='Original-train checkpoint branches only. success_after=False means no success at this truncated endpoint, NOT a terminal task failure. Equal success flags provide no candidate ranking, and object displacement magnitude is not used as a substitute.')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--cache', type=Path, default=ROOT/'outputs/base_cache_20260926_024808_872087')
    args = p.parse_args()
    manifest = json.loads((args.cache/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['limited_probe']:
        raise ValueError('Need audited complete cache')
    output = new_run('expert_direction_supervision', group=('effect_learning',))
    print(output, flush=True)
    (output/'labels').mkdir()
    seen, records, configs, attempts = set(), [], {}, {}
    stats = defaultdict(Counter)
    for row in manifest['files']:
        key = row['task'], row['episode_seed']
        if key in seen:
            raise ValueError('Duplicate episode across splits')
        seen.add(key)
        # Inspect provenance for validation, but do not construct/read its targets this round.
        if row['split'] != 'train':
            records.append(dict(task=key[0], seed=key[1], split=row['split'], status='reserved_not_loaded'))
            continue
        source = Path(row['source'])
        if digest(source) != row['source_sha256']:
            raise ValueError('Changed source HDF5')
        cache = load_cache(Path(row['path']))
        if str(cache['source_sha256']) != row['source_sha256']:
            raise ValueError('Changed cache lineage')
        collection = Path(row['collection'])
        if collection not in configs:
            cp = collection/'resolved_config.json'
            config = json.loads(cp.read_text())
            configs[collection] = [RobotFK(config, side) for side in ('left', 'right')]
            attempts[collection] = {r['seed']:r for r in json.loads((collection/'attempts.json').read_text())}
        attempt = attempts[collection][key[1]]
        if not attempt['replay_success'] or not attempt['format_valid'] or attempt['split'] != 'train' or Path(attempt['hdf5']).resolve() != source.resolve():
            raise ValueError('Unverified successful demonstration')
        robot = configs[collection]
        directions, masks, future_dt, zero_dt = [], [], [], 0
        with h5py.File(source, 'r') as h:
            if str(h.attrs['instruction']) != row['instruction'] or (str(h.attrs['task']), int(h.attrs['seed'])) != key:
                raise ValueError('Instruction/episode mismatch')
            commands = np.asarray(h['joint_action/vector'], np.float32)
            times = timeline(h['cx_measurement/physics_tick'][:], h['cx_dense/tick_before'][:], h['cx_dense/dt_s'][:])
            for n, frame in enumerate(cache['frame_index']):
                expert, valid = expert_chunk(commands, int(frame))
                if not np.array_equal(expert, cache['target'][n]) or not np.array_equal(valid, cache['valid'][n]):
                    raise ValueError('Expert label not from same observation')
                expected_state = commands[frame].copy()
                expected_state[[0,1,2,3,4,5,7,8,9,10,11,12]] /= np.pi
                if not np.array_equal(cache['context'][n,:14], expected_state):
                    raise ValueError('Current command input mismatch')
                direction, mask = direction_targets(cache['base_action'][n], expert, valid, robot)
                elapsed = np.zeros(len(OFFSETS), np.float64)
                elapsed[mask] = times['time_s'][frame+1+OFFSETS[mask]] - times['time_s'][frame]
                zero_dt += int(((elapsed == 0) & mask).sum())
                directions.append(direction); masks.append(mask); future_dt.append(elapsed)
            current_arm = h['cx_measurement/arm_qpos'][:][cache['frame_index']]
            current_gripper = h['cx_measurement/gripper_joint_qpos'][:][cache['frame_index']]
            has_articulation = 'cx_effect/articulation' in h
            objects = list(h['cx_privileged'])
        label = output/'labels'/f'{key[0]}_{key[1]}.npz'
        np.savez_compressed(label, expert_plan_direction=np.asarray(directions), direction_valid=np.asarray(masks), expert_elapsed_s_offline_only=np.asarray(future_dt), source_frame_index=cache['frame_index'], current_arm_qpos=current_arm, current_gripper_qpos=current_gripper)
        s = stats[key[0]]
        s['episodes'] += 1; s['queries'] += len(directions); s['valid_direction_slots'] += int(np.sum(masks)); s['zero_elapsed_direction_slots'] += zero_dt; s['episodes_with_articulation'] += has_articulation
        records.append(dict(task=key[0], seed=key[1], group=f'{key[0]}:{key[1]}', split='train', status='prepared', source=str(source), source_sha256=row['source_sha256'], cache=str(Path(row['path']).resolve()), cache_sha256=digest(Path(row['path'])), collection=str(collection), instruction=row['instruction'], samples=len(directions), label=str(label), label_sha256=digest(label), config_sha256=digest(collection/'resolved_config.json'), urdf_sha256=digest(robot[0].urdf), offline_entities=objects))
        if len([r for r in records if r['status']=='prepared']) % 25 == 0:
            print('PREPARED', sum(r['status']=='prepared' for r in records), flush=True)
    preferences = audit_preferences()
    write_json(output/'preference_audit.json', preferences)
    expert_groups = {(r['task'],r['seed']) for r in records if r['status']=='prepared'}
    intervention_groups = {(r['task'],r['seed']) for r in preferences['records']}
    write_json(output/'index.json', dict(status='complete', records=records, summary={k:dict(v) for k,v in stats.items()}, offsets=(OFFSETS+1).tolist(), direction_units_per_arm=['translation/.01m']*3+['rotation/.1rad']*3+['gripper_command/.1'], cache_manifest=str(args.cache/'manifest.json'), cache_manifest_sha256=digest(args.cache/'manifest.json'), code_sha256=digest(Path(__file__)), expert_intervention_scene_overlap=len(expert_groups & intervention_groups), schema='expert_command_direction_v1', scope='Original 200 training demonstrations only, 40 validation episodes reserved. Targets derived from same-observation expert commands and robot FK; these are planned command corrections, NOT measured object-motion goals or recovery targets. Raw instruction/current images/current proprio/base plan are legal inputs. Future expert action, direction and elapsed time are targets/metadata ONLY. Existing compiled cache used explicit noise, unlike current official internal noise: requires matched deployment control before efficacy claims.'))
    print('COMPLETE '+str(output), flush=True)


if __name__ == '__main__':
    main()

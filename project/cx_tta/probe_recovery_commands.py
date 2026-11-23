"""Audit actual recovery command domains and continuous trace boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import ROOT, new_run, write_json
from .recovery_fit_metrics import ARMS, metrics, recovery_gate
from .robot_kinematics import RobotFK
from .audit_local_geometry import joint_bounds

SOURCE = ROOT / 'outputs/effect_learning/recovery_fit_20261002_075516_235990'
RAW = ROOT / 'outputs/feature_caches/recovery_20261002_072911_916687'
ROBOTWIN = Path('/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin')


def read(path):
    return json.loads(path.read_text())


def audit(source):
    p, audit = read(source/'protocol.json'), read(source/'audit.json')
    if not audit['passed'] or digest(source/'data.npz') != p['data_sha256']:
        raise ValueError('Audited unchanged S1s data required')
    for file in (source/'result.json', source/'protocol.json'):
        if digest(file) != audit['source_sha256'][str(file)]:
            raise ValueError('Changed S1s report')
    with np.load(source/'data.npz') as z:
        data = {k: z[k].copy() for k in z.files}
    with np.load(RAW/'data.npz') as z:
        raw = {k: z[k].copy() for k in z.files}
    rp = read(RAW/'protocol.json')
    ids = p['parts']['recovery']
    robots = [RobotFK(read(Path(p['config'])), side) for side in ('left', 'right')]
    for k in ('base', 'target', 'valid'):
        if not np.array_equal(data[k][ids], raw[k]):
            raise ValueError('Actual recovery source mismatch')
    pairs, bounds, domains = [], [], {}
    for i, row in enumerate(rp['rows']):
        mask = raw['valid'][i]; count = int(mask.sum())
        if not np.array_equal(mask, np.arange(32) < count):
            raise ValueError('Non-prefix valid mask')
        counts = raw['source_native_counts'][i, mask]
        if not np.array_equal(counts, np.arange(row['native_count']+1, row['native_count']+count+1)):
            raise ValueError('Noncontinuous source native count')
        if digest(Path(row['target'])) != row['target_sha256']:
            raise ValueError('Changed original executed target')
        for arm, start in enumerate((0, 7)):
            q = raw['target'][i, mask, start:start+6]; limits = joint_bounds(robots[arm])
            if ((q < limits[:, 0]) | (q > limits[:, 1])).any():
                raise ValueError('Teacher outside native URDF bounds')
        if i and rp['rows'][i-1]['group'] == row['group']:
            previous = rp['rows'][i-1]; n = int(raw['valid'][i-1].sum())
            if previous['source'] != row['source'] or previous['role'] != row['role']:
                raise ValueError('Cross-source seam')
            if raw['source_native_counts'][i-1, n-1] + 1 != counts[0]:
                raise ValueError('Unconnected block labels')
            jump = raw['target'][i, 0] - raw['target'][i-1, n-1]
            pairs.append(dict(before=ids[i-1], last=n-1, after=ids[i], group=row['group'],
                task=row['task'], source=row['source'], before_native=int(counts[0]-1), after_native=int(counts[0]),
                same_plan=bool(raw['plan_ids'][i-1, n-1] == raw['plan_ids'][i, 0]),
                teacher_jump=jump.tolist(), teacher_arm_rms_rad=float(np.sqrt(np.mean(jump[ARMS]**2))),
                teacher_arm_max_abs_rad=float(np.abs(jump[ARMS]).max())))
        bounds.append(dict(group=row['group'], frame=row['native_count'], valid_actions=count,
            within_arm_max_delta_rad=float(np.abs(np.diff(raw['target'][i, :count][:, ARMS], axis=0)).max())))
    if len(pairs) != 4:
        raise ValueError('Expected four audited recovery seams')
    for part, ix in p['parts'].items():
        domains[part] = {}
        for name in ('base', 'target'):
            g = data[name][ix][..., [6, 13]][data['valid'][ix]]
            domains[part][name] = dict(min=float(g.min()), max=float(g.max()), count=int(g.size),
                outside_unit=int(((g<0)|(g>1)).sum()), outside_margin=int(((g<-.02)|(g>1.02)).sum()))
    if domains['recovery']['target']['outside_unit']:
        raise ValueError('Teacher gripper label contract mismatch')
    predictions = []
    files = [source/'protocol.json', source/'audit.json', source/'result.json', source/'data.npz',
             RAW/'protocol.json', RAW/'data.npz', Path(__file__),
             ROOT/'cx_tta/recovery_temporal.py', ROBOTWIN/'envs/_base_task.py', ROBOTWIN/'envs/robot/robot.py']
    for seed in p['seeds']:
        file = source/f'semantic_mixed_se3_s{seed}_u1400_recovery.npz'; files.append(file)
        with np.load(file) as z:
            pred = z['predicted'].copy()
        summary, _ = metrics(pred, raw['base'], raw['target'], raw['valid'], [p['rows'][i] for i in ids], robots)
        clipped = pred.copy(); clipped[..., [6, 13]] = np.clip(clipped[..., [6, 13]], 0, 1)
        cm, _ = metrics(clipped, raw['base'], raw['target'], raw['valid'], [p['rows'][i] for i in ids], robots)
        if not np.array_equal(pred[..., ARMS], clipped[..., ARMS]):
            raise ValueError('Unexpected arm mutation')
        seam_records = []
        for pair in pairs:
            a, b = ids.index(pair['before']), ids.index(pair['after']); n = pair['last']
            dj = pred[b, 0] - pred[a, n]; teacher = np.array(pair['teacher_jump'])
            seam_records.append(dict(group=pair['group'], before_native=pair['before_native'],
                predicted_jump=dj.tolist(), arm_error_rms_rad=float(np.sqrt(np.mean((dj[ARMS]-teacher[ARMS])**2)))))
        predictions.append(dict(seed=seed, raw_summary=summary, posthoc_clip_summary=cm,
            raw_fit_pass=recovery_gate(summary), posthoc_clip_fit_pass=recovery_gate(cm), seams=seam_records))
    result = dict(status='complete', passed=True, source=str(source), pairs=pairs, teacher_blocks=bounds,
        gripper_domains=domains, baseline_predictions=predictions,
        robot_gripper_contract='take_action interpolates RAW gripper targets first; Robot.set_gripper clips to [0,1] afterward. '
            'Pre-clipping before interpolation need not preserve executed motion. No runtime patch or execution claim.',
        next_decoder='Envelope [min(base,0),max(base,1)] prevents additional overshoot, preserves original raw base '
            'at zero/disabled correction. It is not strict [0,1] when original base is outside.',
        next_time_loss='Supervise real valid command increments, not zero velocity. Four seams from the same audited '
            'continuous source/role, no cross-group/future forward inputs. Plan IDs only provenance diagnostics.',
        source_sha256={str(f): digest(f) for f in files},
        scope='Offline interface diagnosis on two studied recovery scenes. No simulation or task success evidence.')
    run = new_run('recovery_commands', group=('reports', 'audits'))
    write_json(run/'result.json', result)
    print(run, flush=True)
    print(json.dumps(dict(passed=True, pairs=len(pairs), teacher_blocks=len(bounds),
        clip_only_fit_pass=[r['posthoc_clip_fit_pass'] for r in predictions],
        teacher_max_seam_rad=max(p['teacher_arm_max_abs_rad'] for p in pairs))))


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--source', type=Path, default=SOURCE)
    audit(parser.parse_args().source.resolve())


if __name__ == '__main__':
    main()

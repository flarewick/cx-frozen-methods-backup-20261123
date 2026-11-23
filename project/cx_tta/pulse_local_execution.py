"""Existing-data action realization and state geometry; all measurements offline."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import new_run, write_json
from .robot_kinematics import RobotFK
from .pulse_reliability_diagnostics import MAIN


def distribution(values):
    values = np.asarray(values)
    return dict(n=len(values), min=float(values.min()), median=float(np.median(values)), max=float(values.max())) if len(values) else None


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    args = p.parse_args()
    index = json.loads((args.dataset / 'index.json').read_text())
    rows = []
    for sid, source in enumerate(index['sources']):
        run = Path(source['run'])
        if digest(run / 'result.json') != source['result_sha256'] or digest(run / 'manifest.json') != source['manifest_sha256']:
            raise ValueError('Changed source')
        manifest = json.loads((run / 'manifest.json').read_text())
        result = json.loads((run / 'result.json').read_text())
        config_path = Path(manifest['source']) / 'resolved_config.json'
        if digest(config_path) != manifest['source_config_sha256']:
            raise ValueError('Changed config')
        robots = [RobotFK(json.loads(config_path.read_text()), side) for side in ('left', 'right')]
        start = manifest['pulse_start']
        base = {f['offset']: f for f in result['branches']['base']['frames']}
        task = manifest['task']
        entity = MAIN[task]
        for name, branch in result['branches'].items():
            if name == 'base':
                continue
            side, axis, sign = name.split('_')
            arm = ('left', 'right').index(side)
            robot = robots[arm]
            direction = np.eye(3)['xyz'.index(axis)] * (1 if sign == 'plus' else -1)
            with np.load(run / f'transition_{name}.npz') as z:
                planned = z['planned_cartesian_residual'][start + 3, arm * 7:arm * 7 + 3] * .01
            frames = {f['offset']: f for f in branch['frames']}
            data = {}
            for offset in (start, start + 4, start + 8, 32):
                actual, reference = frames[offset]['physical_offline_only'], base[offset]['physical_offline_only']
                delta = robot.pose(actual['arm_qpos'][arm])[:3, 3] - robot.pose(reference['arm_qpos'][arm])[:3, 3]
                data[str(offset - start)] = dict(link_delta_m=delta.tolist(), norm_m=float(np.linalg.norm(delta)), requested_axis_cosine=float(delta @ direction / np.linalg.norm(delta)) if np.linalg.norm(delta) > 1e-8 else None)
            before = base[start]['physical_offline_only']
            before_tip = robot.pose(before['arm_qpos'][arm])[:3, 3]
            history = base[start - 4]['physical_offline_only']
            dt = base[start]['physics_time_s'] - base[start - 4]['physics_time_s']
            obj = np.asarray(before['offline_only']['object_poses'][entity][:3])
            old = np.asarray(history['offline_only']['object_poses'][entity][:3])
            rows.append(dict(task=task, seed=manifest['source_seed'], source_index=sid, branch=name, planned_peak_m=float(np.linalg.norm(planned)), realized=data, object_origin_distance_at_start_m=float(np.linalg.norm(obj - before_tip)), historical_object_speed_m_s=float(np.linalg.norm(obj - old) / dt)))
    reports = []
    for task in MAIN:
        selected = [r for r in rows if r['task'] == task]
        for horizon in (4, 8, 24):
            values = [r['realized'][str(horizon)] for r in selected]
            reports.append(dict(task=task, horizon=horizon, rows=len(selected), scenes=len({r['seed'] for r in selected}), planned_peak_m=distribution([r['planned_peak_m'] for r in selected]), realized_link_norm_m=distribution([r['norm_m'] for r in values]), requested_axis_cosine=distribution([r['requested_axis_cosine'] for r in values if r['requested_axis_cosine'] is not None]), opposite_requested_axis=sum(r['requested_axis_cosine'] is not None and r['requested_axis_cosine'] < 0 for r in values)))
    if max(r['realized']['0']['norm_m'] for r in rows) > 1e-6:
        raise ValueError('Different measured control-link common origin')
    out = new_run('pulse_local_execution', group=('reports', 'audits'))
    write_json(out / 'result.json', dict(status='complete', dataset=str(args.dataset), dataset_index_sha256=digest(args.dataset / 'index.json'), code_sha256=digest(Path(__file__)), rows=rows, reports=reports, scope='Measured FK control-link branch-minus-base differences, not exact TCP or object surface distance. h4 is sample near pulse peak, not continuous maximum; h8 ends requested pulse but physical deviations can persist. Historical object speed and current origin distance are privileged diagnostic only; no future label filtering or online router.'))
    print(out, flush=True)

if __name__ == '__main__':
    main()

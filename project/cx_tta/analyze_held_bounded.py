"""Saved-data stage, clipping and coupling analysis; privileged diagnostics only."""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit import digest
from .audit_local_geometry import pose_matrix, error
from .handoff_support import matching_links
from .paths import write_json
from .robot_kinematics import RobotFK


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = read(run/'manifest.json')
    if not read(run/'audit.json')['passed']:
        raise ValueError('Full paired saved-data audit must pass first')
    support = read(Path(manifest['support_run'])/'hold_native/fork.json')
    config = read(Path(manifest['source'])/'resolved_config.json')
    fk = RobotFK(config, 'right')
    caps = np.asarray(manifest['caps'])
    reports = []
    for branch in manifest['branches']:
        folder = run/branch
        result = read(folder/'result.json')
        rows = [x for x in lines(folder/'trace.jsonl') if x['advanced']]
        fork = read(folder/'fork.json')['diagnostics']['physical']
        hand0 = fk.pose(fork['arm_qpos'][1])
        obj0 = pose_matrix(fork['offline_only']['object_poses']['shoe'])
        relative0 = np.linalg.inv(hand0)@obj0
        coupling = []
        for row in rows:
            state = row['offline_only']['physical']
            hand = fk.pose(state['arm_qpos'][1])
            obj = pose_matrix(state['offline_only']['object_poses']['shoe'])
            relative = np.linalg.inv(hand)@obj
            drift = error(relative, relative0)
            contacts = matching_links(row['offline_only']['contacts'], support['object_name'], support['finger_links'])
            coupling.append(dict(count=row['count'], **drift, both_finger_contact=len(contacts)==2,
                ee_rise_m=float(hand[2,3]-hand0[2,3]), object_rise_m=float(obj[2,3]-obj0[2,3])))
        plans = {int(p.stem.split('_')[1]): read(p) for p in folder.glob('plan_*.json')}
        measurements = lines(folder/'measurements.jsonl')
        phases = []
        for phase in sorted({x['phase'] for x in rows if x['desired'] is not None}):
            phase_rows = [x for x in rows if x.get('phase')==phase and x['desired'] is not None]
            m = [x for x in measurements if x['phase']==phase]
            pid = phase_rows[0]['plan']
            meta = plans[pid]
            desired = np.asarray([x['desired'] for x in phase_rows])
            base = np.asarray([x['base'] for x in phase_rows])
            action = np.asarray([x['action'] for x in phase_rows])
            clipped = np.abs(desired-base)>caps+1e-6
            phase_info = dict(phase=phase, kind=meta['kind'], side=meta['side'],
                first_count=phase_rows[0]['count'], last_count=phase_rows[-1]['count'], actions=len(phase_rows),
                plans=len({x['plan'] for x in phase_rows}), completed=phase<result['completed_phases'],
                alignment_actions=sum(x.get('mode')=='alignment' for x in phase_rows),
                clipped_coordinate_fraction=float(clipped.mean()),
                right_joint_clipped_fraction=clipped[:,7:13].mean(axis=0).tolist(),
                right_gripper_clipped_fraction=float(clipped[:,13].mean()),
                max_residual=np.abs(action-base).max(axis=0).tolist(),
                final_measurement=m[-1] if m else None)
            phase_coupling=[x for x in coupling if x['count'] in {r['count'] for r in phase_rows}]
            phase_info['coupling']=dict(both_contact_actions=sum(x['both_finger_contact'] for x in phase_coupling),actions=len(phase_coupling),max_relative_drift_m=max(x['distance_m'] for x in phase_coupling),max_relative_angle_deg=max(x['angle_deg'] for x in phase_coupling),final=phase_coupling[-1])
            if meta['kind']=='move' and m:
                phase_info['goal']=m[0]['goal']
                phase_info['minimum_distance_m']=min(x['arrival']['distance_m'] for x in m)
                phase_info['minimum_angle_deg']=min(x['arrival']['angle_degrees'] for x in m)
                phase_info['first_arrival']=m[0]['arrival']
            phases.append(phase_info)
        lost = next((x['count'] for x in coupling if not x['both_finger_contact']), None)
        obj_final=np.asarray(result['final_diagnostics']['physical']['offline_only']['object_poses']['shoe'])
        quat_final=obj_final[3:].copy()
        if quat_final[0]<0:quat_final*=-1
        report = dict(final_task_errors=dict(xy_absolute_m=np.abs(obj_final[:2]-[0.,-.08]).tolist(),quaternion_component_absolute=np.abs(quat_final-[.5,.5,-.5,-.5]).tolist(),grippers_open=result['final_diagnostics']['grippers_open']),branch=branch, success=result['success'], native_count=result['action_count'],
            teacher_exit=result['teacher_exit'], completed_phases=result['completed_phases'],
            teacher_actions=result['teacher_actions'], queries=result['queries'],
            phases=phases, first_missing_both_contact_count=lost,
            both_contact_actions=sum(x['both_finger_contact'] for x in coupling),
            coupling_actions=len(coupling), max_relative_drift_m=max(x['distance_m'] for x in coupling),
            max_relative_angle_deg=max(x['angle_deg'] for x in coupling),
            maximum_ee_rise_m=max(x['ee_rise_m'] for x in coupling),
            maximum_object_rise_m=max(x['object_rise_m'] for x in coupling),
            final_coupling=coupling[-1])
        write_json(folder/'coupling_analysis.json', dict(rows=coupling,
            interpretation='Relative to initial held state; after intended release, contact loss/drift is expected. Contact alone is not a grasp certificate.'))
        reports.append(report)
    source_paths = [run/'audit.json', run/'manifest.json', run/'result.json']
    for b in manifest['branches']:
        source_paths.extend(run/b/n for n in ('trace.jsonl','result.json','queries.jsonl'))
        if (run/b/'measurements.jsonl').exists(): source_paths.append(run/b/'measurements.jsonl')
    cache_reference=Path(manifest['compiler_cache_source'])/'common_base.npz'
    with np.load(run/'common_base.npz',allow_pickle=False) as current,np.load(cache_reference,allow_pickle=False) as historical:
        first_block=dict(exactly_equal=bool(np.array_equal(current['base'],historical['base'])),max_absolute_difference=float(np.max(np.abs(current['base']-historical['base']))),reference=str(cache_reference),reference_sha256=digest(cache_reference),interpretation='Recorded fresh inference comparison, not future action replay; later branches query their own observations.')
    write_json(run/'analysis.json', dict(reports=reports,fresh_first_block_comparison=first_block,
        source_sha256={str(p):digest(p) for p in source_paths},
        analysis_code_sha256={str(p):digest(p) for p in (Path(__file__), Path(__file__).with_name('robot_kinematics.py'),Path(__file__).with_name('handoff_support.py'),Path(__file__).with_name('audit_local_geometry.py'))},
        geometry_source_sha256={str(p):digest(p) for p in (Path(manifest['source'])/'resolved_config.json', fk.urdf)},
        scope='One preselected developer held state, one policy seed, privileged teacher feasibility; no adapter/holdout/generalization score.'))
    print(json.dumps([{k:v for k,v in x.items() if k!='phases'} for x in reports], indent=2))


if __name__ == '__main__':
    main()

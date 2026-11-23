"""Conservative SO(3) endpoint support certificate for a recorded residual box."""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .audit import digest
from .audit_local_geometry import error, pose_matrix
from .paths import write_json
from .robot_kinematics import RobotFK


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--phase', type=int, default=2)
    args = parser.parse_args()
    run = args.run.resolve()
    if not read(run/'audit.json')['passed']:
        raise ValueError('Complete saved-action audit required')
    manifest = read(run/'manifest.json')
    config_path = Path(manifest['source'])/'resolved_config.json'
    fk = RobotFK(read(config_path), 'right')
    joints = {x.get('name'): x for x in ET.parse(fk.urdf).getroot().findall('joint')}
    if not all(joints[n].get('type') in ('revolute','continuous') for n in fk.names):
        raise ValueError('Certificate assumes this serial revolute chain')
    # Every variable joint on the tip chain must be one of the bounded joints.
    node, moving = fk.tip, []
    while node in fk.joints:
        parent, name, kind, _, _ = fk.joints[node]
        if kind != 'fixed': moving.append(name)
        node = parent
    if set(moving) != set(fk.names) or len(moving) != len(fk.names):
        raise ValueError('Unbounded moving joint on FK tip chain')
    folder = run/'held_compatible'
    trace = [x for x in lines(folder/'trace.jsonl') if x['advanced']]
    rows = [x for x in trace if x.get('phase')==args.phase and x['desired'] is not None]
    measurements = [x for x in lines(folder/'measurements.jsonl')
                    if x['kind']=='move' and x['phase']==args.phase]
    if not rows or not measurements:
        raise ValueError('Recorded motion phase required')
    goal_pose = measurements[0]['goal']
    if not all(x['goal']==goal_pose for x in measurements):
        raise ValueError('Fixed semantic target required')
    goal = pose_matrix(goal_pose)
    caps = np.asarray(manifest['caps'])[7:13]
    if caps.shape!=(6,) or not np.isfinite(caps).all() or np.any(caps<0):
        raise ValueError('Invalid native joint residual caps')
    upper_deg = float(np.degrees(caps.sum()))
    reports = []
    for row in rows:
        base = fk.pose(np.asarray(row['base'])[7:13])
        angle = error(base, goal)['angle_deg']
        reports.append(dict(native_count=row['count'],query=row['query'],
            base_goal_angle_deg=angle,
            command_orientation_lower_bound_deg=max(0.,angle-upper_deg)))
    states = {x['count']:x['offline_only']['physical'] for x in trace}
    fork = read(folder/'fork.json')
    states[fork['prefix_actions']] = fork['diagnostics']['physical']
    parity = [error(fk.pose(states[x['native_count']]['arm_qpos'][1]),
                    pose_matrix(x['arrival']['pose'])) for x in measurements]
    maximum_distance = max(x['distance_m'] for x in parity)
    maximum_angle = max(x['angle_deg'] for x in parity)
    if maximum_distance>1e-5 or maximum_angle>.001:
        raise ValueError('FK/native measured frame parity failed')
    # Independent finite checks on saved applied and desired commands. These
    # validate numerical/frame conventions, not experimental task performance.
    applied_changes = [error(fk.pose(np.asarray(x['action'])[7:13]),
                             fk.pose(np.asarray(x['base'])[7:13]))['angle_deg'] for x in rows]
    if max(applied_changes)>upper_deg+1e-4:
        raise ValueError('Applied command violates derived orientation bound')
    sources = (run/'audit.json',folder/'trace.jsonl',folder/'measurements.jsonl',
               config_path,fk.urdf,Path(__file__),Path(__file__).with_name('robot_kinematics.py'),
               Path(__file__).with_name('audit_local_geometry.py'))
    report = dict(passed=True,phase=args.phase,joint_names=fk.names,
        joint_types=[joints[n].get('type') for n in fk.names],caps_rad=caps.tolist(),
        rotation_change_upper_bound_deg=upper_deg,
        derivation='SO(3) bi-invariant geodesic triangle inequality: changing serial revolute joint angles by delta_i changes endpoint rotation by at most sum(abs(delta_i)). Thus d(goal,R(q)) >= max(0,d(goal,R(base))-sum(caps)). Ignoring URDF limits gives a superset of allowed commands; no optimizer assumption.',
        goal_pose=goal_pose,move_tolerance_deg=3.,actions=len(reports),
        actions_with_lower_bound_above_tolerance=sum(x['command_orientation_lower_bound_deg']>3. for x in reports),
        minimum_lower_bound_deg=min(x['command_orientation_lower_bound_deg'] for x in reports),
        maximum_lower_bound_deg=max(x['command_orientation_lower_bound_deg'] for x in reports),
        applied_command_max_rotation_change_deg=max(applied_changes),
        fk_native_parity=dict(poses=len(parity),max_distance_m=maximum_distance,max_angle_deg=maximum_angle),
        rows=reports,source_sha256={str(p):digest(p) for p in sources},
        scope='Recorded command endpoint support only; not a global dynamic impossibility statement, or a statement about future base actions under different observations/interventions. Privileged diagnostic, no adapter/generalization claim.')
    write_json(run/'orientation_support.json',report)
    (run/'source_snapshot/orientation_support.py').write_text(Path(__file__).read_text())
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_sha256')},indent=2))


if __name__=='__main__':
    main()

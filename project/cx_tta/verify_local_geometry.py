"""Independent saved endpoint and object-hand coupling re-evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_local_geometry import pose_matrix,error,joint_bounds
from .paths import write_json
from .robot_kinematics import RobotFK

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();read=lambda p:json.loads(p.read_text())
    manifest=read(run/'manifest.json');result=read(run/'result.json');source=Path(manifest['source']);sm=read(source/'manifest.json');config=read(Path(sm['source'])/'resolved_config.json');fk=RobotFK(config,'right');limits=joint_bounds(fk)
    rows=[json.loads(x) for x in (source/'compatible/trace.jsonl').read_text().splitlines()];states={64:read(source/'compatible/fork.json')['diagnostics']['physical']};states.update({x['count']:x['offline_only']['physical'] for x in rows if x['advanced']})
    checks=dict(sources=all(digest(Path(p))==h for p,h in manifest['source_sha256'].items()),code=all(digest(Path(p))==h for p,h in manifest['code_sha256'].items()),bounds=True,endpoints=True,transport=True,local_search=True,coupling=True)
    initial_object=pose_matrix(states[64]['offline_only']['object_poses']['shoe'])
    for record in result['records']:
        fixed=np.asarray(record['goals']['fixed']['target_matrix']);transported=np.asarray(record['goals']['object_transport']['target_matrix'])
        obj=pose_matrix(states[record['native_count']]['offline_only']['object_poses']['shoe'])
        checks['transport'] &= np.allclose(transported,obj@np.linalg.inv(initial_object)@fixed,atol=1e-12,rtol=0)
        for name,g in record['goals'].items():
            target=np.asarray(g['target_matrix']);opt=g['optimized'];lo=np.maximum(np.asarray(record['base'])-.1,limits[:,0]);hi=np.minimum(np.asarray(record['base'])+.1,limits[:,1])
            checks['bounds'] &= np.array_equal(lo,np.asarray(opt['lower'])) and np.array_equal(hi,np.asarray(opt['upper']))
            feasible=[]
            for trial in opt['trials']:
                q=np.asarray(trial['qpos']);err=error(fk.pose(q),target);valid=bool(np.all(q>=lo-1e-8) and np.all(q<=hi+1e-8) and err['angle_deg']<=3.+1e-5)
                checks['endpoints'] &= abs(err['distance_m']-trial['distance_m'])<1e-10 and abs(err['angle_deg']-trial['angle_deg'])<1e-7 and valid==trial['endpoint_orientation_feasible']
                if valid:feasible.append(trial)
            checks['local_search'] &= len(opt['trials'])==8 and (not feasible and opt['best'] is None or feasible and opt['best']==min(feasible,key=lambda t:t['distance_m']))
    binding=read(run/'relative_coupling.json');checks['sources'] &= all(digest(Path(p))==h for p,h in binding['source_code_binding'].items())
    for branch,first,last,reported in [('compatible',100,143,binding),('base',65,96,binding['base_first_block'])]:
        trace=[json.loads(x) for x in (source/branch/'trace.jsonl').read_text().splitlines()];transforms=[];contact_frames=0
        if branch=='base':
            state=read(source/branch/'fork.json')['diagnostics']['physical'];reference=np.linalg.inv(fk.pose(state['arm_qpos'][1]))@pose_matrix(state['offline_only']['object_poses']['shoe'])
        for row in trace:
            if row['advanced'] and first<=row['count']<=last:
                state=row['offline_only']['physical'];transforms.append(np.linalg.inv(fk.pose(state['arm_qpos'][1]))@pose_matrix(state['offline_only']['object_poses']['shoe']))
                contact_frames+=all(sorted(['041_shoe',link]) in row['offline_only']['contacts'] for link in ['fr_link7','fr_link8'])
        reference=transforms[0] if branch=='compatible' else reference;errs=[error(t,reference) for t in transforms]
        checks['coupling'] &= len(transforms)==reported['frames'] and contact_frames==reported['both_finger_contact_frames'] and abs(max(e['distance_m'] for e in errs)-reported['max_relative_position_drift_m'])<1e-12 and abs(max(e['angle_deg'] for e in errs)-reported['max_relative_orientation_drift_deg'])<1e-9
    checks={k:bool(v) for k,v in checks.items()};report=dict(passed=all(checks.values()),checks=checks,verifier_sha256=digest(Path(__file__)),scope='offline recorded geometry only; nonconverged optimizer points not certified as optimum or executable trajectory')
    write_json(run/'verification.json',report);print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()

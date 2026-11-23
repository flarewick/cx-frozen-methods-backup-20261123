"""CPU feasibility only: bounded world-axis proposals, never simulated outcomes."""
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .paths import ROOT, new_run, write_json
from .robot_kinematics import RobotFK


def propose(robot, q, direction, distance=.005, cap=.025, damping=.01):
    q=np.asarray(q,float); pose=robot.pose(q); jac=np.zeros((6,6)); eps=1e-5
    for i in range(6):
        d=np.zeros(6);d[i]=eps
        plus,minus=robot.pose(q+d),robot.pose(q-d)
        jac[:3,i]=(plus[:3,3]-minus[:3,3])/(2*eps)
        jac[3:,i]=Rotation.from_matrix(plus[:3,:3]@minus[:3,:3].T).as_rotvec()/(2*eps)
    metric=np.array([1,1,1,.1,.1,.1]);j=metric[:,None]*jac
    target=np.r_[np.asarray(direction)*distance,np.zeros(3)]
    delta=j.T@np.linalg.solve(j@j.T+damping**2*np.eye(6),target)
    factor=min(1.,cap/max(np.max(np.abs(delta)),1e-12))
    limits={x.get('name'):x.find('limit') for x in ET.parse(robot.urdf).getroot().findall('joint')}
    for i,name in enumerate(robot.names):
        limit=limits[name]
        if limit is None:continue
        low,high=float(limit.get('lower','-inf')),float(limit.get('upper','inf'))
        if q[i]<low-1e-5 or q[i]>high+1e-5:raise ValueError('Base violates URDF limits')
        if delta[i]>0:factor=min(factor,max(0.,(high-q[i])/delta[i]))
        elif delta[i]<0:factor=min(factor,max(0.,(low-q[i])/delta[i]))
    delta*=factor; achieved=robot.pose(q+delta)
    translation=achieved[:3,3]-pose[:3,3]
    return delta,dict(distance_m=float(np.linalg.norm(translation)),along_axis_m=float(translation@direction),cosine=float(translation@direction/max(np.linalg.norm(translation),1e-12)),rotation_rad=float(np.linalg.norm(Rotation.from_matrix(achieved[:3,:3]@pose[:3,:3].T).as_rotvec())),max_joint_delta=float(np.max(np.abs(delta))))


def main():
    run=new_run('cartesian_feasibility',group=('effect_learning',))
    h=json.loads((ROOT/'outputs/reports/audits/cumulative_probes_20260928.json').read_text())['records'][0]
    sources=json.loads((__import__('pathlib').Path(h['dataset'])/'index.json').read_text())['sources']
    rows=[]
    for source in sources:
        path=__import__('pathlib').Path(source['run']); manifest=json.loads((path/'manifest.json').read_text())
        config=json.loads((__import__('pathlib').Path(manifest['source'])/'resolved_config.json').read_text())
        with np.load(path/'checkpoint.npz') as z:base=z['base_actions']
        for side,start in [('left',0),('right',7)]:
            robot=RobotFK(config,side)
            for t in (3,11,19,27):
                for axis in range(3):
                    for sign in (-1,1):
                        direction=np.eye(3)[axis]*sign
                        delta,metrics=propose(robot,base[t,start:start+6],direction)
                        rows.append(dict(job=source['id'],side=side,action_index=t,axis=axis,sign=sign,**metrics))
    summary={k:dict(min=float(np.min([r[k] for r in rows])),median=float(np.median([r[k] for r in rows])),max=float(np.max([r[k] for r in rows]))) for k in ('distance_m','cosine','rotation_rad','max_joint_delta')}
    write_json(run/'result.json',dict(scope='CPU planned link displacement only; no contact, realized motion, safety or efficacy claim. Fixed 5mm request, .025rad joint cap, .01 damping, rotation metric .1m/rad. Four predetermined action positions; no outcome selection.',sources=len(sources),rows=rows,summary=summary))
    print(json.dumps(dict(output=str(run),count=len(rows),summary=summary),indent=2))

if __name__=='__main__':main()

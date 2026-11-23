"""PRIVILEGED 3D-state diagnostic replacing only the 84D image-relation block."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .audit import digest
from .paths import new_run,write_json
from .robot_kinematics import RobotFK
from .short_window_dataset import load_jobs


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);args=p.parse_args()
    index=json.loads((args.dataset/'index.json').read_text())
    if digest(args.dataset/'inputs_oracle.npz')!=index['inputs_sha256'] or digest(args.dataset/'labels_offline.npz')!=index['labels_sha256']:raise ValueError('Source changed')
    states={};provenance=[]
    for job in load_jobs():
        run=Path(job['run']);result=json.loads((run/'result.json').read_text());config=json.loads((Path(job['source'])/'resolved_config.json').read_text());robots=[RobotFK(config,s) for s in ('left','right')]
        frames={f['offset']:f for f in result['branches']['base']['frames']};limits=result['branches']['base']['articulation_limits_offline_only']
        objects={}
        for offset in (0,-8):
            frame=frames[offset];physical=frame['physical_offline_only'];truth=physical['offline_only'];poses=[robots[i].pose(physical['arm_qpos'][i]) for i in range(2)]
            for name,value in truth['object_poses'].items():
                xyz=np.array(value[:3]);rot=Rotation.from_quat(np.array(value[3:])[[1,2,3,0]]).as_matrix();features=[]
                for pose in poses:
                    features.extend(pose[:3,:3].T@(xyz-pose[:3,3]));features.extend((pose[:3,:3].T@rot)[:,:2].ravel())
                q=truth['articulation_qpos'].get(name)
                if q is None:features.extend([0.,0.])
                else:
                    if len(q)!=1:raise ValueError('Single DOF oracle only')
                    lo,hi=limits[name][0];features.extend([(q[0]-lo)/max(hi-lo,1e-8),1.])
                objects.setdefault(name,[]).extend(features)
        for name,value in objects.items():
            if len(value)!=40:raise ValueError('Unexpected state width')
            states[job['id'],name]=value
        provenance.append(dict(job=job['id'],result_sha256=digest(run/'result.json')))
    with np.load(args.dataset/'inputs_oracle.npz') as z:data=dict(z)
    original_shape=data['geometry'].shape;data['geometry'][:,:84]=0
    for i,row in enumerate(index['records']):data['geometry'][i,:40]=states[row['job'],row['entity']]
    if data['geometry'].shape!=original_shape:raise ValueError('Architecture changed')
    run=new_run('shared_oracle3d',group=('effect_learning',));np.savez_compressed(run/'inputs_oracle.npz',**data)
    (run/'labels_offline.npz').write_bytes((args.dataset/'labels_offline.npz').read_bytes())
    write_json(run/'index.json',dict(index,inputs_sha256=digest(run/'inputs_oracle.npz'),source_index_sha256=digest(args.dataset/'index.json'),oracle3d_provenance=provenance,scope='PRIVILEGED current/history object 3D poses and joint state relative to measured dual-arm FK. Replaces 84D mask projection block with 40D teacher state plus zeros; no future object state. Diagnostic only, NEVER deployment input.'))
    print(run,flush=True)

if __name__=='__main__':main()

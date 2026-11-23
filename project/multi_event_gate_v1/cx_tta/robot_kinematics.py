"""CPU URDF link FK. Robot proprioception/configuration only; no object truth."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def transform(xyz=(0,0,0),rpy=(0,0,0)):
    out=np.eye(4);out[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();out[:3,3]=xyz;return out


class RobotFK:
    def __init__(self,config,side):
        i=0 if side=='left' else 1
        c=config[side+'_embodiment_config'];self.names=c['arm_joints_name'][i];self.tip=c['move_group'][i]
        self.urdf=Path(config[side+'_robot_file'])/c['urdf_path']
        poses=c['robot_pose'];pose=poses[0 if side=='left' or len(poses)==1 else 1]
        self.root=np.eye(4);self.root[:3,:3]=Rotation.from_quat(np.array(pose[3:])[[1,2,3,0]]).as_matrix();self.root[:3,3]=pose[:3]
        if not config.get('dual_arm_embodied',c.get('dual_arm',False)):
            self.root[0,3]+=(-.5 if side=='left' else .5)*config['embodiment_dis']
        self.joints={}
        for j in ET.parse(self.urdf).getroot().findall('joint'):
            origin=j.find('origin');axis=j.find('axis')
            xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            vec=np.fromstring(axis.get('xyz'),sep=' ') if axis is not None else np.array([1.,0,0])
            if j.get('type')=='fixed':vec=np.array([1.,0,0])  # Fixed joints have no rotation axis.
            if vec.shape!=(3,) or not np.isfinite(vec).all() or np.linalg.norm(vec)<1e-12:raise ValueError('Invalid moving joint axis')
            vec=vec/np.linalg.norm(vec)
            self.joints[j.find('child').get('link')]=(j.find('parent').get('link'),j.get('name'),j.get('type'),transform(xyz,rpy),vec)

    def pose(self,q,link=None):
        q=np.asarray(q,float)
        if q.shape!=(len(self.names),) or not np.isfinite(q).all():raise ValueError('Invalid arm qpos')
        values=dict(zip(self.names,q));chain=[];node=link or self.tip;seen=set()
        while node in self.joints:
            if node in seen:raise ValueError('Cyclic URDF')
            seen.add(node);parent,name,kind,origin,axis=self.joints[node];motion=np.eye(4)
            if kind!='fixed':
                if name not in values:raise ValueError('Unobserved moving joint on chain: '+name)
                if kind in ('revolute','continuous'):motion[:3,:3]=Rotation.from_rotvec(axis*values[name]).as_matrix()
                elif kind=='prismatic':motion[:3,3]=axis*values[name]
                else:raise ValueError('Unsupported URDF joint')
            chain.append(origin@motion);node=parent
        if not chain:raise ValueError('Unknown link')
        out=self.root.copy()
        for t in reversed(chain):out=out@t
        return out


def cartesian_residual(base,proposed,robots):
    """Planned endpoint-link delta per arm, retaining independent gripper command.

    Units: translation / .01m; rotation / .1rad; gripper / .1.
    Not a prediction of realized motion, and not an inverse-kinematics controller.
    """
    base,proposed=np.asarray(base),np.asarray(proposed)
    if base.shape!=proposed.shape or base.ndim!=2 or base.shape[1]!=14:raise ValueError('Invalid action chunk')
    out=[]
    for a,b in zip(base,proposed):
        row=[]
        for start,robot in zip((0,7),robots):
            ta,tb=robot.pose(a[start:start+6]),robot.pose(b[start:start+6])
            row.extend((tb[:3,3]-ta[:3,3])/.01)
            row.extend(Rotation.from_matrix(tb[:3,:3]@ta[:3,:3].T).as_rotvec()/.1)
            row.append((b[start+6]-a[start+6])/.1)
        out.append(row)
    return np.array(out)

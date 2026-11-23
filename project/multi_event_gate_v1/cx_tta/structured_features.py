"""Observable-only camera/spatial/text features; labels and split metadata stay outside."""
from pathlib import Path
import numpy as np
from .audit_pose_capacity import numpy_fk

ARMS=[0,1,2,3,4,5,7,8,9,10,11,12]
CAMERAS=('head','right_wrist','left_wrist')


def split_joint(joint):
    x=np.asarray(joint)
    if x.ndim!=2 or x.shape[1]!=28672 or not np.isfinite(x).all():raise ValueError('Expected finite [N,14*2048] joint features')
    tokens=x.reshape(-1,14,2048)
    visual=tokens[:,:12].reshape(-1,3,2,2,2048).view();text=tokens[:,12:].view()
    visual.setflags(write=False);text.setflags(write=False)
    return visual,text


def project(joint,pre):
    visual,text=split_joint(joint)
    x=np.concatenate([visual.reshape(-1,12,2048),text],axis=1).astype(np.float32)
    x=(x@pre['components'].T-pre['mean'][None,None]@pre['components'].T)/pre['scale']
    return x[:,:12].reshape(-1,3,2,2,len(pre['scale'])),x[:,12:]


def pack_observable(joint,state,past_joint,past_state,has_history,pre,robots):
    """All arguments are available from camera/language/proprioceptive observations."""
    joint=np.asarray(joint);state=np.asarray(state,np.float32);past_state=np.asarray(past_state,np.float32)
    n=len(joint);has=np.asarray(has_history,bool)
    if state.shape!=(n,14) or past_state.shape!=(n,14) or has.shape!=(n,):raise ValueError('Invalid observable shapes')
    if not np.isfinite(state).all() or not np.isfinite(past_state).all():raise ValueError('Nonfinite state')
    current=project(joint,pre);past=project(past_joint,pre)
    scaled=state.copy();scaled[:,ARMS]/=np.pi
    previous=past_state.copy();previous[:,ARMS]/=np.pi
    def ee(x):
        poses=np.stack([numpy_fk(robot,x[:,start:start+6].astype(float)) for robot,start in zip(robots,(0,7))],axis=1)
        return np.concatenate([poses[...,:3,3]/.1,poses[...,:3,:2].reshape(n,2,6)],axis=-1).astype(np.float32)
    out=dict(visual=current[0].astype(np.float32),text=current[1].astype(np.float32),state=scaled,
        past_visual=past[0].astype(np.float32),past_text=past[1].astype(np.float32),past_state=previous,
        has_history=has[:,None].astype(np.float32),ee_pose=ee(state),past_ee_pose=ee(past_state))
    for key in ('past_visual','past_text','past_state','past_ee_pose'):out[key][~has]=0
    for x in out.values():
        if not np.isfinite(x).all():raise ValueError('Nonfinite structured observable')
        x.setflags(write=False)
    return out


def load_structured(run):
    from .probe_candidates import read,verify
    run=Path(run);p=read(run/'protocol.json');verify(p)
    if not read(run/'interface_audit.json')['passed']:raise ValueError('Audited interface required')
    with np.load(run/'inputs.npz') as z:inputs={k:z[k].copy() for k in z.files}
    allowed={'visual','text','state','past_visual','past_text','past_state','has_history','ee_pose','past_ee_pose'}
    if set(inputs)!=allowed:raise ValueError('Unknown forward input')
    return p,inputs

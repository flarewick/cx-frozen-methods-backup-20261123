"""Task-independent, bounded eight-action Cartesian translation probes."""
import numpy as np
from .audit_cartesian_proposals import propose
from .robot_kinematics import cartesian_residual

BRANCHES=('base',)+tuple(f'{side}_{axis}_{sign}' for side in ('left','right') for axis in 'xyz' for sign in ('plus','minus'))

def candidates(base,robots,pulse_start=0):
    base=np.asarray(base)
    if base.shape!=(32,14) or not np.isfinite(base).all() or pulse_start not in (0,8,16,24):
        raise ValueError('Finite 32x14 base and aligned pulse start required')
    # Exact zero at both boundaries; peak request 5mm.
    envelope=np.sin(np.linspace(0,np.pi,8))**2
    envelope[[0,-1]]=0;envelope/=envelope.max()
    out={'base':base.copy()};metrics={}
    for side,start,robot in zip(('left','right'),(0,7),robots):
        for axis,label in enumerate('xyz'):
            for sign,name in ((1,'plus'),(-1,'minus')):
                key=f'{side}_{label}_{name}';actions=base.copy();rows=[]
                for t,weight in enumerate(envelope,start=pulse_start):
                    delta,m=propose(robot,base[t,start:start+6],np.eye(3)[axis]*sign,distance=.005*weight)
                    actions[t,start:start+6]+=delta;rows.append(dict(action_index=t,requested_distance_m=float(.005*weight),**m))
                out[key]=actions;metrics[key]=rows
    return out,dict(proposal_set='cartesian_pulse_v1',pulse_start=pulse_start,pulse_length=8,requested_peak_m=.005,joint_cap_rad=.025,damping=.01,rotation_metric_m_per_rad=.1,envelope=envelope.tolist(),metrics=metrics),{k:cartesian_residual(base,v,robots) for k,v in out.items()}

def pulse_windows(frames,mask,dt,ticks,pulse_start):
    from .short_window_contract import make_windows
    windows=make_windows(frames,mask,dt,ticks,4)
    for window in windows:
        common=window['start']<=pulse_start
        window['same_initial_state_across_branches']=common
        window['label_scope']='common-prefix intervention' if common else 'branch-specific temporal transition'
    return windows

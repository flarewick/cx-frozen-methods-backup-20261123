"""Frozen-command counterfactuals and label-privileged offline diagnostics.

No model, router, simulator, or parameter updates. Selection is an oracle over
a finite action grid, NOT a deployable policy or an execution-success bound.
"""
import numpy as np
from .recovery_fit_metrics import endpoints, recovery_gate
from .audit_pose_capacity import angles
from .audit_local_geometry import joint_bounds

CAPS=np.array([.1]*6+[.2]+[.1]*6+[.2],dtype=np.float64)
ALPHAS=(0.,.25,.5,1.)
FIXED=('base','raw_025','raw_050','raw_100','capped_025','capped_050','capped_100')
SUPPORTS={'raw':[0,1,2,3],'capped':[0,4,5,6]}
ORACLES=('binary_chunk','grid_chunk','grid_arm','guarded_arm')
VARIANTS=FIXED+tuple(f'{s}_{o}' for s in SUPPORTS for o in ORACLES)
TOL={'angle_deg':1e-6,'position_m':1e-9,'gripper_mae':1e-8}


def fixed_actions(base,pred):
    """Only base and candidate inputs; cap first, then scale. No labels/mask."""
    base=base.astype(np.float64);pred=pred.astype(np.float64);delta=pred-base
    bounded=np.clip(delta,-CAPS,CAPS)
    return [base,*[pred.copy() if a==1 else base+a*delta for a in ALPHAS[1:]],
            *[base+a*bounded for a in ALPHAS[1:]]]


def row_metrics(action,target,valid,robots,target_pose=None):
    n,h,_=action.shape;count=valid.sum(1)
    if not np.all(count>0):raise ValueError('Empty command block')
    mask=valid[:,:,None]
    mean=lambda value:(value*mask).sum(1)/count[:,None]
    pp=endpoints(action,robots);tp=endpoints(target,robots) if target_pose is None else target_pose
    angle=angles(pp,tp);distance=np.linalg.norm(pp[...,:3,3]-tp[...,:3,3],axis=-1)
    diff=(action-target).reshape(n,h,2,7)
    arm_mse=mean(np.mean(diff[...,:6]**2,-1));grip_mae=mean(np.abs(diff[...,6]))
    bc=mean(np.mean((diff/CAPS.reshape(2,7))**2,-1))
    rotation=mean(((pp[...,:3,:3]-tp[...,:3,:3])**2).sum((-2,-1))/2/.1**2)
    position=mean(distance**2/.02**2)
    vm=valid[:,1:]&valid[:,:-1];velocity=np.diff(diff[...,:6],axis=1)
    velocity_mse=(np.mean(velocity**2,-1)*vm[:,:,None]).sum(1)/np.maximum(vm.sum(1)[:,None],1)
    q=action.reshape(n,h,2,7)[...,:6];grip=action.reshape(n,h,2,7)[...,6]
    limits=np.stack([joint_bounds(r) for r in robots])
    return dict(angle_deg=mean(angle),position_m=mean(distance),
        pose_under_3deg_1cm_fraction=mean((angle<=3)&(distance<=.01)),
        arm_mse_rad2=arm_mse,gripper_mae=grip_mae,velocity_mse_rad2=velocity_mse,
        gripper_outside_margin=(((grip<-.02)|(grip>1.02))*mask).sum(1),
        joint_limit_violations=(((q<limits[...,0])|(q>limits[...,1]))*mask[...,None]).sum((1,3)),
        dense_bc=bc,rotation_loss=rotation,position_loss=position,score=bc+rotation+position)


def guarded(metric,baseline):
    keep=np.ones_like(metric['score'],dtype=bool)
    for k,tol in TOL.items():keep &= metric[k]<=baseline[k]+tol
    for k in ('gripper_outside_margin','joint_limit_violations'):keep &= metric[k]<=baseline[k]
    return keep


def oracle_indices(metrics,support,mode):
    candidates=SUPPORTS[support];score=np.stack([metrics[i]['score'] for i in candidates])
    if mode=='binary_chunk':
        candidates=[candidates[0],candidates[-1]];score=score[[0,-1]]
    if mode=='guarded_arm':
        allowed=np.stack([guarded(metrics[i],metrics[0]) for i in candidates])
        score=np.where(allowed,score,np.inf)
    if mode.endswith('chunk'):
        selected=np.argmin(score.mean(2),axis=0)[:,None].repeat(2,axis=1)
    else:selected=np.argmin(score,axis=0)
    # Base is first: exact ties choose no correction. One choice per whole arm block.
    return np.asarray(candidates,dtype=np.int16)[selected]


def mix_arms(actions,selection):
    n,h,_=actions[0].shape;source=np.stack(actions).reshape(len(actions),n,h,2,7)
    out=np.empty_like(source[0])
    for arm in (0,1):out[:,:,arm]=source[selection[:,arm],np.arange(n),:,arm]
    return out.reshape(n,h,14)


def material_benefit(metric,baseline):
    # Predeclared 5% dense-score gain and nonworsening pose/grip/range components.
    return guarded(metric,baseline)&((baseline['score']-metric['score'])>np.maximum(1e-8,.05*baseline['score']))


def summarize(metric,action,target,valid,rows,ids,baseline):
    groups=[];arms_index=[0,1,2,3,4,5,7,8,9,10,11,12]
    for group in sorted({rows[i]['group'] for i in ids}):
        ii=np.array([i for i in ids if rows[i]['group']==group]);weights=valid[ii].sum(1)
        velocity_weights=(valid[ii,1:]&valid[ii,:-1]).sum(1);arms={}
        for a,side in enumerate(('left','right')):
            avg=lambda k:float(np.average(metric[k][ii,a],weights=weights))
            out={k:avg(k) for k in ('angle_deg','position_m','gripper_mae','pose_under_3deg_1cm_fraction','score')}
            out.update(arm_rmse_rad=float(np.sqrt(avg('arm_mse_rad2'))),
                velocity_error_rmse_rad=None if not velocity_weights.sum() else float(np.sqrt(np.average(metric['velocity_mse_rad2'][ii,a],weights=velocity_weights))),
                gripper_outside_margin=int(metric['gripper_outside_margin'][ii,a].sum()),
                joint_limit_violations=int(metric['joint_limit_violations'][ii,a].sum()),
                base_angle_deg=float(np.average(baseline['angle_deg'][ii,a],weights=weights)),
                base_position_m=float(np.average(baseline['position_m'][ii,a],weights=weights)),
                base_score=float(np.average(baseline['score'][ii,a],weights=weights)))
            arms[side]=out
        seams=[]
        for before,after in zip(ii[:-1],ii[1:]):
            length=int(valid[before].sum())
            if rows[after]['kind']=='recovery' and rows[before]['frame']+length==rows[after]['frame']:
                seams.append((action[after,0,arms_index]-action[before,length-1,arms_index])-
                             (target[after,0,arms_index]-target[before,length-1,arms_index]))
        groups.append(dict(group=group,task=rows[ii[0]]['task'],rows=len(ii),valid_actions=int(weights.sum()),arms=arms,
            seam_error_rmse_rad=None if not seams else float(np.sqrt(np.mean(np.asarray(seams)**2)))))
    tasks={}
    keys=('angle_deg','position_m','gripper_mae','pose_under_3deg_1cm_fraction','score',
          'arm_rmse_rad','velocity_error_rmse_rad','base_angle_deg','base_position_m','base_score')
    for task in sorted({g['task'] for g in groups}):
        selected=[a for g in groups if g['task']==task for a in g['arms'].values()]
        tasks[task]={k:None if not (values:=[a[k] for a in selected if a[k] is not None]) else float(np.mean(values)) for k in keys}
    summary=dict(groups=groups,tasks=tasks)
    summary['recovery_fit_gate']=recovery_gate(summary)
    summary['preservation_gate']=all(a['angle_deg']<=1 and a['position_m']<=.01 and a['gripper_mae']<=.02
        for g in groups for a in g['arms'].values())
    return summary

"""Full valid native-command metrics; no simulator or success prediction."""
import numpy as np
from .audit_pose_capacity import numpy_fk,angles
from .audit_local_geometry import joint_bounds

ARMS=[0,1,2,3,4,5,7,8,9,10,11,12]

def endpoints(actions,robots):
    return np.stack([numpy_fk(r,actions[...,s:s+6].astype(float)) for r,s in zip(robots,(0,7))],axis=2)

def metrics(pred,base,target,valid,rows,robots):
    pp,bp,tp=[endpoints(x,robots) for x in (pred,base,target)]
    angle=angles(pp,tp);base_angle=angles(bp,tp)
    position=np.linalg.norm(pp[...,:3,3]-tp[...,:3,3],axis=-1)
    base_position=np.linalg.norm(bp[...,:3,3]-tp[...,:3,3],axis=-1)
    groups=[];blocks=[]
    for group in sorted({r['group'] for r in rows}):
        ids=np.array([i for i,r in enumerate(rows) if r['group']==group]);v=valid[ids]
        arms={}
        for a,(side,start) in enumerate(zip(('left','right'),(0,7))):
            q=pred[ids,:,start:start+6];t=target[ids,:,start:start+6];grip=pred[ids,:,start+6];tg=target[ids,:,start+6]
            aa=angle[ids,:,a][v];bb=base_angle[ids,:,a][v];dd=position[ids,:,a][v]
            temporal=v[:,1:]&v[:,:-1];velocity=(np.diff(q,axis=1)-np.diff(t,axis=1))[temporal]
            bounds=joint_bounds(robots[a]);large=bb>=45
            arms[side]=dict(angle_deg=float(aa.mean()),position_m=float(dd.mean()),
                base_angle_deg=float(bb.mean()),base_position_m=float(base_position[ids,:,a][v].mean()),
                pose_under_3deg_1cm_fraction=float(((aa<=3)&(dd<=.01)).mean()),
                angle_p95_deg=float(np.quantile(aa,.95)),position_p95_m=float(np.quantile(dd,.95)),
                arm_rmse_rad=float(np.sqrt(np.mean((q[v]-t[v])**2))),
                gripper_mae=float(np.abs(grip[v]-tg[v]).mean()),
                velocity_error_rmse_rad=None if not len(velocity) else float(np.sqrt(np.mean(velocity**2))),
                large_angle_deg=None if not large.any() else float(aa[large].mean()),
                large_count=int(large.sum()),
                gripper_outside_margin=int(((grip[v]<-.02)|(grip[v]>1.02)).sum()),
                joint_limit_violations=int(((q[v]<bounds[:,0])|(q[v]>bounds[:,1])).sum()))
        seams=[]
        for previous,current in zip(ids[:-1],ids[1:]):
            n=int(valid[previous].sum())
            if rows[current]['kind']=='recovery' and rows[previous]['frame']+n==rows[current]['frame']:
                seam=(pred[current,0,ARMS]-pred[previous,n-1,ARMS])-(target[current,0,ARMS]-target[previous,n-1,ARMS])
                seams.append(seam)
        groups.append(dict(group=group,task=rows[ids[0]]['task'],rows=len(ids),valid_actions=int(v.sum()),arms=arms,
            seam_error_rmse_rad=None if not seams else float(np.sqrt(np.mean(np.stack(seams)**2)))))
    for i,row in enumerate(rows):
        v=valid[i];blocks.append(dict(group=row['group'],frame=row['frame'],valid_actions=int(v.sum()),
            angle_deg=float(angle[i][v].mean()),position_m=float(position[i][v].mean()),
            arm_rmse_rad=float(np.sqrt(np.mean((pred[i][v][:,ARMS]-target[i][v][:,ARMS])**2))),
            gripper_mae=float(np.abs(pred[i][v][:,[6,13]]-target[i][v][:,[6,13]]).mean())))
    tasks={}
    keys=('angle_deg','position_m','base_angle_deg','base_position_m','pose_under_3deg_1cm_fraction',
          'arm_rmse_rad','gripper_mae','velocity_error_rmse_rad','large_angle_deg')
    for task in sorted({r['task'] for r in rows}):
        selected=[g for g in groups if g['task']==task]
        tasks[task]={k:None if not (values:=[v[k] for g in selected for v in g['arms'].values() if v[k] is not None])
                     else float(np.mean(values)) for k in keys}
    return dict(tasks=tasks,groups=groups,blocks=blocks),dict(angle=angle,base_angle=base_angle,position=position)

def recovery_gate(summary):
    # Strict per-arm, per-scene fit gate. Offline path imitation only.
    return all(v['angle_deg']<=3 and v['position_m']<=.01 and v['pose_under_3deg_1cm_fraction']>=.9
        and v['arm_rmse_rad']<=.05 and v['gripper_mae']<=.05
        and (v['velocity_error_rmse_rad'] is None or v['velocity_error_rmse_rad']<=.03)
        and v['gripper_outside_margin']==0 and v['joint_limit_violations']==0
        for g in summary['groups'] for v in g['arms'].values()) and all(
        g['seam_error_rmse_rad'] is None or g['seam_error_rmse_rad']<=.03 for g in summary['groups'])

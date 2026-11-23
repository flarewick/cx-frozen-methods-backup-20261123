"""Recompute pose metrics from saved commands using independent float64 CPU FK."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .audit import digest
from .paths import write_json
from .robot_kinematics import RobotFK
from .audit_local_geometry import joint_bounds

def numpy_fk(robot,q):
    chain=[];node=robot.tip
    while node in robot.joints:
        parent,name,kind,origin,axis=robot.joints[node];chain.append((name,kind,origin,axis));node=parent
    out=np.broadcast_to(robot.root,(*q.shape[:-1],4,4)).copy()
    for name,kind,origin,axis in reversed(chain):
        out=out@origin
        if kind=='fixed':continue
        motion=np.broadcast_to(np.eye(4),out.shape).copy()
        angles=q[...,robot.names.index(name)]
        motion[...,:3,:3]=Rotation.from_rotvec((angles[...,None]*axis).reshape(-1,3)).as_matrix().reshape(*q.shape[:-1],3,3)
        out=out@motion
    return out

def angles(a,b):
    relative=a[...,:3,:3]@b[...,:3,:3].swapaxes(-1,-2)
    return np.degrees(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).reshape(a.shape[:-2])

def episode_metrics(angle,baseangle,translation,mask,large,indices,rows,pred,target,valid):
    result={};episode_rows=[]
    for task in sorted({rows[i]['task'] for i in indices}):
        stats=[]
        for group in sorted({rows[i]['group'] for i in indices if rows[i]['task']==task}):
            ids=[j for j,i in enumerate(indices) if rows[i]['group']==group];m=mask[ids];lm=large[ids];value=dict(task=task,group=group,angle_deg=float(angle[ids][m].mean()),base_angle_deg=float(baseangle[ids][m].mean()),translation_m=float(translation[ids][m].mean()),native_action_mse=float(((pred[ids]-target[ids])**2).mean(-1)[valid[ids]].mean()),large_angle_deg=None,large_base_angle_deg=None,large_reduction_fraction=None,large_under_15deg_fraction=None,bounded_lower_bound_deg=None,large_count=int(lm.sum()))
            if lm.any():value.update(large_angle_deg=float(angle[ids][lm].mean()),large_base_angle_deg=float(baseangle[ids][lm].mean()),large_reduction_fraction=float((angle[ids][lm]<baseangle[ids][lm]).mean()),large_under_15deg_fraction=float((angle[ids][lm]<15).mean()),bounded_lower_bound_deg=float(np.maximum(baseangle[ids][lm]-np.degrees(.6),0).mean()))
            stats.append(value);episode_rows.append(value)
        result[task]={k:None if not (values:=[s[k] for s in stats if s[k] is not None]) else float(np.mean(values)) for k in ('angle_deg','base_angle_deg','translation_m','native_action_mse','large_angle_deg','large_base_angle_deg','large_reduction_fraction','large_under_15deg_fraction','bounded_lower_bound_deg')}
    return result,episode_rows

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();read=lambda f:json.loads(f.read_text());protocol=read(run/'protocol.json');result=read(run/'result.json')
    if result['status']!='complete' or len(result['reports'])!=9:raise ValueError('Nine completed capacity variants required')
    prepared=Path(protocol['prepared'])
    if digest(prepared/'manifest.json')!=protocol['prepared_manifest_sha256']:raise ValueError('Changed prepared manifest')
    for file,sha in dict(protocol['source_sha256'],**protocol['gate_sources']).items():
        if digest(Path(file))!=sha:raise ValueError('Changed source '+file)
    with np.load(prepared/'data.npz') as z:data={k:z[k].copy() for k in z.files}
    if digest(prepared/'data.npz')!=protocol['data_sha256']:raise ValueError('Changed training arrays')
    config=read(Path(protocol['config']));robots=[RobotFK(config,s) for s in ('left','right')];rows=protocol['rows'];files=[];summaries=[];episodes=[];parity=[]
    fit={r['group'] for r in rows if r['pilot_split']=='fit'};held={r['group'] for r in rows if r['pilot_split']=='inner_held'}
    if fit&held:raise ValueError('Episode overlap')
    for report in result['reports']:
        seed,variant=report['seed'],report['variant'];files.append(run/f'{variant}_s{seed}.pt')
        for part in ('inner_held','anchor_held'):
            path=run/f'{variant}_s{seed}_{part}.npz';files.append(path)
            with np.load(path) as z:saved={k:z[k].copy() for k in z.files}
            indices=saved['indices'];expected=np.array([i for i,r in enumerate(rows) if r['pilot_split']==part])
            if not np.array_equal(indices,expected) or not all(np.array_equal(saved[k],data[k][indices]) for k in ('base','target','valid')):raise ValueError('Partition/target mismatch')
            offsets=protocol['offsets']
            def end(values):return np.stack([numpy_fk(r,values[:,offsets,start:start+6].astype(float)) for r,start in zip(robots,(0,7))],2)
            pp,bp,tp=[end(saved[k]) for k in ('predicted','base','target')]
            angle,ba=angles(pp,tp),angles(bp,tp);translation=np.linalg.norm(pp[...,:3,3]-tp[...,:3,3],axis=-1);mask=np.broadcast_to(saved['valid'][:,offsets,None],angle.shape);large=mask&(ba>=45.)
            maxangle=float(np.max(np.abs(angle-saved['angle'])));maxbase=float(np.max(np.abs(ba-saved['base_angle'])));maxtrans=float(np.max(np.abs(translation-saved['translation'])))
            if maxangle>.06 or maxbase>.06 or maxtrans>2e-6:raise ValueError('Independent FK/angle parity failed')
            if not np.array_equal(mask,saved['mask']) or not np.array_equal(large,saved['large']):raise ValueError('Threshold/validity selection mismatch')
            delta=saved['predicted']-saved['base'];caps=np.array([.1]*6+[.2]+[.1]*6+[.2])
            if variant=='bounded_bc' and (np.max(np.abs(delta)-caps)>1e-6 or np.max(angles(pp,bp))>np.degrees(.6)+1e-3):raise ValueError('Bounded support violation')
            metrics,erows=episode_metrics(angle,ba,translation,mask,large,indices,rows,saved['predicted'],saved['target'],saved['valid'])
            for task,values in metrics.items():
                for k,value in values.items():
                    actual=report['metrics'][part][task][k]
                    if value is None:
                        if actual is not None:raise ValueError('Empty subset mismatch')
                    elif abs(value-actual)>(.06 if 'angle' in k or k=='bounded_lower_bound_deg' else 1e-5):raise ValueError('Reported aggregate mismatch '+k)
            summaries.append(dict(seed=seed,variant=variant,partition=part,metrics=metrics));episodes.extend(dict(seed=seed,variant=variant,partition=part,**r) for r in erows);parity.append(dict(seed=seed,variant=variant,partition=part,max_angle_difference_deg=maxangle,max_translation_difference_m=maxtrans))
    write_json(run/'independent_metrics.json',dict(records=summaries,episodes=episodes))
    write_json(run/'audit.json',dict(passed=True,complete_variants=9,expert_fit_groups=len(fit),expert_held_groups=len(held),shared_parameter_counts=sorted({r['parameters'] for r in result['reports']}),checks=['Exact data/target/partition identities','Source hashes including feature and target/repetition gates','No episode overlap','Independent float64 scipy FK and SO(3) metrics','Original bounded joint/rotation support','Recorded exact checkpoint reload'],fk_parity=parity,source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'history.json',Path(__file__),*files]},scope='Audit of a small offline shared adapter capacity comparison. Expert action endpoint orientation is not an object-progress/recovery label; no closed-loop success or adapter-task holdout claim. Proposal variants change command support.'))
    print(json.dumps(dict(passed=True,variants=9,held_groups=len(held),max_angle_parity_deg=max(r['max_angle_difference_deg'] for r in parity)),indent=2))

if __name__=='__main__':main()

"""Independent dense FK/aggregation, source, normalization and budget audit."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_pose_capacity import numpy_fk,angles
from .dataset import context_features
from .paths import write_json
from .robot_kinematics import RobotFK
from .audit_local_geometry import joint_bounds

def read(p):return json.loads(p.read_text())

def strict_fit(summary):
    for group in summary['groups']:
        if group['seam_error_rmse_rad'] is not None and group['seam_error_rmse_rad']>.03:return False
        for arm in group['arms'].values():
            if (arm['angle_deg']>3 or arm['position_m']>.01 or arm['pose_under_3deg_1cm_fraction']<.9
                or arm['arm_rmse_rad']>.05 or arm['gripper_mae']>.05 or arm['gripper_outside_margin']
                or arm['joint_limit_violations'] or (arm['velocity_error_rmse_rad'] is not None and arm['velocity_error_rmse_rad']>.03)):
                return False
    return True

def preservation(summary,*,anchor):
    return all(v['angle_deg']<=1 and v['position_m']<=.01 and (not anchor or v['gripper_mae']<=.02)
               for g in summary['groups'] for v in g['arms'].values())

def close(actual,expected,key):
    if expected is None:
        if actual is not None:raise ValueError('Unexpected metric '+key)
    elif abs(actual-expected)>1e-6:raise ValueError(f'Metric mismatch {key}: {actual} vs {expected}')

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path);run=parser.parse_args().run.resolve()
    p=read(run/'protocol.json');r=read(run/'result.json')
    if r['status']!='complete':raise ValueError('Completed experiment required')
    for file,sha in p['source_sha256'].items():
        if digest(Path(file))!=sha:raise ValueError('Changed source '+file)
    if digest(run/'data.npz')!=p['data_sha256'] or digest(run/'weights.npz')!=p['weights_sha256']:raise ValueError('Changed frozen arrays')
    with np.load(run/'data.npz') as z:data={k:z[k].copy() for k in z.files}
    expert=Path(p['expert_source']);context=Path(p['context_source']);ca=read(context/'audit.json')
    if ca['schema']!='recovery_context_v2_state_div_pi' or not ca['passed']:raise ValueError('Corrected context required')
    with np.load(expert/'data.npz') as z:
        n=len(z['base'])
        if any(not np.array_equal(z[k],data[k][:n]) for k in ('semantic','coarse','base','target','valid')):
            raise ValueError('Expert data mismatch')
    with np.load(context/'data.npz') as z:
        if any(not np.array_equal(z[k],data[k][n:]) for k in ('semantic','coarse','base','target','valid')):
            raise ValueError('Recovery data mismatch')
    old=Path(ca['source']);oldp=read(old/'protocol.json');state_checks=[]
    for name in ('semantic','coarse'):
        with np.load(expert/f'{name}_preprocess.npz') as z:
            decoded=data[name][n:]*z['context_std']+z['context_mean']
        normalized=[]
        for row in oldp['rows']:
            with np.load(row['observation']) as z:
                normalized.append(context_features([z[k+'_rgb'] for k in ('head','right','left')],z['command_state'])[:14])
        normalized=np.stack(normalized);err=float(np.abs(decoded[:,64:78]-normalized).max())
        if err>2e-6:raise ValueError('Current native state/pi contract failed')
        for i,j in enumerate(oldp['history_indices']):
            expected=np.zeros(14) if j<0 else normalized[j]
            if np.max(np.abs(decoded[i,142:156]-expected))>2e-6:raise ValueError('Historical state/pi contract failed')
        state_checks.append(dict(representation=name,max_state_contract_error=err))
    rows=p['rows'];parts=p['parts'];groups={s:{rows[i]['group'] for i in ids} for s,ids in parts.items()}
    for s in ('recovery','ordinary','anchor_fit'):
        if groups[s]&(groups['inner_held']|groups['anchor_held']):raise ValueError('Training/held overlap')
    if any(rows[i]['pilot_split']!='fit' for i in parts['ordinary']) or any(rows[i]['pilot_split']!='anchor_fit' for i in parts['anchor_fit']):
        raise ValueError('Invalid preservation source split')
    with np.load(run/'weights.npz') as z:
        for condition,setting in p['conditions'].items():
            expected=np.zeros(len(rows),np.float32)
            for component in setting['components']:
                ids=parts[component];tasks={rows[i]['task'] for i in ids};counts=Counter(rows[i]['group'] for i in ids)
                taskgroups={t:{rows[i]['group'] for i in ids if rows[i]['task']==t} for t in tasks}
                w=np.array([1/(len(tasks)*len(taskgroups[rows[i]['task']])*counts[rows[i]['group']]) for i in ids]);w/=w.sum()
                expected[ids]=w
            if not np.array_equal(z[condition],expected):raise ValueError('Training weights changed')
    expected={(c,s,u) for c in p['conditions'] for s in p['seeds'] for u in p['updates']}
    if {(v['condition'],v['seed'],v['update']) for v in r['reports']}!=expected or len(r['reports'])!=len(expected):raise ValueError('Budget/seed coverage incomplete')
    for seed in p['seeds']:
        if len({v['initial_state_sha256'] for v in r['reports'] if v['seed']==seed})!=1:raise ValueError('Initialization differs across conditions')
    robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')];files=[];maxparity=0.;validated=0
    for report in r['reports']:
        if report['parameters']!=40718 or report['reload']!='exact':raise ValueError('Architecture/reload mismatch')
        if digest(Path(report['checkpoint']))!=report['checkpoint_sha256']:raise ValueError('Changed checkpoint')
        files.append(Path(report['checkpoint']))
        required=set(parts) if report['update']==max(p['updates']) else {'recovery','ordinary','anchor_fit'}
        if {f['part'] for f in report['files']}!=required:raise ValueError('Evaluation coverage differs')
        for entry in report['files']:
            file=Path(entry['path']);files.append(file);part=entry['part']
            if digest(file)!=entry['sha256']:raise ValueError('Changed prediction')
            with np.load(file) as z:saved={k:z[k].copy() for k in z.files}
            ids=saved['indices'];valid=saved['valid'];pred=saved['predicted'];target=saved['target'];base=saved['base']
            if ids.tolist()!=parts[part] or any(not np.array_equal(saved[k],data[k][ids]) for k in ('base','target','valid')):
                raise ValueError('Prediction/source pairing mismatch')
            if not np.isfinite(pred).all():raise ValueError('Invalid commands')
            def end(a):return np.stack([numpy_fk(bot,a[...,offset:offset+6].astype(float)) for bot,offset in zip(robots,(0,7))],2)
            pp,bp,tp=[end(a) for a in (pred,base,target)];a,b=angles(pp,tp),angles(bp,tp)
            pos=np.linalg.norm(pp[...,:3,3]-tp[...,:3,3],axis=-1);bpos=np.linalg.norm(bp[...,:3,3]-tp[...,:3,3],axis=-1)
            if not np.allclose(a,saved['angle'],atol=1e-9) or not np.allclose(pos,saved['position'],atol=1e-12):raise ValueError('Saved geometry mismatch')
            # Independent scalar URDF FK checks first and final VALID states of each block.
            for j in range(len(ids)):
                for t in (0,int(valid[j].sum())-1):
                    for arm,(bot,offset) in enumerate(zip(robots,(0,7))):
                        diff=float(np.max(np.abs(bot.pose(pred[j,t,offset:offset+6].astype(float))-pp[j,t,arm])))
                        maxparity=max(maxparity,diff)
                        if diff>1e-10:raise ValueError('Scalar FK parity failed')
            summary=report['metrics'][part]
            arm_indices=[0,1,2,3,4,5,7,8,9,10,11,12]
            if len(summary['blocks'])!=len(ids):raise ValueError('Block coverage mismatch')
            for j,block in enumerate(summary['blocks']):
                mask=valid[j]
                if block['group']!=rows[ids[j]]['group'] or block['frame']!=rows[ids[j]]['frame'] or block['valid_actions']!=int(mask.sum()):
                    raise ValueError('Block identity mismatch')
                ref=dict(angle_deg=float(a[j][mask].mean()),position_m=float(pos[j][mask].mean()),
                    arm_rmse_rad=float(np.sqrt(np.mean((pred[j][mask][:,arm_indices]-target[j][mask][:,arm_indices])**2))),
                    gripper_mae=float(np.abs(pred[j][mask][:,[6,13]]-target[j][mask][:,[6,13]]).mean()))
                for key,value in ref.items():close(block[key],value,'block_'+key)
            for group in summary['groups']:
                loc=np.array([j for j,i in enumerate(ids) if rows[i]['group']==group['group']]);mask=valid[loc]
                if int(mask.sum())!=group['valid_actions']:raise ValueError('Action count mismatch')
                seams=[]
                for before,after in zip(loc[:-1],loc[1:]):
                    count=int(valid[before].sum())
                    if rows[ids[after]]['kind']=='recovery' and rows[ids[before]]['frame']+count==rows[ids[after]]['frame']:
                        predicted_jump=pred[after,0,arm_indices]-pred[before,count-1,arm_indices]
                        target_jump=target[after,0,arm_indices]-target[before,count-1,arm_indices]
                        seams.append(predicted_jump-target_jump)
                close(group['seam_error_rmse_rad'],None if not seams else float(np.sqrt(np.mean(np.array(seams)**2))),'seam')
                for arm,(side,offset) in enumerate(zip(('left','right'),(0,7))):
                    aa=a[loc,:,arm][mask];bb=b[loc,:,arm][mask];dd=pos[loc,:,arm][mask]
                    q=pred[loc,:,offset:offset+6];t=target[loc,:,offset:offset+6];g=pred[loc,:,offset+6];tg=target[loc,:,offset+6]
                    velocity=(q[:,1:]-q[:,:-1])-(t[:,1:]-t[:,:-1]);vm=mask[:,1:]&mask[:,:-1];large=bb>=45
                    limits=joint_bounds(robots[arm])
                    ref=dict(angle_deg=float(aa.mean()),position_m=float(dd.mean()),base_angle_deg=float(bb.mean()),
                        base_position_m=float(bpos[loc,:,arm][mask].mean()),pose_under_3deg_1cm_fraction=float(np.mean((aa<=3)&(dd<=.01))),
                        angle_p95_deg=float(np.percentile(aa,95)),position_p95_m=float(np.percentile(dd,95)),
                        arm_rmse_rad=float(np.sqrt(np.mean((q[mask]-t[mask])**2))),gripper_mae=float(np.abs(g[mask]-tg[mask]).mean()),
                        velocity_error_rmse_rad=None if not vm.any() else float(np.sqrt(np.mean(velocity[vm]**2))),
                        large_angle_deg=None if not large.any() else float(aa[large].mean()),large_count=int(large.sum()),
                        gripper_outside_margin=int(np.count_nonzero((g[mask]<-.02)|(g[mask]>1.02))),
                        joint_limit_violations=int(np.count_nonzero((q[mask]<limits[:,0])|(q[mask]>limits[:,1]))))
                    for key,value in ref.items():close(group['arms'][side][key],value,key)
            for task,values in summary['tasks'].items():
                for key,value in values.items():
                    selected=[arm[key] for group in summary['groups'] if group['task']==task for arm in group['arms'].values() if arm[key] is not None]
                    close(value,None if not selected else float(np.mean(selected)),'task_'+key)
            validated+=1
        if strict_fit(report['metrics']['recovery'])!=report['recovery_fit_pass']:raise ValueError('Recovery decision mismatch')
    final=[x for x in r['reports'] if x['update']==max(p['updates'])];decisions=[]
    for condition in p['conditions']:
        selected=[x for x in final if x['condition']==condition]
        perseed=[dict(seed=x['seed'],recovery_fit=x['recovery_fit_pass'],ordinary_fit=preservation(x['metrics']['ordinary'],anchor=False),
            anchor_fit=preservation(x['metrics']['anchor_fit'],anchor=True),anchor_held=preservation(x['metrics']['anchor_held'],anchor=True)) for x in selected]
        aggregate={part:{task:{key:None if not (v:=[x['metrics'][part]['tasks'][task][key] for x in selected if x['metrics'][part]['tasks'][task][key] is not None]) else float(np.mean(v))
            for key in selected[0]['metrics'][part]['tasks'][task]} for task in selected[0]['metrics'][part]['tasks']} for part in parts}
        decisions.append(dict(condition=condition,seeds=perseed,aggregate=aggregate,all_fit_and_preserve_gates=all(
            all(v[k] for k in ('recovery_fit','ordinary_fit','anchor_fit','anchor_held')) for v in perseed)))
    write_json(run/'decision.json',dict(status='complete',conditions=decisions,promote=False,
        reason='Offline seen-scene developer fitting only; no independent recovery test or closed-loop success.',
        context_erratum='All recovery training used v2 state/pi context, not S1r v1.',
        scope=p['scope']))
    write_json(run/'audit.json',dict(passed=True,snapshots=len(expected),predictions=validated,models=len(final),
        independent_scalar_fk_max_error=maxparity,state_normalization_checks=state_checks,
        checks=['Dense actual valid-step metrics and independent scalar FK','Frozen expert and corrected recovery contexts',
                'No developer held rows in fitting and exact component weights','Matched initialization/parameter counts/budgets',
                'Exact checkpoint reload and unchanged native labels/base/masks','Per-arm/scene gates and task aggregates'],
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'decision.json',run/'data.npz',
            run/'weights.npz',run/'loss_interface_check.json',Path(__file__),*files]},scope=p['scope']))
    print(json.dumps(dict(passed=True,models=len(final),snapshots=len(expected),predictions=validated,
        fit_and_preserve_pass=[x['condition'] for x in decisions if x['all_fit_and_preserve_gates']],scalar_fk_error=maxparity),indent=2))

if __name__=='__main__':main()

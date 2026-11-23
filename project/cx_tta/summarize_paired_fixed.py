"""S2e final-only offline errors; explicit branch boundaries, no success claims."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .paired_fixed_data import load
from .probe_candidates import read,verify
from .paths import write_json
from .audit import digest
from .robot_kinematics import RobotFK
from .candidate_diagnostic import row_metrics,CAPS
from .recovery_fit_metrics import endpoints,recovery_gate,ARMS


def group_summary(metric,action,target,valid,rows,ids,edges,baseline,group_key='group'):
    groups=[]
    for group in sorted({rows[i].get(group_key,rows[i]['group']) for i in ids}):
        ii=np.array([i for i in ids if rows[i].get(group_key,rows[i]['group'])==group]);weights=valid[ii].sum(1)
        velocity_weights=(valid[ii,1:]&valid[ii,:-1]).sum(1);arms={}
        for arm,side in enumerate(('left','right')):
            avg=lambda key:float(np.average(metric[key][ii,arm],weights=weights))
            out={key:avg(key) for key in ('angle_deg','position_m','gripper_mae','pose_under_3deg_1cm_fraction','score')}
            out.update(arm_rmse_rad=float(np.sqrt(avg('arm_mse_rad2'))),
                velocity_error_rmse_rad=None if not velocity_weights.sum() else float(np.sqrt(np.average(metric['velocity_mse_rad2'][ii,arm],weights=velocity_weights))),
                joint_limit_violations=int(metric['joint_limit_violations'][ii,arm].sum()),
                gripper_outside_margin=int(metric['gripper_outside_margin'][ii,arm].sum()),
                base_score=float(np.average(baseline['score'][ii,arm],weights=weights)))
            arms[side]=out
        selected=set(ii.tolist());seams=[]
        for e in edges:
            before,after,last=e['before'],e['after'],e['last']
            if before in selected and after in selected:
                seams.append((action[after,0,ARMS]-action[before,last,ARMS])-(target[after,0,ARMS]-target[before,last,ARMS]))
        entry=dict(group=group,task=rows[ii[0]]['task'],rows=len(ii),valid_actions=int(weights.sum()),arms=arms,
            seams=len(seams),seam_error_rmse_rad=None if not seams else float(np.sqrt(np.mean(np.asarray(seams)**2))))
        entry['recovery_fit_gate']=recovery_gate({'groups':[entry]})
        entry['preservation_gate']=all(v['angle_deg']<=1 and v['position_m']<=.01 and v['gripper_mae']<=.02 for v in arms.values())
        groups.append(entry)
    return dict(groups=groups,recovery_fit_gate=recovery_gate({'groups':groups}),
        preservation_gate=all(g['preservation_gate'] for g in groups))


def summarize(run):
    p,a,ctx,s=load(run);assert read(run/'candidate_audit.json')['passed'];result=read(run/'result.json')
    rows=p['rows'];robots=[RobotFK(read(Path(p['config'])),side) for side in ('left','right')]
    tp=endpoints(a['target'],robots);base_metric=row_metrics(a['base'],a['target'],a['valid'],robots,tp)
    train=p['train_indices'];cal=p['calibration_indices']
    parts={'new_recovery':p['new_indices'],
        'old_recovery':[i for i in train if rows[i]['kind']=='recovery' and rows[i]['source_block']=='background'],
        'expert_train':[i for i in train if rows[i]['kind']=='expert'],
        'anchor_train':[i for i in train if rows[i]['kind']=='anchor'],
        'expert_developer':[i for i in cal if rows[i]['kind']=='expert'],
        'anchor_developer':[i for i in cal if rows[i]['kind']=='anchor']}
    def summaries(action,metric):
        out={name:group_summary(metric,action,a['target'],a['valid'],rows,ids,p['seam_pairs'],base_metric) for name,ids in parts.items()}
        out['new_branches']=group_summary(metric,action,a['target'],a['valid'],rows,p['new_indices'],p['seam_pairs'],base_metric,'branch_group')
        return out
    baseline=summaries(a['base'],base_metric);reports=[]
    for record in result['reports']:
        with np.load(record['artifact']) as z:pred=z['predicted'].astype(np.float64)
        capped=a['base']+np.clip(pred-a['base'],-CAPS,CAPS)
        variants={}
        for name,action in [('raw',pred),('capped',capped)]:
            variants[name]=summaries(action,row_metrics(action,a['target'],a['valid'],robots,tp))
        reports.append(dict(condition=record['condition'],encoding=record['encoding'],seed=record['seed'],variants=variants))
        print('SUMMARIZED',record['condition'],record['encoding'],record['seed'],flush=True)
    comparisons=[]
    for group in sorted({rows[i]['group'] for i in p['new_indices'] if rows[i]['task']=='turn_switch'}):
        condition=next(k for k,v in p['omitted_groups'].items() if v==group)
        for encoding in p['encodings']:
            for seed in p['seeds']:
                pair={c:next(r for r in reports if (r['condition'],r['encoding'],r['seed'])==(c,encoding,seed)) for c in ('include_all',condition)}
                for variant in ('raw','capped'):
                    selected={label:next(g for g in r['variants'][variant]['new_recovery']['groups'] if g['group']==group)
                              for label,r in zip(('included','omitted'),pair.values())}
                    base=next(g for g in baseline['new_recovery']['groups'] if g['group']==group)
                    comparisons.append(dict(group=group,encoding=encoding,seed=seed,variant=variant,base=base,**selected))
    verify(p)
    summary=dict(status='complete',baseline=baseline,reports=reports,comparisons=comparisons,
        aggregation='Within each scene or branch, valid-action weighted; both arms reported separately. '
            'No forks counted as independent scenes. Seam errors only explicit same-branch saved edges.',
        scope=p['scope'],source_sha256={str(run/'protocol.json'):digest(run/'protocol.json'),
            str(run/'result.json'):digest(run/'result.json'),str(run/'candidate_audit.json'):digest(run/'candidate_audit.json'),
            str(Path(__file__)):digest(Path(__file__))})
    write_json(run/'summary.json',summary)
    lines=['# S2e paired scene omission (offline only)','',
        'Each cell: right-arm angle degrees / position cm / gripper MAE. Means over three optimization seeds; all seeds retained in summary.json.',
        '', '|Scene|Encoding|Variant|Base|Included|Omitted|Included gates|Omitted gates|',
        '|---|---|---|---|---|---|---|---|']
    for group in sorted({v['group'] for v in comparisons}):
        for encoding in p['encodings']:
            for variant in ('raw','capped'):
                selected=[v for v in comparisons if (v['group'],v['encoding'],v['variant'])==(group,encoding,variant)]
                cells=[]
                for label in ('base','included','omitted'):
                    values=np.mean([[v[label]['arms']['right'][k] for k in ('angle_deg','position_m','gripper_mae')] for v in selected],0)
                    cells.append(f'{values[0]:.2f} / {values[1]*100:.2f} / {values[2]:.4f}')
                gates=[sum(v[label]['recovery_fit_gate'] for v in selected) for label in ('included','omitted')]
                lines.append(f'|{group}|{encoding}|{variant}|'+ '|'.join(cells)+f'|{gates[0]}/3|{gates[1]}/3|')
    lines+=['','Gates are full-scene both-arm offline imitation criteria, NOT task success. Included scenes are fit upper bounds. '
        'Capped outputs use original arm0.1rad/gripper0.2 bounds; raw outputs are new action centers. '
        'Preservation and each branch/arm/seam are in summary.json. No live evaluation, router or TTA.']
    (run/'summary.md').write_text('\n'.join(lines)+'\n');print('SUMMARY',run/'summary.md',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    threadpool_limits(2);summarize(args.run.resolve())

"""Paired final-model recovery dependence, preservation and finite-grid diagnostics."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .paths import write_json
from .audit import digest
from .probe_candidates import read,verify
from .matched_recovery_data import load
from .robot_kinematics import RobotFK
from .candidate_diagnostic import fixed_actions,row_metrics,summarize
from .component_selector import component_grid,choose,assemble


def main(run):
    p,a,_,_=load(run)
    if not read(run/'candidate_audit.json')['passed']:raise ValueError('Audited complete candidate matrix required')
    result=read(run/'result.json');rows=p['rows'];robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')]
    base,target,valid=(a[k] for k in ('base','target','valid'))
    baseline=row_metrics(base.astype(float),target,valid,robots)
    rec=np.array([i for i,r in enumerate(rows) if r['kind']=='recovery']);rr=[rows[i] for i in rec]
    inv={v:i for i,v in enumerate(rec)};chains=[]
    for group in sorted({r['group'] for r in rr}):
        pairs=[dict(e,before=inv[e['before']],after=inv[e['after']]) for e in p['seam_pairs'] if e['group']==group]
        chains.append(dict(group=group,rows=[i for i,r in enumerate(rr) if r['group']==group],pairs=pairs))
    part={f'{role}_{kind}':[i for i in indices if rows[i]['kind']==kind]
          for role,indices in [('train',p['train_indices']),('calibration',p['calibration_indices'])]
          for kind in ('expert','anchor','recovery')}
    part={k:v for k,v in part.items() if v}
    recovery_baseline={k:v[rec] for k,v in baseline.items()}
    baseline_summary=summarize(recovery_baseline,base[rec].astype(float),target[rec],valid[rec],rr,list(range(len(rec))),recovery_baseline)
    reports=[]
    for entry in result['reports']:
        if digest(Path(entry['artifact']))!=entry['sha256']:raise ValueError('Changed prediction')
        with np.load(entry['artifact']) as z:pred=z['predicted'].copy()
        out={k:entry[k] for k in ('condition','encoding','seed')};out['parts']={};out['recovery_oracles']={}
        actions=fixed_actions(base,pred)
        for name,index in [('raw',3),('capped',6)]:
            action=actions[index];metric=row_metrics(action,target,valid,robots)
            out['parts'][name]={key:summarize(metric,action,target,valid,rows,indices,baseline) for key,indices in part.items()}
        fixed=fixed_actions(base[rec],pred[rec]);metrics=[row_metrics(x,target[rec],valid[rec],robots) for x in fixed]
        for support in ('raw','capped'):
            grid=component_grid(fixed,metrics,target[rec],valid[rec],support)
            for timing in ('local','sequence'):
                setting=dict(support=support,coupling='split',guard='tolerant',timing=timing)
                selected,records=choose(grid,setting,fixed,target[rec],chains)
                action=assemble(fixed,support,selected)
                m=row_metrics(action,target[rec],valid[rec],robots)
                out['recovery_oracles'][f'{support}_{timing}']=dict(
                    summary=summarize(m,action,target[rec],valid[rec],rr,list(range(len(rec))),recovery_baseline),
                    choices=selected.tolist(),sequence_records=records)
        reports.append(out);print('SCORED',entry['condition'],entry['encoding'],entry['seed'],flush=True)
    paired=[]
    for encoding in p['encodings']:
        for seed in p['seeds']:
            both=next(r for r in reports if (r['condition'],r['encoding'],r['seed'])==('both',encoding,seed))
            for condition in ('omit_shoe','omit_switch'):
                omitted=next(r for r in reports if (r['condition'],r['encoding'],r['seed'])==(condition,encoding,seed))
                target_group=p['omitted_groups'][condition]
                for support in ('raw','capped'):
                    for group in sorted({r['group'] for r in rr}):
                        b=next(g for g in both['parts'][support]['train_recovery']['groups'] if g['group']==group)
                        o=next(g for g in omitted['parts'][support]['train_recovery']['groups'] if g['group']==group)
                        from .recovery_fit_metrics import recovery_gate
                        pair=dict(encoding=encoding,seed=seed,condition=condition,support=support,group=group,
                            excluded_this_scene=group==target_group,both_gate=recovery_gate(dict(groups=[b])),
                            omission_gate=recovery_gate(dict(groups=[o])),arms={})
                        for side in ('left','right'):
                            pair['arms'][side]={k:dict(both=b['arms'][side][k],omission=o['arms'][side][k],
                                omission_minus_both=o['arms'][side][k]-b['arms'][side][k])
                                for k in ('angle_deg','position_m','arm_rmse_rad','gripper_mae','score')}
                        paired.append(pair)
    verify(p)
    write_json(run/'analysis.json',dict(status='complete',baseline_recovery=baseline_summary,reports=reports,paired=paired,
        source_sha256={str(f):digest(f) for f in (run/'protocol.json',run/'result.json',run/'candidate_audit.json',Path(__file__))},
        scope='Two already-developed recovery scenes. train_recovery names the fixed six-row slot partition, '
              'not actual supervised exposure: consult condition/excluded_this_scene. Original calibration is '
              'development-only. Teacher-known finite-grid choices are not a deployable router or proof of '
              'global recovery impossibility. No task success evaluation.'))
    print('ANALYSIS_COMPLETE',run,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    threadpool_limits(2);main(parser.parse_args().run.resolve())

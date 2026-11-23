"""Report all S1y conditions and separate candidate potential from selection."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .paths import write_json
from .audit import digest
from .probe_candidates import read, verify
from .probe_scene_gain import load_inputs
from .scene_gain_probe import row_weights, selected_metrics
from .candidate_diagnostic import fixed_actions, row_metrics, summarize, mix_arms
from .component_selector import component_grid, choose, assemble
from .robot_kinematics import RobotFK


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();threadpool_limits(2)
    if not read(run/'probe_audit.json')['passed']:raise ValueError('Completed probe audit required')
    p,data,context,proposals=load_inputs(run);r=read(run/'probe_result.json')
    robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')]
    conditions=[];potential=[];recovery=[]
    rows=p['rows'];val=p['parts']['validation'];w=row_weights(rows,val,'expert')
    mass=np.broadcast_to(w[None,:,None],(3,len(rows),2))
    for entry in read(run/'probe_data_manifest.json')['records']:
        with np.load(entry['file']) as z:panel={k:z[k].copy() for k in z.files}
        encoding,support=entry['encoding'],entry['support']
        oracle=np.where(panel['benefit'],panel['gain'],-np.inf);oracle[...,0]=0;oc=oracle.argmax(-1)
        for part,ids in p['parts'].items():
            for kind in ('expert','anchor','recovery'):
                weight=row_weights(rows,ids,kind)
                if weight.sum():potential.append(dict(encoding=encoding,support=support,part=part,kind=kind,
                    teacher_known=selected_metrics(panel,oc,weight)))
        for architecture in p['probe_design']['models']:
            reports=[x for x in r['reports'] if (x['encoding'],x['support'],x['architecture'])==(encoding,support,architecture)]
            components=[]
            for x in reports:
                with np.load(x['artifact']) as z:choice=z['choices']
                gain=np.take_along_axis(panel['gain'],choice[...,None],-1)[...,0]
                harm=np.take_along_axis(panel['harm'],choice[...,None],-1)[...,0]
                components.append(dict(probe_seed=x['seed'],**{name:dict(
                    gain=float(np.average(gain[...,c],weights=mass)),harm=float(np.average(harm[...,c],weights=mass)),
                    coverage=float(np.average(choice[...,c]>0,weights=mass))) for c,name in enumerate(('arm','grip'))}))
            conditions.append(dict(encoding=encoding,support=support,architecture=architecture,
                expert_gain_seeds=[x['validation_selected']['expert']['pooled']['gain'] for x in reports],
                expert_harm_seeds=[x['validation_selected']['expert']['pooled']['harm'] for x in reports],
                expert_coverage_seeds=[x['validation_selected']['expert']['pooled']['coverage'] for x in reports],
                component_selected=components,
                arm_benefit_auc=[x['validation_rank']['arm']['benefit_auc'] for x in reports],
                arm_gain_auc=[x['validation_rank']['arm']['gain_auc'] for x in reports],
                grip_benefit_auc=[x['validation_rank']['grip']['benefit_auc'] for x in reports],
                grip_gain_auc=[x['validation_rank']['grip']['gain_auc'] for x in reports],
                matched_proxies=reports[0]['validation_rank'],
                all_off_probe_seeds=[x['seed'] for x in reports if x['calibration']['gain_threshold'] is None],
                calibration_positive_options=[sum(t['expert']['gain']>0 for t in x['calibration_trials']) for x in reports],
                anchor_tolerant_harm=[x['validation_selected']['anchor']['pooled']['harm'] for x in reports],
                task_gain=[{task:value['gain'] for task,value in x['validation_selected']['expert']['per_task'].items()} for x in reports]))
    # Post-hoc description only: frozen predictions when each entire recovery scene was excluded.
    ids=np.array([i for i,row in enumerate(rows) if row['kind']=='recovery'])
    base,target,valid=[data[k][ids] for k in ('base','target','valid')]
    rr=[rows[i] for i in ids];baseline=row_metrics(base.astype(float),target,valid,robots)
    for encoding in p['encodings']:
        for seed in range(3):
            fixed=fixed_actions(base,proposals[encoding][seed,ids])
            metrics=[row_metrics(x,target,valid,robots) for x in fixed]
            for support in ('raw','capped'):
                grid=component_grid(fixed,metrics,target,valid,support)
                for guard in ('strict','tolerant'):
                    setting=dict(support=support,coupling='split',guard=guard,timing='local')
                    choices,_=choose(grid,setting,fixed,target,[])
                    action=assemble(fixed,support,choices)
                    metric=row_metrics(action,target,valid,robots)
                    summary=summarize(metric,action,target,valid,rr,list(range(len(ids))),baseline)
                    recovery.append(dict(encoding=encoding,seed=seed,support=support,rule='teacher_known_split_'+guard,
                        summary=summary))
            for fi,rule in [(0,'base'),(3,'raw_full'),(6,'capped_full')]:
                summary=summarize(metrics[fi],fixed[fi],target,valid,rr,list(range(len(ids))),baseline)
                recovery.append(dict(encoding=encoding,seed=seed,support='raw' if fi==3 else ('capped' if fi==6 else 'base'),
                    rule=rule,summary=summary))
    write_json(run/'analysis.json',dict(status='complete',conditions=conditions,potential=potential,
        recovery_excluded_scene_diagnostic=recovery,
        pooled_units='Candidate seeds and arm/component/alpha are correlated repeated observations. Only8expert validation scenes and1cup anchor scene.',
        posthoc_scope='Recovery teacher-known re-selection is descriptive after primary probe results. All models/candidates unchanged, no retuning or deployment.',
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'probe_protocol.json',run/'probe_result.json',run/'probe_audit.json',Path(__file__)]}))
    for c in conditions:
        print('CONDITION',c['encoding'],c['support'],c['architecture'],
            'arm_auc',round(np.mean(c['arm_benefit_auc']),3),'gains',[round(v,6) for v in c['expert_gain_seeds']],
            'off',c['all_off_probe_seeds'],flush=True)
    for encoding in p['encodings']:
        subset=[x for x in recovery if x['encoding']==encoding and x['support']=='raw' and x['rule']=='teacher_known_split_tolerant']
        print('EXCLUDED_RECOVERY',encoding,'tolerant_pass',sum(x['summary']['recovery_fit_gate'] for x in subset),'/3',flush=True)


if __name__=='__main__':main()

"""Post-hoc demand strata and reasons for conservative oracle rejection."""
import argparse
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import write_json
from .probe_candidates import read
from .candidate_diagnostic import SUPPORTS,material_benefit
from .summarize_time_encoding import failures


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();p=read(run/'protocol.json');r=read(run/'result.json')
    if not read(run/'audit.json')['passed']:raise ValueError('Independent audit required')
    records=[];rejections=[];failed=[];sources=[]
    for condition in sorted({x['condition'] for x in r['reports']}):
        reports=sorted([x for x in r['reports'] if x['condition']==condition],key=lambda x:x['seed']);panels=[]
        for report in reports:
            path=Path(report['panel']);sources.append(path)
            if digest(path)!=report['panel_sha256']:raise ValueError('Changed panel')
            with np.load(path) as z:panel={k:z[k].copy() for k in z.files}
            panels.append(panel)
            vi=p['variants'].index('raw_grid_arm');gi=p['variants'].index('raw_guarded_arm')
            for i in p['parts']['recovery']:
                for arm,side in enumerate(('left','right')):
                    chosen=int(panel['choices'][vi,i,arm]);guarded=int(panel['choices'][gi,i,arm])
                    if chosen==guarded:continue
                    reasons={k:dict(base=float(panel[k][0,i,arm]),chosen=float(panel[k][chosen,i,arm]))
                        for k,tol in p['guard_tolerance'].items() if panel[k][chosen,i,arm]>panel[k][0,i,arm]+tol}
                    for k in ('joint_limit_violations','gripper_outside_margin'):
                        if panel[k][chosen,i,arm]>panel[k][0,i,arm]:reasons[k]=dict(base=float(panel[k][0,i,arm]),chosen=float(panel[k][chosen,i,arm]))
                    rejections.append(dict(condition=condition,seed=report['seed'],row=i,group=p['rows'][i]['group'],
                        frame=p['rows'][i]['frame'],arm=side,unconstrained_choice=p['variants'][chosen],
                        guarded_choice=p['variants'][guarded],blocking_metrics=reasons))
            for variant in ('raw_025','raw_050','raw_100','capped_100','raw_grid_arm','raw_guarded_arm','capped_grid_arm'):
                failed.append(dict(condition=condition,seed=report['seed'],variant=variant,
                    failures=failures(report['summaries'][variant]['recovery'])))
        baseline=panels[0]['pose_under_3deg_1cm_fraction'][0]
        if any(not np.array_equal(z['pose_under_3deg_1cm_fraction'][0],baseline) for z in panels):raise ValueError('Base differs by seed')
        demand=baseline<.9  # Existing recovery endpoint criterion, NOT a fitted cutoff.
        for support,indices in SUPPORTS.items():
            flags=np.stack([np.stack([material_benefit({k:v[i] for k,v in z.items() if k!='choices'},
                {k:v[0] for k,v in z.items() if k!='choices'}) for i in indices[1:]]) for z in panels])
            stable=flags.all(0).any(0);anyseed=flags.any((0,1));full=flags[:,-1].all(0)
            for part,ids in p['parts'].items():
                for stratum,mask in [('base_pose_demand',demand),('base_pose_already_90percent',~demand)]:
                    tasks={}
                    for task in sorted({p['rows'][i]['task'] for i in ids}):
                        ii=[i for i in ids if p['rows'][i]['task']==task];mm=mask[ii]
                        tasks[task]=dict(row_arms=int(mm.sum()),stable_some_alpha=int((mm&stable[ii]).sum()),
                            stable_full=int((mm&full[ii]).sum()),no_material_benefit_any_seed=int((mm&~anyseed[ii]).sum()))
                    records.append(dict(condition=condition,support=support,part=part,stratum=stratum,
                        tasks=tasks,totals={key:sum(v[key] for v in tasks.values()) for key in next(iter(tasks.values()))}))
    write_json(run/'demand_diagnostic.json',dict(status='complete',records=records,
        recovery_guard_rejections=rejections,recovery_failures=failed,
        scope='Post-hoc descriptive stratification only: base pose demand is <90% valid endpoints within3deg/1cm, '
            'using the preexisting recovery criterion. This is not a task-failure or gripper-demand label. '
            'Counts are block-arm units, not independent episodes; no new selection or router fitting.',
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'audit.json',Path(__file__),*sources]}))
    for x in records:
        if x['part']=='inner_held' and x['support']=='raw':print(x['condition'],x['stratum'],x['totals'])

if __name__=='__main__':main()

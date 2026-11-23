"""Aggregate all fixed component/sequence rules without selecting a model."""
import argparse
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import write_json
from .probe_candidates import read
from .summarize_candidates import fractions
from .component_selector import STATES


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();p=read(run/'protocol.json');r=read(run/'result.json')
    if not read(run/'audit.json')['passed']:raise ValueError('Independent audit required')
    conditions=[]
    for condition in sorted({x['condition'] for x in r['reports']}):
        reports=sorted([x for x in r['reports'] if x['condition']==condition],key=lambda x:x['seed']);rules={}
        for rule in p['rules']:
            values={}
            for part in p['parts']:
                stats=[x['summaries'][rule][part] for x in reports]
                tasks={t:{k:None if stats[0]['tasks'][t][k] is None else float(np.mean([v['tasks'][t][k] for v in stats]))
                    for k in stats[0]['tasks'][t]} for t in stats[0]['tasks']}
                values[part]=dict(tasks=tasks,recovery_fit=[v['recovery_fit_gate'] for v in stats],
                    preservation=[v['preservation_gate'] for v in stats])
            rules[rule]=values
        panels=[]
        for report in reports:
            with np.load(report['panel']) as z:panels.append({k:z[k].copy() for k in z.files if k.endswith('_benefit')})
        stable={}
        for support in ('raw','capped'):
            common=np.stack([x[support+'_joint_benefit'] for x in panels]).all(0)
            masks=dict(joint_any=common.any(0),joint_arm_active=common[[i for i,(a,g) in enumerate(STATES) if a>0]].any(0),
                joint_grip_only=common[[i for i,(a,g) in enumerate(STATES) if a==0 and g>0]].any(0))
            for component in ('arm','grip'):
                masks[component+'_gain']=np.stack([x[support+'_'+component+'_benefit'] for x in panels]).all(0)[1:].any(0)
            stable[support]={part:{name:fractions(value,p['rows'],ids) for name,value in masks.items()} for part,ids in p['parts'].items()}
        conditions.append(dict(condition=condition,rules=rules,stable_component_benefits=stable))
    write_json(run/'analysis.json',dict(status='complete',conditions=conditions,promote=False,train_router=False,
        scope=p['scope'],source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'audit.json',Path(__file__)]}))
    for c in conditions:
        print(c['condition'],{s:{k:v['count'] for k,v in c['stable_component_benefits'][s]['inner_held'].items()} for s in ('raw','capped')},flush=True)

if __name__=='__main__':main()

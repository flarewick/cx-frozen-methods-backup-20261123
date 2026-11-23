"""Report all frozen candidate diagnostics and three-seed benefit intersections."""
import argparse
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import write_json
from .probe_candidates import read
from .candidate_diagnostic import SUPPORTS,material_benefit


def fractions(mask,rows,ids):
    groups=[]
    for group in sorted({rows[i]['group'] for i in ids}):
        ii=[i for i in ids if rows[i]['group']==group]
        groups.append(dict(group=group,task=rows[ii[0]]['task'],rows=len(ii),
            row_arms=2*len(ii),count=int(mask[ii].sum()),fraction=float(mask[ii].mean())))
    tasks={task:float(np.mean([g['fraction'] for g in groups if g['task']==task])) for task in sorted({g['task'] for g in groups})}
    return dict(count=int(mask[ids].sum()),row_arms=2*len(ids),tasks=tasks,groups=groups,
        macro_fraction=float(np.mean(list(tasks.values()))))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();p=read(run/'protocol.json');r=read(run/'result.json')
    if not read(run/'audit.json')['passed']:raise ValueError('Audited diagnostics required')
    conditions=[]
    for condition in sorted({x['condition'] for x in r['reports']}):
        selected=sorted([x for x in r['reports'] if x['condition']==condition],key=lambda x:x['seed'])
        variants={}
        for variant in p['variants']:
            parts={}
            for part in p['parts']:
                stats=[x['summaries'][variant][part] for x in selected]
                tasks={task:{key:None if stats[0]['tasks'][task][key] is None else
                    dict(mean=float(np.mean([s['tasks'][task][key] for s in stats])),
                         seeds=[s['tasks'][task][key] for s in stats])
                    for key in stats[0]['tasks'][task]} for task in stats[0]['tasks']}
                parts[part]=dict(tasks=tasks,recovery_fit_gates=[s['recovery_fit_gate'] for s in stats],
                    preservation_gates=[s['preservation_gate'] for s in stats])
            variants[variant]=parts
        panels=[]
        for x in selected:
            with np.load(x['panel']) as z:panels.append({k:z[k].copy() for k in z.files if k!='choices'})
        stability={}
        for support,indices in SUPPORTS.items():
            flags=np.stack([np.stack([material_benefit({k:v[i] for k,v in z.items()},
                {k:v[0] for k,v in z.items()}) for i in indices[1:]]) for z in panels])
            full=flags[:,-1].all(0);common=flags.all(0).any(0);any_seed=flags.any((0,1))
            masks=dict(stable_full=full,stable_some_alpha=common,stable_scaled_not_full=common&~full,
                no_material_benefit_any_seed=~any_seed,seed_or_strength_sensitive=any_seed&~common,
                some_alpha_each_seed=flags.any(1).all(0))
            stability[support]={part:{name:fractions(mask,p['rows'],ids) for name,mask in masks.items()}
                for part,ids in p['parts'].items()}
        conditions.append(dict(condition=condition,seeds=[x['seed'] for x in selected],variants=variants,stability=stability))
    write_json(run/'analysis.json',dict(status='complete',conditions=conditions,promote=False,train_router=False,
        reason='Teacher-known finite-grid choices show candidate potential only. Observable gate identifiability '
            'and independent recovery coverage are not established; no automatic model/route selection.',
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'audit.json',Path(__file__)]},
        scope=p['scope']))
    for c in conditions:
        print(c['condition'],{s:{k:c['stability'][s]['inner_held'][k]['count'] for k in
            ('stable_full','stable_some_alpha','stable_scaled_not_full','no_material_benefit_any_seed','seed_or_strength_sensitive')} for s in SUPPORTS})

if __name__=='__main__':main()

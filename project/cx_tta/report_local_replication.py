"""Report all frozen repetitions and decide the preregistered auxiliary-module gate."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import write_json
from .pulse_reliability_diagnostics import MAIN


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    index=json.loads((a.run/'result.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text())
    if index['status']!='complete' or digest(a.run/'protocol.json')!=index['protocol_sha256']:raise ValueError('Incomplete or changed protocol')
    runs={(r['variant'],r['seed']):Path(r['run']) for r in index['runs']}
    lookup={};audit=[]
    for (variant,seed),run in runs.items():
        m=json.loads((run/'manifest.json').read_text());result=json.loads((run/'result.json').read_text());reload=json.loads((run/'reload_audit.json').read_text())
        if m['seed']!=seed or m['epochs']!=80 or m['dataset_index_sha256']!=protocol['datasets'][variant]['index_sha256'] or result['status']!='complete' or reload['status']!='passed':raise ValueError('Replication protocol mismatch')
        for r in result['reports']:
            lookup[variant,seed,r['unit'],r['task'],r['entity'],r['horizon'],r['kind']]=r
        audit.append(dict(variant=variant,seed=seed,run=str(run),result_sha256=digest(run/'result.json'),reload_folds=len(reload['folds'])))
    for seed in protocol['seeds']:
        for unit in ('group','task'):
            with np.load(runs['full',seed]/f'{unit}_predictions.npz') as x,np.load(runs['robot_only',seed]/f'{unit}_predictions.npz') as y:
                if not np.array_equal(x['action_only'],y['action_only'],equal_nan=True):raise ValueError('Action control differs across representations')
    reports=[]
    for unit in ('group','task'):
        for task,entity in MAIN.items():
            for horizon in (4,8,12,16,20,24):
                for kind in ('translation_m','rotation_rad','joint_fraction'):
                    key=(unit,task,entity,horizon,kind)
                    if ('full',0,*key) not in lookup:continue
                    by_seed=[]
                    for seed in protocol['seeds']:
                        full=lookup['full',seed,*key];robot=lookup['robot_only',seed,*key]
                        if full['status']!='complete':continue
                        mse=full['mse'];r_mse=robot['mse']['conditioned']
                        by_seed.append(dict(seed=seed,zero_mse=mse['zero'],full_mse=mse['conditioned'],robot_only_mse=r_mse,action_only_mse=mse['action_only'],shuffled_mse=mse['shuffled'],full_improvement_vs_zero=1-mse['conditioned']/mse['zero'] if mse['zero']>1e-16 else None,full_improvement_vs_robot=1-mse['conditioned']/r_mse if r_mse>1e-16 else None,full_improvement_vs_action=1-mse['conditioned']/mse['action_only'] if mse['action_only']>1e-16 else None))
                    reports.append(dict(unit=unit,task=task,entity=entity,horizon=horizon,kind=kind,status='complete' if by_seed else 'unsupported_output_in_training',seeds=by_seed))
    gate_rows=[r for r in reports if r['horizon']==4 and r['kind']=='translation_m' and r['task']!='turn_switch']
    gate=all(len(r['seeds'])==3 and all(s['full_improvement_vs_zero']>0 and s['full_improvement_vs_robot']>0 and s['full_improvement_vs_action']>0 for s in r['seeds']) for r in gate_rows)
    write_json(a.run/'analysis.json',dict(status='complete',runs=audit,all_action_controls_identical=True,gate='passed_development_only' if gate else 'not_passed',reports=reports,scope='Three fixed optimization seeds; same 16 development scenes, not new physical repeats or independent validation. Every horizon retained. Gate demands positive seed-wise zero/action/robot increments on all three rigid primary objects, both scene and task holds. No inference about task success. Failure downgrades full object-effect module from mandatory prerequisite; do not expand this same experiment.'))
    print('GATE',gate)
    for r in gate_rows:
        print(r['unit'],r['task'],[(s['seed'],round(s['full_improvement_vs_zero']*100,2),round(s['full_improvement_vs_robot']*100,2),round(s['full_improvement_vs_action']*100,2)) for s in r['seeds']])
    print(a.run/'analysis.json')

if __name__=='__main__':main()

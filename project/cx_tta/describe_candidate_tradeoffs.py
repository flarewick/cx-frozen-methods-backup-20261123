"""Descriptive score/component tradeoffs; does not alter preregistered choices."""
import argparse
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import write_json
from .probe_candidates import read
from .candidate_diagnostic import SUPPORTS,material_benefit
from .summarize_candidates import fractions


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();p=read(run/'protocol.json');r=read(run/'result.json')
    if not read(run/'audit.json')['passed']:raise ValueError('Audited frozen predictions required')
    records=[];inputs=[]
    for report in r['reports']:
        file=Path(report['panel']);inputs.append(file)
        if digest(file)!=report['panel_sha256']:raise ValueError('Changed panel')
        with np.load(file) as z:panel={k:z[k].copy() for k in z.files}
        baseline={k:v[0] for k,v in panel.items() if k!='choices'};breakdown={}
        for support,pool in SUPPORTS.items():
            metric={k:v[pool[-1]] for k,v in panel.items() if k!='choices'}
            score_gain=baseline['score']-metric['score']>np.maximum(1e-8,.05*baseline['score'])
            pose_harm=(metric['angle_deg']>baseline['angle_deg']+1e-6)|(metric['position_m']>baseline['position_m']+1e-9)
            grip_harm=metric['gripper_mae']>baseline['gripper_mae']+1e-8
            benefit=material_benefit(metric,baseline)
            labels=dict(score_gain_5percent=score_gain,score_gain_with_pose_regression=score_gain&pose_harm,
                score_gain_with_gripper_regression=score_gain&grip_harm,guarded_benefit=benefit,
                guarded_benefit_with_velocity_error_increase=benefit&(metric['velocity_mse_rad2']>baseline['velocity_mse_rad2']+1e-12))
            breakdown[support]={part:{name:fractions(value,p['rows'],ids) for name,value in labels.items()}
                for part,ids in p['parts'].items()}
        choices={}
        for i,variant in enumerate(p['variants'][7:],start=7):
            support=variant.split('_',1)[0];pool=SUPPORTS[support]
            choices[variant]={part:{str(alpha):fractions(panel['choices'][i]==idx,p['rows'],ids)
                for alpha,idx in zip(p['alphas'],pool)} for part,ids in p['parts'].items()}
        records.append(dict(condition=report['condition'],seed=report['seed'],full_candidate_tradeoffs=breakdown,
            oracle_choice_frequencies=choices))
    write_json(run/'tradeoffs.json',dict(status='complete',records=records,
        scope='Post-hoc descriptive decomposition of frozen S1w results. No threshold fitting, new oracle, '
            'model selection, router training or success inference. Component counts overlap; velocity increase '
            'is target-difference error, not a claim of physical instability.',
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'audit.json',Path(__file__),*inputs]}))
    print('Tradeoff decomposition complete')

if __name__=='__main__':main()

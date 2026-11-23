"""Independent saved-prediction verification and paired seed summaries."""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit import digest
from .audit_correction_budget import CAPS, balanced
from .paths import write_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text())
    data=json.loads((a.run/'data_contract.json').read_text())
    result=json.loads((a.run/'result.json').read_text())
    if result['status']!='complete' or len(result['reports'])!=6:
        raise ValueError('Incomplete six-fit probe')
    verified=0;baseline={};reference={};hashes={}
    for report in result['reports']:
        stem=f"seed{report['seed']}_{report['mode']}"
        if digest(a.run/(stem+'.pt'))!=report['weights_sha256']:
            raise ValueError('Changed model')
        for split,ids in data['partitions'].items():
            path=a.run/f'{stem}_{split}.npz';hashes[path.name]=digest(path)
            with np.load(path,allow_pickle=False) as z:values=dict(z)
            if values['indices'].tolist()!=ids:raise ValueError('Row misalignment')
            if split in reference and any(not np.array_equal(values[k],reference[split][k]) for k in ('base','target','valid','indices')):
                raise ValueError('Unpaired labels')
            reference.setdefault(split,values)
            rows=[data['rows'][i] for i in ids]
            base,target,pred=[values[k].astype(float) for k in ('base','target','predicted')];valid=values['valid']
            if not np.isfinite(pred).all() or (np.abs(pred-base)>CAPS+1e-6).any():raise ValueError('Invalid bounded output')
            feasible=base+np.clip(target-base,-CAPS,CAPS)
            for name,x,y in [('expert_mse',pred,target),('base_expert_mse',base,target),('reachable_mse',pred,feasible),('base_reachable_mse',base,feasible),('residual_energy',pred,base)]:
                errors=((((x-y)/CAPS)**2).mean(-1)*valid).sum(-1)/valid.sum(-1)
                for task,expected in report['metrics'][split][name].items():
                    ti=[i for i,r in enumerate(rows) if r['task']==task]
                    actual=balanced(errors[ti],[rows[i] for i in ti])
                    if not np.isclose(actual,expected,rtol=2e-5,atol=1e-8):raise ValueError('Metric mismatch')
            verified+=1
    lookup={(r['seed'],r['mode']):r['metrics'] for r in result['reports']}
    summary={}
    for split in ('fit','inner_held','anchor_fit','anchor_held'):
        summary[split]={}
        metric='residual_energy' if split.startswith('anchor') else 'expert_mse'
        tasks=lookup[(0,'expert_mse')][split][metric]
        for task in tasks:
            ordinary=np.array([lookup[(s,'expert_mse')][split][metric][task] for s in protocol['seeds']])
            reachable=np.array([lookup[(s,'reachable_mse')][split][metric][task] for s in protocol['seeds']])
            row=dict(metric=metric,expert_objective=ordinary.tolist(),reachable_objective=reachable.tolist(),
                paired_relative_reduction=(1-reachable/ordinary).tolist(),
                paired_relative_reduction_mean=float(np.mean(1-reachable/ordinary)),
                better_seed_count=int((reachable<ordinary).sum()))
            if not split.startswith('anchor'):
                zero=np.array([lookup[(s,'expert_mse')][split]['base_expert_mse'][task] for s in protocol['seeds']])
                row.update(expert_improvement_vs_base=(1-ordinary/zero).tolist(),reachable_improvement_vs_base=(1-reachable/zero).tolist())
            summary[split][task]=row
    write_json(a.run/'paired_audit.json',dict(status='passed',verified_prediction_files=verified,
        predictions_sha256=hashes,summary=summary,report_code_sha256=digest(Path(__file__)),
        limits='Three training seeds on the same previously used developer split, not independent scene resampling. Anchor residual energy measures deviation, not harm rate. Paired means are descriptive; no efficacy significance claim or seed selection.'))
    print(json.dumps(dict(status='passed',verified_prediction_files=verified,summary=summary),indent=2),flush=True)


if __name__=='__main__':main()

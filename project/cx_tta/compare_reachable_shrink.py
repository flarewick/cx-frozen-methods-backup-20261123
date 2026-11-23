"""Post-hoc global attenuation control, fitted only on original training anchors."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .compare_preservation_shrink import score
from .paths import write_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    result=json.loads((a.run/'result.json').read_text())
    data=json.loads((a.run/'data_contract.json').read_text())
    by={(r['seed'],r['mode']):r for r in result['reports']};reports=[]
    for seed in (0,1,2):
        old,new=[by[(seed,m)]['metrics'] for m in ('expert_mse','reachable_mse')]
        before=np.mean(list(old['anchor_fit']['residual_energy'].values()))
        after=np.mean(list(new['anchor_fit']['residual_energy'].values()))
        gamma=float(np.sqrt(after/before));metrics={}
        for split in ('inner_held','anchor_fit','anchor_held'):
            path=a.run/f'seed{seed}_expert_mse_{split}.npz'
            with np.load(path) as z:
                prediction=z['base']+gamma*(z['predicted']-z['base'])
                metrics[split]=score(prediction,z['target'],z['valid'],z['indices'],data['rows'])
        if not np.isclose(np.mean(list(metrics['anchor_fit'].values())),after,rtol=1e-5):
            raise ValueError('Training anchor energy not matched')
        reports.append(dict(seed=seed,gamma=gamma,scaled_expert_model=metrics,
            reachable_model={s:new[s]['residual_energy' if s.startswith('anchor') else 'expert_mse'] for s in metrics}))
    write_json(a.run/'shrink_diagnostic.json',dict(status='complete',reports=reports,
        code_sha256=digest(Path(__file__)),scope='Post-hoc diagnostic after six training runs. One coefficient per seed, shared across all tasks, derived solely from task/scene-balanced fit-anchor residual energy; no held labels set gamma. No model selection. Mean held anchor energy does not measure task success.'))
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()

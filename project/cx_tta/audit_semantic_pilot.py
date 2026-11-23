"""Independent integrity and metric audit of bounded semantic/preservation experiments."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import write_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());result=json.loads((a.run/'result.json').read_text());rows=protocol['rows'];partitions=protocol['partitions']
    if result['status']!='complete':raise ValueError('Incomplete training')
    groups={s:{rows[i]['group'] for i in ids} for s,ids in partitions.items()}
    names=list(groups)
    if any(groups[s]&groups[t] for k,s in enumerate(names) for t in names[k+1:]):raise ValueError('Episode leakage')
    preprocess=[]
    for representation in ('coarse','semantic'):
        with np.load(a.run/f'{representation}_preprocess.npz') as z:
            if not np.array_equal(z['fit_indices'],partitions['fit']):raise ValueError('Preprocessing fitted on nontrain rows')
            for i,j in enumerate(z['history_indices']):
                if j>=0 and not (rows[j]['group']==rows[i]['group'] and rows[j]['pilot_split']==rows[i]['pilot_split'] and rows[j]['frame']<rows[i]['frame']):raise ValueError('History leakage')
        preprocess.append(dict(representation=representation,status='passed'))
    caps=np.array([.1]*6+[.2]+[.1]*6+[.2]);checks=[]
    for report in result['reports']:
        variant=report['variant']
        for split in ('inner_held','anchor_fit','anchor_held'):
            with np.load(a.run/f'{variant}_{split}.npz') as z:
                pred,base,target,valid,ids=z['predicted'],z['base'],z['target'],z['valid'],z['indices']
            if not np.array_equal(ids,partitions[split]):raise ValueError('Wrong saved order')
            if split.startswith('anchor') and not np.array_equal(base,target):raise ValueError('Anchor has nonzero correction target')
            err=((((pred-target)/caps)**2).mean(-1)*valid).sum(-1)/valid.sum(-1)
            zero=((((base-target)/caps)**2).mean(-1)*valid).sum(-1)/valid.sum(-1)
            records=[rows[i] for i in ids];metrics={}
            for task in sorted({r['task'] for r in records}):
                scene_metrics=[]
                for group in sorted({r['group'] for r in records if r['task']==task}):
                    take=np.array([r['group']==group for r in records]);scene_metrics.append(float(err[take].mean()))
                    if np.max(np.abs(pred[take]-base[take])-caps)>1e-6:raise ValueError('Residual exceeds shared cap')
                metrics[task]=float(np.mean(scene_metrics))
                if not np.isclose(metrics[task],report['metrics'][split][task],rtol=2e-6,atol=1e-9):raise ValueError('Saved metrics do not recompute')
            checks.append(dict(variant=variant,split=split,metrics=metrics,rows=len(ids),scenes=len({r['group'] for r in records}),prediction_mse_recomputed=True))
    by={r['variant']:r for r in result['reports']};comparisons=[]
    for candidate,reference in [('semantic_bc','coarse_bc'),('semantic_direction','semantic_bc'),('semantic_bc_preserve','semantic_bc'),('semantic_direction_preserve','semantic_direction'),('semantic_direction_preserve','semantic_bc_preserve')]:
        pair={}
        for split in ('inner_held','anchor_held'):
            left,right=by[candidate]['metrics'][split],by[reference]['metrics'][split]
            pair[split]={t:1-left[t]/right[t] if right[t]>1e-16 else None for t in left}
        comparisons.append(dict(candidate=candidate,reference=reference,relative_error_reduction=pair))
    write_json(a.run/'audit.json',dict(status='passed',split_scene_counts={s:len(v) for s,v in groups.items()},preprocess=preprocess,predictions=checks,comparisons=comparisons,result_sha256=digest(a.run/'result.json'),scope='Independent saved-prediction metric, cap, source-order, PCA and causal-history audit. Positive anchor reduction is less deviation from L0, NOT improved task success. Only cup and shoe have held success scenes.'))
    print('PASSED',json.dumps(comparisons,indent=2),flush=True)

if __name__=='__main__':main()

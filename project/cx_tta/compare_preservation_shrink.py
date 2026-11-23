"""Training-anchor-only global shrink control; no held-label threshold selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from .paths import write_json


def score(pred,target,valid,ids,rows):
    caps=np.array([.1]*6+[.2]+[.1]*6+[.2]);err=((((pred-target)/caps)**2).mean(-1)*valid).sum(-1)/valid.sum(-1)
    records=[rows[i] for i in ids];out={}
    for task in sorted({r['task'] for r in records}):
        groups=sorted({r['group'] for r in records if r['task']==task})
        out[task]=float(np.mean([err[[r['group']==g for r in records]].mean() for g in groups]))
    return out


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());result=json.loads((a.run/'result.json').read_text());by={r['variant']:r for r in result['reports']};reports=[]
    for reference in ('semantic_bc','semantic_direction'):
        candidate=reference+'_preserve';ref=np.mean(list(by[reference]['metrics']['anchor_fit'].values()));pres=np.mean(list(by[candidate]['metrics']['anchor_fit'].values()))
        factor=float(np.clip(np.sqrt(pres/max(ref,1e-16)),0,1));metrics={}
        for split in ('inner_held','anchor_fit','anchor_held'):
            with np.load(a.run/f'{reference}_{split}.npz') as z:
                predicted=z['base']+factor*(z['predicted']-z['base']);metrics[split]=score(predicted,z['target'],z['valid'],z['indices'],protocol['rows'])
        reports.append(dict(reference=reference,preservation=candidate,global_factor=factor,threshold_fit_split='anchor_fit_only',shrink_metrics=metrics,preservation_metrics=by[candidate]['metrics'],scope='Single global factor matched to task/scene-balanced fit-anchor residual energy, clipped [0,1]; no held outcomes used to set factor. Preservation must outperform this tradeoff to support context-specific protection.'))
    write_json(a.run/'shrink_control.json',dict(status='complete',reports=reports));print(json.dumps(reports,indent=2))

if __name__=='__main__':main()

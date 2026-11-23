"""Reload shared checkpoints, check fold separation and saved predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .audit import digest
from .effect_learning_probe import weights
from .shared_effect_model import SharedEffect,fold_indices
from .paths import write_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    torch.set_num_threads(2)
    manifest=json.loads((run/'manifest.json').read_text());dataset=Path(manifest['dataset']);index=json.loads((dataset/'index.json').read_text());records=index['records'];result=json.loads((run/'result.json').read_text())
    if result['status']!='complete' or digest(dataset/'index.json')!=manifest['dataset_index_sha256']:raise ValueError('Changed/incomplete data')
    with np.load(dataset/'inputs_oracle.npz') as z:vision=z['vision'];geometry=z['geometry'];action=z['action']
    with np.load(dataset/'labels_offline.npz') as z:y=z['target'];mask=z['mask']
    visible=np.array([r['visible_supervision_only'] for r in records]);reports=[]
    for unit in ('group','task'):
        with np.load(run/(unit+'_predictions.npz')) as z:saved=z['conditioned']
        for fold,held in enumerate(sorted({r[unit] for r in records})):
            train,test=fold_indices(records,held,unit);tag=f'{unit}_{fold:02d}'
            with np.load(run/(tag+'_preprocess.npz')) as z:
                pre=dict(z)
            if not np.array_equal(pre['fit_rows'],train) or not set(pre['pca_fit_rows']).issubset(train):raise ValueError('Normalizer leakage')
            v=vision@pre['pca_components'].T-pre['pca_mean']@pre['pca_components'].T
            context=((np.c_[v,geometry]-pre['context_mean'])/pre['context_std']).astype(np.float32);a=(action/pre['action_scale']).astype(np.float32)
            checkpoint=torch.load(run/(tag+'_conditioned.pt'),map_location='cpu',weights_only=True)
            model=SharedEffect(checkpoint['context_dim'],checkpoint['action_dim']);model.load_state_dict(checkpoint['state_dict']);model.eval()
            with torch.no_grad():
                pred=model(torch.from_numpy(context),torch.from_numpy(a)).numpy();zero=model(torch.from_numpy(context[test]),torch.zeros_like(torch.from_numpy(a[test]))).numpy()
            support=np.array(checkpoint['supported']);expected=(mask[train]&visible[train,None]).any(0)
            if not np.array_equal(support,expected) or np.any(zero!=0):raise ValueError('Support/zero-action contract changed')
            diff=float(np.max(np.abs(pred[np.ix_(test,support)]-saved[np.ix_(test,support)])))
            if diff>1e-5:raise ValueError('Reload prediction mismatch')
            scores={}
            for split,ids in [('fit',train),('held',test)]:
                w=weights([records[i] for i in ids])[:,None];valid=mask[ids]&visible[ids,None]&support;den=(w*valid).sum(0)
                zero_mse=(w*valid*y[ids]**2).sum(0)/np.maximum(den,1e-12);mse=(w*valid*(pred[ids]-y[ids])**2).sum(0)/np.maximum(den,1e-12)
                scores[split]=dict(zero_mse=zero_mse.tolist(),mse=mse.tolist(),supported= support.tolist())
            reports.append(dict(unit=unit,held=held,reload_max_abs_error=diff,scores=scores))
    write_json(run/'reload_audit.json',dict(status='passed',folds=reports,scope='Conditioned checkpoint restoration, zero-input anchoring and fold-only preprocessing; no deployment validation'))
    print('PASSED',len(reports),'folds',flush=True)

if __name__=='__main__':main()

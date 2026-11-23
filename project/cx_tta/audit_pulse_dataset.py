"""Check real pulse dataset grouping, common inputs and separated supervision."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import output_path,write_json


def audit(dataset):
    dataset=output_path(dataset);index=json.loads((dataset/'index.json').read_text());labels=json.loads((dataset/'labels_offline.json').read_text())
    if digest(dataset/'inputs_observable.npz')!=index['inputs_sha256'] or digest(dataset/'labels_offline.json')!=index['labels_sha256']:raise ValueError('Dataset changed')
    split_path=Path(index['split_plan'])
    if digest(split_path)!=index['split_plan_sha256']:raise ValueError('Frozen split changed')
    split={(r['task'],r['seed']):r['split'] for r in json.loads(split_path.read_text())['scenes']}
    with np.load(dataset/'inputs_observable.npz') as z:inputs=dict(z)
    if set(inputs)!={'context','planned_base_fk','planned_residual_fk'}:raise ValueError('Unexpected input keys')
    rows=index['records'];n=len(rows)
    if len(labels)!=n or any(len(x)!=n or not np.isfinite(x).all() for x in inputs.values()):raise ValueError('Shape or finite check failed')
    if any(inputs[k].shape!=(n,448) for k in ('planned_base_fk','planned_residual_fk')):raise ValueError('Wrong planned action shape')
    groups={};baselines=0
    for i,(row,label) in enumerate(zip(rows,labels)):
        if split[row['task'],row['seed']]!=row['split']:raise ValueError('Scene split differs from frozen plan')
        key=(row['source_index'],row['horizon'])
        if key in groups:
            j=groups[key]
            if not np.array_equal(inputs['context'][i],inputs['context'][j]) or not np.array_equal(inputs['planned_base_fk'][i],inputs['planned_base_fk'][j]):raise ValueError('Candidates have different observable context or base plan')
        else:groups[key]=i
        if row['branch']=='base':
            baselines+=1
            if np.any(inputs['planned_residual_fk'][i]!=0):raise ValueError('Nonzero base residual')
            for entity in label['counterfactual']['rigid'].values():
                if np.max(np.abs(np.r_[entity['translation_world_m'],entity['rotation_world_rad']]))>1e-12:raise ValueError('Nonzero base target')
        for k in ('planned_base_fk','planned_residual_fk'):
            actions=inputs[k][i].reshape(32,14)
            if np.any(actions[:row['start']]!=0) or np.any(actions[row['stop']:]!=0):raise ValueError('Unmasked horizon')
    for source in index['sources']:
        run=Path(source['run'])
        if digest(run/'manifest.json')!=source['manifest_sha256'] or digest(run/'result.json')!=source['result_sha256']:raise ValueError('Source changed')
        for prefix in ('current','history'):
            path=source[prefix+'_observed_file']
            if path is not None and digest(Path(path))!=source[prefix+'_observed_sha256']:raise ValueError('Observable frame changed')
    return dict(status='passed',rows=n,independent_scenes=len({r['group'] for r in rows}),decision_points=len(index['sources']),base_rows=baselines,valid_full_horizon_rows=sum(l['full_horizon_valid_offline_only'] for l in labels),scope='Interface and grouping audit only; no generalization or trained model claim')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);args=p.parse_args();result=audit(args.dataset);write_json(args.dataset/'interface_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()

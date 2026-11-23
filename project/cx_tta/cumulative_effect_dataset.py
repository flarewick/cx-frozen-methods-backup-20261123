"""Common-origin cumulative horizons; never compare diverged local start states."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .object_effect_labels import entity_effects
from .paths import new_run,write_json
from .short_window_dataset import SCALE


def planned_prefix(actions,horizon):
    actions=np.asarray(actions)
    if actions.shape!=(32,14) or horizon not in (8,16,24,32):raise ValueError('Invalid horizon/actions')
    output=np.zeros_like(actions);output[:horizon]=actions[:horizon]
    return output


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--horizon',type=int,choices=(8,16,24,32),required=True);args=p.parse_args()
    source=args.dataset;index=json.loads((source/'index.json').read_text())
    if digest(source/'inputs.npz')!=index['inputs_sha256']:raise ValueError('Input cache changed')
    with np.load(source/'inputs.npz') as z:contexts=z['context']
    source_rows={(r['job'],r['branch']):i for i,r in enumerate(index['records']) if r['start']==0}
    run=new_run('cumulative',group=('effect_learning',));print(run,flush=True)
    rows=[];inputs=[];actions=[];labels=[];excluded=[]
    for entry in index['sources']:
        path=Path(entry['run'])
        if digest(path/'result.json')!=entry['result_sha256']:raise ValueError('Source changed')
        result=json.loads((path/'result.json').read_text());caps=np.array(json.loads((path/'proposal_spec.json').read_text())['caps'])
        baseframes={f['offset']:f for f in result['branches']['base']['frames']}
        with np.load(path/'checkpoint.npz') as z:base=z['base_actions']
        for name,b in result['branches'].items():
            i=source_rows[entry['id'],name];row=dict(index['records'][i],stop=args.horizon);frames={f['offset']:f for f in b['frames']}
            with np.load(path/f'transition_{name}.npz') as z:
                if not z['executed_mask'][:args.horizon].all():excluded.append(row);continue
                residual=z['residual_actions'];duration=float(z['physics_dt_s'][:args.horizon].sum())
            if duration<=0:excluded.append(row);continue
            before=frames[0];after=frames[args.horizon];reference=baseframes[args.horizon]
            # Original context ends in the first 8 planned base actions. Replace with the full padded prefix.
            inputs.append(np.r_[contexts[i,:-112],(planned_prefix(base,args.horizon)/SCALE).ravel()])
            actions.append((planned_prefix(residual,args.horizon)/caps).ravel());rows.append(row)
            labels.append(dict(counterfactual=entity_effects(reference['physical_offline_only']['offline_only'],after['physical_offline_only']['offline_only'],b['articulation_limits_offline_only']),visibility_before=before['visibility_offline_only'],visibility_after=after['visibility_offline_only'],base_visibility_after=reference['visibility_offline_only'],duration_s=duration))
    np.savez_compressed(run/'inputs.npz',context=np.array(inputs,np.float32),action=np.array(actions,np.float32))
    write_json(run/'labels_offline.json',labels)
    write_json(run/'index.json',dict(status='complete',records=rows,excluded=excluded,sources=index['sources'],inputs_sha256=digest(run/'inputs.npz'),labels_sha256=digest(run/'labels_offline.json'),source_index_sha256=digest(source/'index.json'),horizon=args.horizon,scope='Post-hoc train-only cumulative same-origin diagnostic; not local counterfactual after branches diverge'))
    print(f'COMPLETE horizon={args.horizon} rows={len(rows)} output={run}',flush=True)

if __name__=='__main__':main()

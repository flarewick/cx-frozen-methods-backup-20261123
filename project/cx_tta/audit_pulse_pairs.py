"""Offline +/- response decomposition; diagnose, do not assume physical odd symmetry."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from .paths import new_run,write_json
from .audit import digest


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    index=json.loads((a.dataset/'index.json').read_text());labels=json.loads((a.dataset/'labels_offline.json').read_text())
    if digest(a.dataset/'inputs_observable.npz')!=index['inputs_sha256'] or digest(a.dataset/'labels_offline.json')!=index['labels_sha256']:raise ValueError('Changed dataset')
    with np.load(a.dataset/'inputs_observable.npz') as z:actions=z['planned_residual_fk']
    lookup={(r['source_index'],r['horizon'],r['branch']):i for i,r in enumerate(index['records'])};groups=defaultdict(list);asym=[]
    for i,row in enumerate(index['records']):
        if not row['branch'].endswith('_plus'):continue
        j=lookup[row['source_index'],row['horizon'],row['branch'].removesuffix('_plus')+'_minus'];plus,minus=labels[i],labels[j]
        if not plus['full_horizon_valid_offline_only'] or not minus['full_horizon_valid_offline_only']:continue
        asym.append(float(np.linalg.norm(actions[i]+actions[j])/max(np.linalg.norm(actions[i])+np.linalg.norm(actions[j]),1e-12)))
        for family,entities in plus['counterfactual'].items():
            for entity,value in entities.items():
                other=minus['counterfactual'][family][entity]
                kinds=[('translation_m','translation_world_m',.005),('rotation_rad','rotation_world_rad',.02)] if family=='rigid' else [('joint_fraction','delta_range_fraction',.005)]
                for kind,key,threshold in kinds:
                    yp,ym=np.asarray(value[key]),np.asarray(other[key])
                    if family=='articulated':
                        valid=np.asarray(value['range_valid_mask'])&np.asarray(other['range_valid_mask']);yp,ym=yp[valid],ym[valid]
                    if not len(yp):continue
                    odd=(yp-ym)/2;even=(yp+ym)/2
                    groups[row['task'],entity,kind,row['horizon']].append(dict(odd=float(odd@odd),even=float(even@even),event=bool(max(np.linalg.norm(yp),np.linalg.norm(ym))>threshold),group=row['group']))
    reports=[]
    for (task,entity,kind,h),rows in sorted(groups.items()):
        odd=sum(r['odd'] for r in rows);even=sum(r['even'] for r in rows)
        reports.append(dict(task=task,entity=entity,kind=kind,horizon=h,pairs=len(rows),event_pairs=sum(r['event'] for r in rows),event_scenes=len({r['group'] for r in rows if r['event']}),odd_energy_fraction=odd/(odd+even) if odd+even>1e-16 else None))
    run=new_run('pulse_pair_audit',group=('reports','audits'));write_json(run/'result.json',dict(status='complete',dataset=str(a.dataset),input_antisymmetry_error=dict(median=float(np.median(asym)),max=float(np.max(asym))),reports=reports,scope='Algebraic odd/even decomposition of paired branch-minus-base outcome vectors. Near-opposite input does not imply opposite physical response under unilateral contact. No filtering, training or policy correction.'))
    print(run/'result.json')
    print('input_antisymmetry_error',np.median(asym),np.max(asym))
    for r in reports:
        if r['horizon']==24 and r['event_pairs']:print(r)

if __name__=='__main__':main()

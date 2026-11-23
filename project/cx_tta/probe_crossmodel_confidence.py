"""Post-hoc robustness: two models score a third model's nonconsensus benefit."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .paths import write_json
from .audit import digest
from .probe_candidates import read,verify,load_candidate
from .observable_candidate_scores import rank_stats


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();run=args.run.resolve()
    p=read(run/'protocol.json');r=read(run/'result.json');threadpool_limits(2)
    protocol=run/'crossmodel_protocol.json'
    if args.prepare_only:
        if not read(run/'observable_audit.json')['passed']:raise ValueError('Primary observable check must finish first')
        files=[run/x for x in ('protocol.json','result.json','audit.json','observable_audit.json','observable_result.json','observable_scores.npz')]
        files.extend(Path(x['panel']) for x in r['reports']);files.append(Path(__file__))
        write_json(protocol,dict(status='frozen_before_crossmodel_check',source_sha256={str(f):digest(f) for f in files},
            design='All4 conditions, raw/capped arm/grip labels, all3 held model seeds. Labels are positive component '
                'gain at ANY fixed positive alpha for only the held model, not three-seed consensus. '
                'Scores use only the OTHER TWO initialization models. No held model commands or labels in score.',
            scores=['relative_agreement','low_disagreement','high_correction_magnitude','near_fit_context'],
            amplitude_control='Positive amplitude is an explicitly post-hoc control for whether relative agreement '
                'mainly measures correction need. It does NOT replace or reorient primary frozen scores.',
            metrics='Same scene/task-balanced AUC and top10% rank description, report each held model plus mean over3 '
                'models; no pooling seeds into independent samples, no model/threshold selection or training.',
            scope='Exploratory robustness check after primary result. Same12 reused developer scenes and same training '
                'set across candidate models; excludes direct consensus-label self-coupling, but is not new-scene '
                'validation, calibrated deployment, scene cross-fitting or task-success evidence.'))
        print(protocol);return
    cp=read(protocol);verify(cp);verify(p)
    with np.load(p['data']) as z:data={k:z[k].copy() for k in z.files}
    with np.load(run/'observable_scores.npz') as z:
        saved={k:z[k].copy() for k in z.files}
    ids=saved['held_indices'];weights=saved['held_weights'];tasks=np.array([p['rows'][i]['task'] for i in ids]).repeat(2)
    records=[]
    for condition in sorted({x['condition'] for x in p['models']}):
        models=sorted([x for x in p['models'] if x['condition']==condition],key=lambda x:x['seed'])
        reports=sorted([x for x in r['reports'] if x['condition']==condition],key=lambda x:x['seed'])
        proposals=np.stack([load_candidate(x,p,data) for x in models]).astype(float)
        for held in range(3):
            others=[i for i in range(3) if i!=held]
            residual=(proposals[others]-data['base'][None]).reshape(2,len(data['base']),32,2,7)
            with np.load(reports[held]['panel']) as z:labels={k:z[k].copy() for k in z.files if k.endswith('_benefit')}
            for component,delta in [('arm',residual[...,:6]/.1),('grip',residual[...,6:]/.2)]:
                disagreement=np.sqrt(np.var(delta,axis=0).mean(axis=(1,3)))
                amplitude=np.sqrt((delta.mean(0)**2).mean(axis=(1,3)))
                scores=dict(relative_agreement=-disagreement/(amplitude+1e-6),low_disagreement=-disagreement,
                    high_correction_magnitude=amplitude,near_fit_context=saved[f'{condition}_{component}_near_fit_context'])
                for support in ('raw','capped'):
                    y=labels[support+'_'+component+'_benefit'][1:].any(0)[ids].ravel()
                    for name,values in scores.items():
                        score=values[ids].ravel()
                        records.append(dict(condition=condition,held_seed=models[held]['seed'],scoring_seeds=[models[i]['seed'] for i in others],
                            component=component,support=support,score=name,inner_held=rank_stats(score,y,weights),
                            per_task={t:rank_stats(score[tasks==t],y[tasks==t],weights[tasks==t]) for t in sorted(set(tasks))}))
        print('CROSSMODEL',condition,flush=True)
    aggregate=[]
    for condition in sorted({x['condition'] for x in records}):
        for component in ('arm','grip'):
            for support in ('raw','capped'):
                for score in cp['scores']:
                    rows=[x for x in records if (x['condition'],x['component'],x['support'],x['score'])==(condition,component,support,score)]
                    aggregate.append(dict(condition=condition,component=component,support=support,score=score,
                        auc_seeds=[x['inner_held']['auc'] for x in rows],auc_mean=float(np.mean([x['inner_held']['auc'] for x in rows])),
                        top10_precision_mean=float(np.mean([x['inner_held']['top10_weighted_precision'] for x in rows])),
                        weighted_prevalence_mean=float(np.mean([x['inner_held']['label_prevalence'] for x in rows]))))
    verify(cp);verify(p)
    write_json(run/'crossmodel_result.json',dict(status='complete',reports=records,aggregate=aggregate,
        source_sha256={str(protocol):digest(protocol)},scope=cp['scope'],no_models_or_thresholds_fitted=True))

if __name__=='__main__':main()

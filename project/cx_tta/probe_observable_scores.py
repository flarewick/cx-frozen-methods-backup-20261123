"""No-fit observable ranking check on reused, candidate-untrained developer scenes."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .audit import digest
from .paths import write_json
from .probe_candidates import read,verify,load_candidate
from .observable_candidate_scores import scores,weighted_auc,rank_stats


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();threadpool_limits(2)
    p=read(run/'protocol.json');r=read(run/'result.json');verify(p)
    if not read(run/'audit.json')['passed']:raise ValueError('Audited component results required first')
    with np.load(p['data']) as z:data={k:z[k].copy() for k in z.files}
    # Verify the frozen observable-context construction and original fit-only PCA/scaler split.
    with np.load(p['feature_preprocess']) as z:
        if sorted(z['fit_indices'].tolist())!=p['parts']['expert_fit']:raise ValueError('Preprocessing used held rows')
        history=z['history_indices'].copy()
    old=read(Path(p['expert_rows_protocol']))['rows']
    for i,j in enumerate(history):
        if j>=0 and (j>=i or old[j]['group']!=old[i]['group'] or old[j]['frame']>=old[i]['frame']):
            raise ValueError('Noncausal expert/anchor history')
    ca=read(Path(p['recovery_context_audit']));rp=read(Path(ca['source'])/'protocol.json')
    if not ca['passed'] or ca['schema']!='recovery_context_v2_state_div_pi':raise ValueError('Invalid recovery context')
    for i,j in enumerate(rp['history_indices']):
        if j>=i or j< -1:raise ValueError('Future recovery context')
        if j>=0 and (ca['rows'][j]['group']!=ca['rows'][i]['group'] or ca['rows'][j]['frame']>=ca['rows'][i]['frame']):
            raise ValueError('Noncausal recovery grouping')
    ids=np.array(p['parts']['inner_held']);rows=[p['rows'][i] for i in ids]
    groups=np.array([x['group'] for x in rows]).repeat(2);tasks=np.array([x['task'] for x in rows]).repeat(2)
    unique_tasks=sorted(set(tasks));weights=np.empty(len(groups))
    for task in unique_tasks:
        taskgroups=sorted(set(groups[tasks==task]))
        for group in taskgroups:weights[groups==group]=1/(len(unique_tasks)*len(taskgroups)*sum(groups==group))
    # Fixed scene-resampling draws are shared across all models/proxies/supports.
    rng=np.random.default_rng(621047);boot=[]
    for _ in range(1000):
        w=np.zeros_like(weights)
        for task in unique_tasks:
            gs=np.array(sorted(set(groups[tasks==task])))
            for g in rng.choice(gs,len(gs),replace=True):w[groups==g]+=weights[groups==g]
        boot.append(w)
    reports=[];saved={};auc_parity=0.
    for condition in sorted({x['condition'] for x in p['models']}):
        models=sorted([x for x in p['models'] if x['condition']==condition],key=lambda x:x['seed'])
        selected=sorted([x for x in r['reports'] if x['condition']==condition],key=lambda x:x['seed'])
        proposals=np.stack([load_candidate(x,p,data) for x in models]);panels=[]
        for x in selected:
            with np.load(x['panel']) as z:panels.append({k:z[k].copy() for k in z.files if k.endswith('_benefit')})
        proxy=scores(data['semantic'],data['base'],proposals,data['semantic'][p['parts']['expert_fit']])
        for component,values in proxy.items():
            for key,value in values.items():saved[f'{condition}_{component}_{key}']=value
        for support in ('raw','capped'):
            for component in ('arm','grip'):
                label=np.stack([z[support+'_'+component+'_benefit'] for z in panels]).all(0)[1:].any(0)
                if label[p['parts']['anchor_held']].any():raise ValueError('Base-target anchors cannot have positive component gain')
                saved[f'{condition}_{support}_{component}_label']=label
                y=label[ids].ravel()
                for name,value in proxy[component].items():
                    score=value[ids].ravel();stats=rank_stats(score,y,weights)
                    # Independent pairwise weighted ROC-AUC including half credit for ties.
                    if y.any() and (~y).any():
                        diff=score[y,None]-score[None,~y];ww=weights[y,None]*weights[None,~y]
                        brute=float(np.sum(((diff>0)+.5*(diff==0))*ww)/ww.sum())
                        auc_parity=max(auc_parity,abs(brute-stats['auc']))
                        if abs(brute-stats['auc'])>1e-12:raise ValueError('Weighted AUC formula mismatch')
                    intervals=[weighted_auc(score,y,w) for w in boot];usable=[v for v in intervals if v is not None]
                    per_task={t:rank_stats(score[tasks==t],y[tasks==t],weights[tasks==t]) for t in unique_tasks}
                    reports.append(dict(condition=condition,support=support,component=component,score=name,
                        inner_held=stats,per_task=per_task,bootstrap_auc95=None if not usable else np.quantile(usable,[.025,.975]).tolist(),
                        bootstrap_valid_replicates=len(usable),
                        successful_anchor_score_ranges={t:[float(value[[i for i in p['parts']['anchor_held'] if p['rows'][i]['task']==t]].min()),
                            float(value[[i for i in p['parts']['anchor_held'] if p['rows'][i]['task']==t]].max())]
                            for t in sorted({p['rows'][i]['task'] for i in p['parts']['anchor_held']})}))
        print('OBSERVABLE_PROXIES',condition,flush=True)
    file=run/'observable_scores.npz';np.savez_compressed(file,**saved,held_indices=ids,held_weights=weights)
    verify(p)
    write_json(run/'observable_audit.json',dict(passed=True,weighted_auc_independent_max_error=auc_parity,
        fit_preprocessing_rows=len(p['parts']['expert_fit']),expert_history_checks=len(history),recovery_history_checks=len(rp['history_indices']),
        input_contract='scores(context,base,three_candidates,fit_reference_context) only; no labels/masks/metadata. '
            'No probe training or target-based calibration; all32 observable predictions used.',
        score_file_sha256=digest(file),source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'audit.json',Path(__file__)]}))
    write_json(run/'observable_result.json',dict(status='complete',reports=reports,promote=False,train_router=False,
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'audit.json',run/'observable_audit.json',file,Path(__file__)]},
        scope=p['observable_probe'],limitations='Only four prespecified scalar proxies; not a learned-representation '
            'identifiability test. Developer positives are component geometry/action gains, not recovery/success labels. '
            'No held labels used in training. No selected winning proxy, deployment cutoff or fitted router.'))

if __name__=='__main__':main()

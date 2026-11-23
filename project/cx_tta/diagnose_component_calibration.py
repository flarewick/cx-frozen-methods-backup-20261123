"""S1z: frozen-head ranking and component calibration on development scenes only."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
from .paths import ROOT, new_run, write_json
from .audit import digest
from .probe_candidates import read
from .scene_gain_probe import row_weights, selected_metrics, calibrate


SCORES = {'gain': (0, (0., .01, .05, .1, .2)),
          'benefit': (1, (.5, .6, .7, .8, .9))}
RISKS = (.05, .1, .2)


def options(prediction, score):
    head, thresholds = SCORES[score]
    result = [(None, None, np.zeros(prediction.shape[:-2], np.int16))]
    for threshold in thresholds:
        for risk in RISKS:
            allowed = (prediction[..., head] > threshold) & (prediction[..., 2] <= risk)
            choice = np.where(allowed, prediction[..., head], -np.inf).argmax(-1) + 1
            choice[~allowed.any(-1)] = 0
            result.append((threshold, risk, choice.astype(np.int16)))
    return result


def metrics(panel, choices, rows, ids):
    w = row_weights(rows, ids, 'expert')
    return dict(pooled=selected_metrics(panel, choices, w),
        components={name:selected_metrics({k:v[:,:,:,c:c+1] for k,v in panel.items()},
                                           choices[:,:,:,c:c+1], w)
                    for c,name in enumerate(('arm','grip'))},
        per_task={task:selected_metrics(panel, choices, row_weights(rows,
            [i for i in ids if rows[i]['task']==task], 'expert'))
            for task in sorted({rows[i]['task'] for i in ids if rows[i]['kind']=='expert'})})


def fit_options(opts, panel, rows, ids, mode):
    """No held labels participate. Same component risk guards in shared/split arms."""
    w = row_weights(rows, ids, 'expert')
    aw = row_weights(rows, ids, 'anchor')
    stats = []
    for threshold,risk,choice in opts:
        parts = []
        chosen_harm = np.take_along_axis(panel['harm'], choice[...,None], -1)[...,0]
        for c in range(2):
            m = selected_metrics({k:v[:,:,:,c:c+1] for k,v in panel.items()}, choice[:,:,:,c:c+1], w)
            parts.append(dict(**m, eligible=m['harm'] <= .05 and not chosen_harm[:,aw>0,:,c].any()))
        stats.append(parts)
    def key(j, component=None):
        items = stats[j] if component is None else [stats[j][component]]
        t,r,_ = opts[j]
        return (float(np.mean([x['gain'] for x in items])),
                -float(np.mean([x['coverage'] for x in items])),
                0. if t is None else t, 0. if r is None else -r)
    if mode == 'shared':
        j=max((j for j,s in enumerate(stats) if all(x['eligible'] for x in s)),key=key)
        selected=[j,j]
    elif mode == 'split':
        selected=[max((j for j,s in enumerate(stats) if s[c]['eligible']),key=lambda j:key(j,c)) for c in range(2)]
    elif mode == 'pooled_reference':
        eligible=[j for j,s in enumerate(stats) if np.mean([x['harm'] for x in s])<=.05 and
                  not np.take_along_axis(panel['harm'], opts[j][2][...,None], -1)[:,aw>0].any()]
        j=max(eligible,key=key);selected=[j,j]
    else:
        raise ValueError(mode)
    choice=np.stack([opts[j][2][:,:,:,c] for c,j in enumerate(selected)],-1)
    return choice, [dict(option=j,threshold=opts[j][0],risk=opts[j][1]) for j in selected], stats


def ranking(prediction, panel, rows, ids):
    """Pairwise ordering within the SAME state, seed, side and component."""
    w=row_weights(rows,ids,'expert')
    out={}
    for c,name in enumerate(('arm','grip')):
        truth=panel['gain'][:,:,:,c,1:]
        mass=np.broadcast_to(w[None,:,None],truth.shape[:-1])
        differences=np.stack([truth[...,j]-truth[...,i] for i,j in combinations(range(3),2)],-1)
        weights=np.broadcast_to(mass[...,None],differences.shape)
        unequal=np.abs(differences)>1e-8
        entries={}
        for score,(head,_) in SCORES.items():
            values=prediction[:,:,:,c,:,head]
            delta=np.stack([values[...,j]-values[...,i] for i,j in combinations(range(3),2)],-1)
            concordant=np.where(delta==0,.5,(delta*differences>0).astype(float))
            denom=weights[unequal].sum()
            best=values.argmax(-1)
            chosen=np.take_along_axis(truth,best[...,None],-1)[...,0]
            entries[score]=dict(pair_accuracy=None if not denom else float(np.sum(concordant[unequal]*weights[unequal])/denom),
                always_active_gain=float(np.average(chosen,weights=mass)),
                regret_to_best_positive_alpha=float(np.average(truth.max(-1)-chosen,weights=mass)),
                alpha_fractions=[float(np.average(best==j,weights=mass)) for j in range(3)])
        entries['distinct_pair_fraction']=float(weights[unequal].sum()/weights.sum())
        entries['teacher_known_best_including_off_gain']=float(np.average(np.maximum(0,truth.max(-1)),weights=mass))
        entries['fixed_alpha_gain']=[float(np.average(truth[...,j],weights=mass)) for j in range(3)]
        out[name]=entries
    return out


def run(source):
    p=read(source/'protocol.json');result=read(source/'probe_result.json')
    manifest=read(source/'probe_data_manifest.json')
    for name in ('candidate_audit.json','probe_audit.json'):
        if not read(source/name)['passed']:raise ValueError('Audited S1y inputs required')
    # Slice to original calibration BEFORE any ranking/selection operation.
    ids=p['parts']['calibration'];rows=[p['rows'][i] for i in ids]
    groups=sorted({r['group'] for r in rows if r['kind']=='expert'})
    if len(groups)!=8 or len({r['group'] for r in rows if r['kind']=='anchor'})!=1:
        raise ValueError('Unexpected calibration scene coverage')
    dest=new_run('component_calibration',group=('reports','analyses'))
    files=[Path(__file__),source/'protocol.json',source/'probe_result.json',source/'probe_data_manifest.json',
           ROOT/'cx_tta/scene_gain_probe.py']
    files += [Path(r['artifact']) for r in result['reports']]+[Path(r['file']) for r in manifest['records']]
    write_json(dest/'protocol.json',dict(status='frozen_before_new_selection_results',source=str(source),
        source_sha256={str(f):digest(f) for f in files},rows=rows,source_indices=ids,
        scores={k:dict(head=v[0],thresholds=v[1]) for k,v in SCORES.items()},risk_thresholds=RISKS,
        comparison='2x2 frozen gain/benefit head ranking x shared/split component thresholds; '
                   'both use <=.05 all-unit error-harm for EACH component and zero anchor harm. '
                   'Additional original pooled-risk shared-gain reference isolates guard change.',
        validation='Leave one of EIGHT calibration expert scenes out; other seven + same CUP anchor calibrate. '
                   'No new S1y validation evaluation. Development-only; original calibration already inspected. '
                   'Repeated seeds/arms/frames are not independent scenes; cup anchor is NOT held out.',
        objective='Task/scene-balanced realized normalized gain; fixed grid plus off. '
                  'Ties prefer lower coverage then larger score threshold then smaller risk.',
        ranking='Within same state/seed/arm/component, compare 3 positive alpha pair gain orders; '
                'label differences <=1e-8 excluded, prediction ties half credit.',
        formal_training=False,deployment=False,tta=False))
    panels={}
    for entry in manifest['records']:
        f=Path(entry['file'])
        if digest(f)!=entry['sha256']:raise ValueError('Changed labels')
        with np.load(f) as z:panels[entry['encoding'],entry['support']]={k:z[k][:,ids].copy() for k in z.files}
    reports=[];checks=dict(original_calibrations_reproduced=0,held_scene_poison_checks=0,
                          split_objective_dominance_checks=0,validation_rows_used=0)
    all_ids=list(range(len(rows)))
    for entry in result['reports']:
        if digest(Path(entry['artifact']))!=entry['sha256']:raise ValueError('Changed prediction')
        with np.load(entry['artifact']) as z:pred=z['prediction'][:,ids].copy()
        panel=panels[entry['encoding'],entry['support']]
        original,_=calibrate(pred,panel,rows,all_ids)
        if any(original[k]!=entry['calibration'][k] for k in ('gain_threshold','risk_threshold')):
            raise ValueError('Original full calibration changed')
        checks['original_calibrations_reproduced']+=1
        item={k:entry[k] for k in ('encoding','support','architecture','seed')}
        item['within_state_ranking']=ranking(pred,panel,rows,all_ids)
        item['rules']={}
        choices_to_save={}
        for score in SCORES:
            opts=options(pred,score)
            modes=('shared','split','pooled_reference') if score=='gain' else ('shared','split')
            full_objectives={}
            for mode in modes:
                full,thresholds,_=fit_options(opts,panel,rows,all_ids,mode)
                fm=metrics(panel,full,rows,all_ids)
                full_objectives[mode]=fm['pooled']['gain']
                cross=np.zeros_like(full);cover=np.zeros(len(rows),int);folds=[]
                for group in groups:
                    train=[i for i,r in enumerate(rows) if r['group']!=group]
                    held=[i for i,r in enumerate(rows) if r['group']==group]
                    choice,setting,_=fit_options(opts,panel,rows,train,mode)
                    poisoned={k:v.copy() for k,v in panel.items()}
                    for k,v in poisoned.items():v[:,held]=999. if k=='gain' else ~v[:,held]
                    _,again,_=fit_options(opts,poisoned,rows,train,mode)
                    if setting!=again:raise ValueError('Held label affected thresholds')
                    checks['held_scene_poison_checks']+=1
                    cross[:,held]=choice[:,held];cover[held]+=1
                    folds.append(dict(held_scene=group,thresholds=setting,metrics=metrics(panel,choice,rows,held)))
                if any(cover[i]!=1 for i,r in enumerate(rows) if r['kind']=='expert'):
                    raise ValueError('Missing or repeated held scene')
                name=f'{score}_{mode}'
                item['rules'][name]=dict(full_calibration=dict(thresholds=thresholds,metrics=fm),
                    cross_calibration=metrics(panel,cross,rows,all_ids),folds=folds)
                choices_to_save[name]=cross
            if full_objectives['split']+1e-12<full_objectives['shared']:
                raise ValueError('Split search must contain shared feasible optimum')
            checks['split_objective_dominance_checks']+=1
        tag=f"{entry['encoding']}_{entry['support']}_{entry['architecture']}_s{entry['seed']}"
        np.savez_compressed(dest/(tag+'.npz'),**choices_to_save)
        reports.append(item)
        print('COMPLETE',tag,flush=True)
    frozen=read(dest/'protocol.json')
    if any(digest(Path(f))!=sha for f,sha in frozen['source_sha256'].items()):raise ValueError('Input mutation')
    write_json(dest/'result.json',dict(status='complete',reports=reports,scope=frozen['validation']))
    write_json(dest/'audit.json',dict(passed=True,**checks,source_hashes_unchanged=True))
    print('RUN',dest,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('source',type=Path)
    run(parser.parse_args().source.resolve())

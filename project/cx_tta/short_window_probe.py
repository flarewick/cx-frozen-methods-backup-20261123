"""Train-scene LOSO oracle entity readouts; not shared M1 or held-out validation."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .effect_learning_probe import weights, shuffle_within_checkpoint
from .semantic_effect_probe import squared_distances,rbf,fit_predict
from .paths import new_run,write_json

ALPHA=.1  # Fixed before this diagnostic, no held-scene tuning.


def targets(index,labels):
    groups={}
    for i,(row,label) in enumerate(zip(index['records'],labels)):
        if row['branch']=='base' or row['start']!=0 or 'counterfactual' not in label:continue
        for family,entities in label['counterfactual'].items():
            for entity,value in entities.items():
                views=[label[k][entity]['visible_in_any_camera'] for k in ('visibility_before','visibility_after','base_visibility_after')]
                observable=all(views)
                values=[('translation_m',value['translation_world_m']),('rotation_rad',value['rotation_world_rad'])] if family=='rigid' else [('joint_fraction',np.array(value['delta_range_fraction'])[value['range_valid_mask']])]
                for kind,y in values:
                    if len(y):groups.setdefault((row['task'],entity,kind),[]).append((i,np.array(y,float),observable))
    return groups


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--semantic-cache',type=Path);args=p.parse_args()
    path=args.dataset;index=json.loads((path/'index.json').read_text());labels=json.loads((path/'labels_offline.json').read_text())
    if index['status']!='complete' or any(r['split']!='train' for r in index['records']):raise ValueError('Train-only dataset required')
    if digest(path/'inputs.npz')!=index['inputs_sha256'] or digest(path/'labels_offline.json')!=index['labels_sha256']:raise ValueError('Dataset changed')
    with np.load(path/'inputs.npz') as x:context=x['context'].astype(float);actions=x['action'].astype(float)
    semantic=None
    if args.semantic_cache:
        cache=json.loads((args.semantic_cache/'result.json').read_text())
        if cache['status']!='complete' or cache['limited']:raise ValueError('Complete semantic cache required')
        values={}; sources={r['id']:Path(r['run']) for r in index['sources']}
        for row in cache['records']:
            key=(row['job'],row['offset'])
            if key in values or row['job'] not in sources or row['offset'] not in (-8,0):raise ValueError('Cache identity mismatch')
            result=json.loads((sources[row['job']]/'result.json').read_text())
            frame=next(f for f in result['branches']['base']['frames'] if f['offset']==row['offset'])
            if Path(row['source'])!=sources[row['job']]/frame['observed_file'] or row['source_sha256']!=frame['observed_sha256'] or digest(Path(row['file']))!=row['sha256']:raise ValueError('Cache source/hash mismatch')
            with np.load(row['file']) as z:
                v=z['pooled'].astype(float).reshape(3,-1)
                if str(z['source_sha256'])!=row['source_sha256'] or not np.isfinite(v).all():raise ValueError('Invalid cache values')
                values[key]=(v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)).ravel()
        if set(values)!={(j,o) for j in sources for o in (-8,0)}:raise ValueError('Missing semantic frames')
        semantic=np.stack([np.r_[values[r['job'],0],values[r['job'],-8]] for r in index['records']])
        # Remove coarse RGB; retain current/history command, measured state and planned actions.
        context=context[:,np.r_[0:14,590:618,1194:context.shape[1]]]
    run=new_run('short_probe',group=('effect_learning',));print(run,flush=True);reports=[];predictions={}
    for (task,entity,kind),items in targets(index,labels).items():
        ids=np.array([a[0] for a in items]);y=np.stack([a[1] for a in items]);visible=np.array([a[2] for a in items]);records=[index['records'][i] for i in ids]
        groups=sorted({r['group'] for r in records});dist=squared_distances(context[ids]);vdist=squared_distances(semantic[ids]) if semantic is not None else None;action=actions[ids]
        shuffled=[shuffle_within_checkpoint(action,records,s) for s in (0,1,2)]
        modes=['zero','no_action','action_only','conditioned','shuffled_0','shuffled_1','shuffled_2']
        preds={m:np.full_like(y,np.nan) for m in modes};folds=[]
        for group in groups:
            train=np.array([i for i,r in enumerate(records) if r['group']!=group and visible[i]],int)
            test=np.array([i for i,r in enumerate(records) if r['group']==group],int)
            if not len(train):folds.append({'held':group,'status':'no_visible_train_labels'});continue
            kc,_=rbf(dist,train)
            if vdist is not None:kc=(kc+rbf(vdist,train)[0])/2
            for mode in modes:
                if mode=='zero':preds[mode][test]=0;continue
                a=shuffled[int(mode[-1])] if mode.startswith('shuffled') else action
                a=np.c_[a,a*a];ka=a@a.T;ka/=max(float(np.diag(ka)[train].mean()),1e-12)
                k=kc if mode=='no_action' else ka if mode=='action_only' else kc*ka
                preds[mode][test],_=fit_predict(k,y,records,train,test,ALPHA,centered=False)
            folds.append({'held':group,'status':'complete','train_rows':len(train),'test_rows':len(test)})
        eligible=np.isfinite(preds['conditioned']).all(axis=1);metrics={}
        for name,mask in [('all_labeled',eligible),('visible_start_and_end',eligible&visible)]:
            selected=np.flatnonzero(mask)
            if not len(selected):continue
            w=weights([records[i] for i in selected]);scores={m:float(w@np.mean((v[selected]-y[selected])**2,axis=1)) for m,v in preds.items()}
            scores['shuffled_mean']=float(np.mean([scores[f'shuffled_{s}'] for s in (0,1,2)]))
            metrics[name]={'rows':len(selected),'mse':scores,'relative_improvement_vs_zero':{m:1-v/scores['zero'] for m,v in scores.items()} if scores['zero']>1e-16 else None}
        key=f'{len(reports):03d}';predictions[key+'_target']=y;predictions[key+'_row_ids']=ids;predictions[key+'_visible']=visible
        for m,v in preds.items():predictions[key+'_'+m]=v
        reports.append(dict(task=task,entity=entity,kind=kind,rows=len(ids),visible_rows=int(visible.sum()),scene_count=len(groups),folds=folds,metrics=metrics,prediction_key=key))
    np.savez_compressed(run/'predictions.npz',**predictions)
    write_json(run/'result.json',dict(status='complete',dataset=str(path),dataset_index_sha256=digest(path/'index.json'),alpha=ALPHA,reports=reports,semantic_cache=str(args.semantic_cache) if args.semantic_cache else None,
      scope='Oracle task/entity-associated readouts, RGB/history (coarse or frozen semantic cache as recorded), fixed-alpha leave-one-TRAIN-scene-out; not shared entity discovery, held-out validation, residual controller or TTA',
      visibility_protocol='Train loss only when entity rendered at start and both branch/base ends. All-label and conditional-visible scores retained; future visibility is offline supervision, never input. Pixel visibility does not guarantee 3D observability.'))
    print(f'COMPLETE readouts={len(reports)} output={run}',flush=True)

if __name__=='__main__':main()

"""Bounded CPU shared-head training: original-train scene/task held-out probes."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from .audit import digest
from .effect_learning_probe import weights
from .paths import new_run,write_json,replace_json
from .shared_effect_model import SharedEffect,fold_indices


def shuffled_actions(action,records,seed=0):
    rng=np.random.default_rng(seed);out=np.empty_like(action)
    lookup={(r['job'],r['entity'],r['horizon'],r['branch']):i for i,r in enumerate(records)}
    for job in sorted({r['job'] for r in records}):
        branches=sorted({r['branch'] for r in records if r['job']==job});perm=rng.permutation(len(branches))
        while np.any(perm==np.arange(len(branches))):perm=rng.permutation(len(branches))
        mapping={b:branches[j] for b,j in zip(branches,perm)}
        for i,r in enumerate(records):
            if r['job']==job:out[i]=action[lookup[job,r['entity'],r['horizon'],mapping[r['branch']]]]
    return out


def fit_preprocess(vision,geometry,action,records,train):
    unique={}
    for i in train:unique.setdefault((records[i]['job'],records[i]['entity']),i)
    ids=list(unique.values());pca=PCA(n_components=min(32,len(ids)-1,vision.shape[1]),svd_solver='randomized',random_state=0)
    pca.fit(vision[ids]);v=pca.transform(vision)
    context=np.c_[v,geometry];mean=context[train].mean(0);std=np.maximum(context[train].std(0),.05)
    action_scale=np.maximum(np.sqrt(np.mean(action[train]**2,axis=0)),.1)
    return ((context-mean)/std).astype(np.float32),(action/action_scale).astype(np.float32),dict(pca_mean=pca.mean_,pca_components=pca.components_,context_mean=mean,context_std=std,action_scale=action_scale,fit_rows=np.array(train),pca_fit_rows=np.array(ids))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--epochs',type=int,default=80);parser.add_argument('--seed',type=int,default=0);args=parser.parse_args()
    if args.epochs<1:parser.error('Positive epochs required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    index=json.loads((args.dataset/'index.json').read_text());records=index['records']
    if index['status']!='complete' or any(r['split']!='train' for r in records):raise ValueError('Only original train scenes allowed')
    if digest(args.dataset/'inputs_oracle.npz')!=index['inputs_sha256'] or digest(args.dataset/'labels_offline.npz')!=index['labels_sha256']:raise ValueError('Dataset changed')
    with np.load(args.dataset/'inputs_oracle.npz') as z:vision=z['vision'];geometry=z['geometry'];action=z['action']
    with np.load(args.dataset/'labels_offline.npz') as z:y=z['target'];mask=z['mask'];scale=z['scale']
    visibility=np.array([r['visible_supervision_only'] for r in records]);shuffled=shuffled_actions(action,records,args.seed)
    run=new_run('shared_effect',group=('effect_learning',));print(run,flush=True)
    modes=('conditioned','action_only','no_action','shuffled')
    write_json(run/'manifest.json',dict(dataset=str(args.dataset),dataset_index_sha256=digest(args.dataset/'index.json'),epochs=args.epochs,seed=args.seed,width=32,learning_rate=.002,weight_decay=.0001,loss='scene-balanced masked smooth L1, per-output normalization',scope='Privileged object mask/association; single shared 7-output model, no task/entity head selection or IDs as features. Fixed budget, no evaluation-based early stopping. Original train only.',code_sha256=digest(Path(__file__)),model_sha256=digest(Path(__file__).with_name('shared_effect_model.py'))))
    reports=[];folds=[];n=len(records)
    for unit in ('group','task'):
        predictions={mode:np.full((n,7),np.nan,np.float32) for mode in modes};supported=np.zeros((n,7),bool)
        for fold,held in enumerate(sorted({r[unit] for r in records})):
            train,test=fold_indices(records,held,unit);context,a,pre=fit_preprocess(vision,geometry,action,records,train)
            support=(mask[train]&visibility[train,None]).any(axis=0);supported[test]=support
            tag=f'{unit}_{fold:02d}';np.savez_compressed(run/(tag+'_preprocess.npz'),**pre)
            x=torch.from_numpy(context[train]);target=torch.from_numpy(y[train]);w=torch.from_numpy(weights([records[i] for i in train]).astype(np.float32))[:,None]
            valid=torch.from_numpy((mask[train]&visibility[train,None]).astype(np.float32));denom=(w*valid).sum(0).clamp(min=1e-12)
            for mode in modes:
                torch.manual_seed(args.seed)
                model=SharedEffect(context.shape[1],a.shape[1],mode='conditioned' if mode=='shuffled' else mode)
                source_a=(shuffled/pre['action_scale']).astype(np.float32) if mode=='shuffled' else a
                input_a=torch.from_numpy(source_a[train]);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
                model.train()
                for epoch in range(args.epochs):
                    opt.zero_grad();out=model(x,input_a);loss=((torch.nn.functional.smooth_l1_loss(out,target,reduction='none')*valid*w).sum(0)/denom).sum()/max(int(support.sum()),1);loss.backward();opt.step()
                model.eval()
                with torch.no_grad():predictions[mode][test]=model(torch.from_numpy(context[test]),torch.from_numpy(source_a[test])).numpy()
                predictions[mode][np.ix_(test,~support)]=np.nan
                torch.save(dict(state_dict=model.state_dict(),mode=mode,context_dim=context.shape[1],action_dim=a.shape[1],supported=support.tolist(),scale=scale.tolist()),run/(tag+'_'+mode+'.pt'))
                folds.append(dict(unit=unit,held=held,mode=mode,train_rows=len(train),test_rows=len(test),train_scenes=len({records[i]['group'] for i in train}),supported=support.tolist(),train_loss=float(loss.detach()),parameters=sum(p.numel() for p in model.parameters())))
            replace_json(run/'progress.json',dict(status='running',folds=folds));print(f'FOLD {unit} {held} complete',flush=True)
        np.savez_compressed(run/(unit+'_predictions.npz'),**predictions,target=y,mask=mask,supported=supported)
        for task,entity,horizon in sorted({(r['task'],r['entity'],r['horizon']) for r in records}):
            take=np.array([i for i,r in enumerate(records) if (r['task'],r['entity'],r['horizon'])==(task,entity,horizon)],int)
            for kind,columns in [('translation_m',np.arange(3)),('rotation_rad',np.arange(3,6)),('joint_fraction',np.array([6]))]:
                required=mask[np.ix_(take,columns)].all(1)
                if not required.any():continue
                ids=take[required];supported_rows=supported[np.ix_(ids,columns)].all(1)
                if not supported_rows.all():reports.append(dict(unit=unit,task=task,entity=entity,horizon=horizon,kind=kind,status='unsupported_output_in_training',rows=len(ids)));continue
                w=weights([records[i] for i in ids]);actual=y[np.ix_(ids,columns)]*scale[columns]
                mse={'zero':float(w@np.mean(actual**2,axis=1))}
                for mode,pred in predictions.items():mse[mode]=float(w@np.mean(((pred[np.ix_(ids,columns)]-y[np.ix_(ids,columns)])*scale[columns])**2,axis=1))
                reports.append(dict(unit=unit,task=task,entity=entity,horizon=horizon,kind=kind,status='complete',rows=len(ids),visible_rows=int(visibility[ids].sum()),mse=mse,improvement={k:1-v/mse['zero'] for k,v in mse.items()} if mse['zero']>1e-16 else None))
    write_json(run/'result.json',dict(status='complete',reports=reports,folds=folds,scope='Shared neural oracle-object diagnosis; group=leave-one-original-train-scene-out, task=leave-one-original-train-task-out. Not original validation, not blind benchmark, not policy success rate. Single seed fixed 80-epoch default; no instruction embedding or learned object detector. Switch-only joint output unidentifiable in switch task holdout.'))
    print('COMPLETE '+str(run),flush=True)

if __name__=='__main__':main()

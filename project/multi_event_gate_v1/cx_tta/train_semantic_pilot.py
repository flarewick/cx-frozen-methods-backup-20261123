"""Fixed semantic/direction/preservation ablation; all folds stay episode-disjoint."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from .audit import digest
from .dataset import load_cache, context_features
from .direction_adapter import DirectionAdapter, losses
from .paths import new_run, write_json, replace_json
from .train_direction_pilot import evaluate


def past_indices(records):
    history=np.full(len(records),-1,int);previous={}
    for i,row in enumerate(records):
        key=row['group']
        if key in previous:
            j=previous[key]
            if records[j]['frame']>=row['frame'] or records[j]['pilot_split']!=row['pilot_split']:
                raise ValueError('History is noncausal or crosses partition')
            history[i]=j
        previous[key]=i
    return history


def load_features(cache):
    manifest=json.loads((cache/'manifest.json').read_text());result=json.loads((cache/'result.json').read_text())
    if manifest['smoke_only'] or result['status']!='complete' or digest(Path(manifest['plan']))!=manifest['plan_sha256']:
        raise ValueError('Full frozen feature cache required')
    plan=json.loads(Path(manifest['plan']).read_text());rows=[];arrays={k:[] for k in ('joint','coarse','base','target','valid','direction','direction_valid')}
    if len(result['records'])!=len(plan['records']):raise ValueError('Incomplete feature coverage')
    for encoded,expected in zip(result['records'],plan['records']):
        r=encoded['source']
        if r!=expected or digest(Path(encoded['file']))!=encoded['sha256']:raise ValueError('Feature source mismatch')
        with np.load(encoded['file']) as z:joint=z['joint'].astype(np.float32);frames=z['frames']
        if r['kind']=='expert':
            if digest(Path(r['cache']))!=r['cache_sha256'] or digest(Path(r['label']))!=r['label_sha256']:raise ValueError('Changed expert targets')
            d=load_cache(Path(r['cache']))
            with np.load(r['label']) as z:labels=dict(z)
            if not np.array_equal(frames,d['frame_index']) or not np.array_equal(frames,labels['source_frame_index']):raise ValueError('Frame alignment mismatch')
            values=dict(joint=joint,coarse=d['context'],base=d['base_action'],target=d['target'],valid=d['valid'],direction=labels['expert_plan_direction'],direction_valid=labels['direction_valid'])
        else:
            if digest(Path(r['checkpoint']))!=r['checkpoint_sha256'] or frames.tolist()!=[r['chunk']]:raise ValueError('Changed anchor frame')
            with np.load(r['checkpoint']) as z:d=dict(z)
            coarse=context_features([d[k+'_rgb'] for k in ('head','right','left')],d['command_state'])[None]
            base=d['base_actions'][None]
            values=dict(joint=joint,coarse=coarse,base=base,target=base.copy(),valid=np.ones((1,32),bool),direction=np.zeros((1,5,14),np.float32),direction_valid=np.zeros((1,5),bool))
        for key,value in values.items():arrays[key].append(value)
        rows.extend(dict(task=r['task'],group=r['group'],frame=int(f),kind=r['kind'],pilot_split=r['pilot_split']) for f in frames)
    arrays={k:np.concatenate(v) for k,v in arrays.items()};history=past_indices(rows)
    fit=np.array([i for i,r in enumerate(rows) if r['pilot_split']=='fit']);held=np.array([i for i,r in enumerate(rows) if r['pilot_split']=='inner_held'])
    if {rows[i]['group'] for i in fit}&{rows[i]['group'] for i in held}:raise ValueError('Expert split leakage')
    contexts={};preprocess={}
    for name,features in [('coarse',arrays['coarse'][:,14:]),('semantic',arrays['joint'])]:
        pca=PCA(n_components=64,svd_solver='randomized',random_state=0).fit(features[fit]);v=pca.transform(features).astype(np.float32)
        current=np.c_[v,arrays['coarse'][:,:14]];past=np.zeros_like(current);exists=history>=0;past[exists]=current[history[exists]]
        combined=np.c_[current,past,exists.astype(np.float32)]
        mean=combined[fit].mean(0);std=np.maximum(combined[fit].std(0),.05)
        contexts[name]=((combined-mean)/std).astype(np.float32)
        preprocess[name]=dict(pca_mean=pca.mean_,pca_components=pca.components_,context_mean=mean,context_std=std,fit_indices=fit,history_indices=history)
    return plan,rows,arrays,contexts,preprocess


def balanced_weights(rows):
    tasks={r['task'] for r in rows};groups={t:{r['group'] for r in rows if r['task']==t} for t in tasks};counts=Counter(r['group'] for r in rows)
    return torch.tensor([1/(len(tasks)*len(groups[r['task']])*counts[r['group']]) for r in rows],dtype=torch.double)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);threadpool_limits(2)
    plan,rows,arrays,contexts,pre=load_features(a.cache);contract=plan['training_contract']
    run=new_run('semantic_adapter_pilot',group=('effect_learning',));print(run,flush=True)
    partitions={s:np.array([i for i,r in enumerate(rows) if r['pilot_split']==s]) for s in ('fit','inner_held','anchor_fit','anchor_held')}
    for s,ids in partitions.items():
        if not len(ids):raise ValueError('Missing required split')
    fit=partitions['fit'];anchors=partitions['anchor_fit']
    if {rows[i]['group'] for i in anchors}&{rows[i]['group'] for i in partitions['anchor_held']}:raise ValueError('Anchor split leakage')
    write_json(run/'protocol.json',dict(cache=str(a.cache.resolve()),cache_manifest_sha256=digest(a.cache/'manifest.json'),cache_result_sha256=digest(a.cache/'result.json'),contract=contract,rows=rows,partitions={s:ids.tolist() for s,ids in partitions.items()},context_dim=contexts['semantic'].shape[1],model_code_sha256=digest(Path(__file__).with_name('direction_adapter.py')),trainer_sha256=digest(Path(__file__)),feature_code_sha256=digest(Path(__file__).with_name('cache_joint_prefix.py')),scope='PCA and normalization fit on expert fit rows only; no anchor or held rows used in preprocessing fit. Strict same-episode past observations, irregular gaps. Fixed final epoch, no hyperparameter search.'))
    for name,params in pre.items():np.savez_compressed(run/f'{name}_preprocess.npz',**params)
    reports=[];all_history={}
    for variant in contract['modes']:
        representation='coarse' if variant=='coarse_bc' else 'semantic';mode='conditioned' if 'direction' in variant else 'bc';preserve=variant.endswith('_preserve')
        torch.manual_seed(contract['seed']);model=DirectionAdapter(context_dim=contexts[representation].shape[1],mode=mode);opt=torch.optim.AdamW(model.parameters(),lr=contract['lr'])
        full=[torch.from_numpy(x) for x in (contexts[representation],arrays['base'],arrays['target'],arrays['valid'],arrays['direction'],arrays['direction_valid'])]
        data={s:[t[ids] for t in full] for s,ids in partitions.items()};split_rows={s:[rows[i] for i in ids] for s,ids in partitions.items()}
        baseline,_=evaluate(model,data['inner_held'],split_rows['inner_held'])
        generator=torch.Generator().manual_seed(contract['seed']);agenerator=torch.Generator().manual_seed(contract['seed']+123)
        weights=balanced_weights(split_rows['fit']);aw=balanced_weights(split_rows['anchor_fit']);history=[]
        for epoch in range(contract['epochs']):
            sampled=torch.multinomial(weights,len(fit),replacement=True,generator=generator);model.train()
            for start in range(0,len(sampled),64):
                ids=sampled[start:start+64];batch=[x[ids] for x in data['fit']];opt.zero_grad();loss,_=losses(model,*batch,auxiliary_weight=contract['auxiliary_weight'])
                if preserve:
                    ai=torch.multinomial(aw,len(ids),replacement=True,generator=agenerator);_,delta,_=model(data['anchor_fit'][0][ai],data['anchor_fit'][1][ai]);loss=loss+contract['preservation_weight']*(delta/model.caps).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();opt.step()
            val,_=evaluate(model,data['inner_held'],split_rows['inner_held']);anchor,_=evaluate(model,data['anchor_held'],split_rows['anchor_held']);history.append(dict(epoch=epoch+1,expert_held=val,anchor_held=anchor));replace_json(run/'progress.json',dict(variant=variant,history=history,completed=reports));print(variant,epoch+1,val,flush=True)
        torch.save(model.state_dict(),run/f'{variant}.pt');restored=DirectionAdapter(context_dim=contexts[representation].shape[1],mode=mode);restored.load_state_dict(torch.load(run/f'{variant}.pt',weights_only=True));results={}
        for part in ('inner_held','anchor_fit','anchor_held'):
            metrics,pred=evaluate(model,data[part],split_rows[part]);_,again=evaluate(restored,data[part],split_rows[part])
            if not np.array_equal(pred,again):raise ValueError('Reload mismatch')
            delta=pred-data[part][1].numpy()
            if not np.isfinite(pred).all() or np.max(np.abs(delta)-model.caps.numpy())>1e-6:raise ValueError('Residual cap violation')
            np.savez_compressed(run/f'{variant}_{part}.npz',predicted=pred,base=data[part][1].numpy(),target=data[part][2].numpy(),valid=data[part][3].numpy(),indices=partitions[part])
            results[part]=metrics
        reports.append(dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),baseline=baseline,metrics=results,improvement_vs_zero={t:1-results['inner_held'][t]/baseline[t] for t in baseline},reload='exact'));all_history[variant]=history
    write_json(run/'history.json',all_history);write_json(run/'result.json',dict(status='complete',reports=reports,scope='Fixed single-seed 10-epoch bounded pilot, expert 48 fit/12 held episodes plus 5 fit/2 held L0 success scenes. Original held-out sets untouched. Anchor error is deviation from successful recorded policy, NOT task success or recovery. Explicit instruction joint features encode observable state too; compare same-feature BC to isolate direction/preservation. No model promotion based on fit-anchor memorization.'))
    print('COMPLETE '+str(run),flush=True)

if __name__=='__main__':main()

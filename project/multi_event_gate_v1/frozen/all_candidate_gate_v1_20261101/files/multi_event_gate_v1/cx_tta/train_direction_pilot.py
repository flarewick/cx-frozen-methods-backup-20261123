"""Fixed-budget expert-direction pilot, inner episode split; no deployment claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .audit import digest
from .dataset import load_cache
from .direction_adapter import DirectionAdapter, losses
from .paths import new_run, write_json, replace_json


def read_parts(index):
    eligible=[r for r in index['records'] if r['status']=='prepared']
    rng=np.random.default_rng(0);held=set()
    for task in sorted({r['task'] for r in eligible}):
        groups=sorted(r['group'] for r in eligible if r['task']==task)
        if len(groups)!=50:raise ValueError('Expected original 50 training episodes per task')
        held.update(rng.choice(groups,5,replace=False).tolist())
    parts={s:[] for s in ('fit','inner_held')};rows={s:[] for s in parts}
    for r in eligible:
        if r['split']!='train' or digest(Path(r['cache']))!=r['cache_sha256'] or digest(Path(r['label']))!=r['label_sha256']:raise ValueError('Changed data or held-out source')
        cache=load_cache(Path(r['cache']))
        with np.load(r['label']) as z:lab=dict(z)
        if not np.array_equal(cache['frame_index'],lab['source_frame_index']):raise ValueError('Misaligned inputs')
        part='inner_held' if r['group'] in held else 'fit'
        parts[part].append([cache['context'],cache['base_action'],cache['target'],cache['valid'],lab['expert_plan_direction'],lab['direction_valid']])
        rows[part].extend([dict(task=r['task'],group=r['group'],frame=int(f)) for f in cache['frame_index']])
    if {r['group'] for r in rows['fit']}&{r['group'] for r in rows['inner_held']}:raise ValueError('Episode leakage')
    tensors={s:[torch.from_numpy(np.concatenate([x[k] for x in values])) for k in range(6)] for s,values in parts.items()}
    return tensors,rows


def evaluate(model,data,rows):
    model.eval();error=[];predictions=[]
    with torch.no_grad():
        for start in range(0,len(rows),64):
            c,b,y,v,d,dv=[t[start:start+64] for t in data]
            pred,_,_=model(c,b);predictions.append(pred.numpy())
            error.append(((((pred-y)/model.caps).square().mean(-1)*v).sum(-1)/v.sum(-1)).numpy())
    error=np.concatenate(error);reports={}
    for task in sorted({r['task'] for r in rows}):
        groups={r['group'] for r in rows if r['task']==task}
        reports[task]=float(np.mean([np.mean(error[[i for i,r in enumerate(rows) if r['group']==g]]) for g in sorted(groups)]))
    return reports,np.concatenate(predictions)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    index=json.loads((a.dataset/'index.json').read_text())
    if index['status']!='complete':raise ValueError('Incomplete direction dataset')
    data,rows=read_parts(index);run=new_run('direction_adapter_pilot',group=('effect_learning',));print(run,flush=True)
    modes=('bc','auxiliary','conditioned');epochs=10;seed=0
    write_json(run/'protocol.json',dict(status='frozen_before_training',dataset=str(a.dataset),dataset_index_sha256=digest(a.dataset/'index.json'),modes=modes,epochs=epochs,seed=seed,learning_rate=3e-4,auxiliary_weight=.1,rows=rows,selection='Final epoch 10 only; no early stopping or hyperparameter search',partition='Original 200 train episodes -> fixed per-task 45 fit / 5 inner-held; original 40 demonstration validation and six intervention validation scenes untouched',model_sha256=digest(Path(__file__).with_name('direction_adapter.py')),trainer_sha256=digest(Path(__file__)),scope='Coarse observable B2 features; instruction enters implicitly via frozen base actions. Planned expert FK correction, NOT object goal/progress. Three controls share declared parameter count but BC/aux conditioning channels dormant. Fixed offline initialization pilot, not method promotion.'))
    task_groups={t:sorted({r['group'] for r in rows['fit'] if r['task']==t}) for t in sorted({r['task'] for r in rows['fit']})}
    counts={g:sum(r['group']==g for r in rows['fit']) for gs in task_groups.values() for g in gs}
    weights=torch.tensor([1/(len(task_groups)*len(task_groups[r['task']])*counts[r['group']]) for r in rows['fit']],dtype=torch.double)
    results=[];histories={}
    for mode in modes:
        torch.manual_seed(seed);model=DirectionAdapter(mode=mode);opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
        baseline,_=evaluate(model,data['inner_held'],rows['inner_held'])
        generator=torch.Generator().manual_seed(seed);history=[]
        for epoch in range(epochs):
            sampled=torch.multinomial(weights,len(weights),replacement=True,generator=generator);model.train();losses_epoch=[]
            for start in range(0,len(sampled),64):
                ids=sampled[start:start+64];batch=[t[ids] for t in data['fit']];opt.zero_grad();loss,detail=losses(model,*batch)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();opt.step();losses_epoch.append(float(loss.detach()))
            validation,_=evaluate(model,data['inner_held'],rows['inner_held']);history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses_epoch)),inner_held_by_task=validation))
            replace_json(run/'progress.json',dict(mode=mode,epoch=epoch+1,history=history,completed=results));print(mode,epoch+1,validation,flush=True)
        final,pred=evaluate(model,data['inner_held'],rows['inner_held']);torch.save(model.state_dict(),run/f'{mode}_final.pt')
        restored=DirectionAdapter(mode=mode);restored.load_state_dict(torch.load(run/f'{mode}_final.pt',weights_only=True));check,pred_reload=evaluate(restored,data['inner_held'],rows['inner_held'])
        if not np.array_equal(pred,pred_reload):raise ValueError('Checkpoint reload changed predictions')
        np.savez_compressed(run/f'{mode}_inner_predictions.npz',predicted=pred,target=data['inner_held'][2].numpy(),valid=data['inner_held'][3].numpy())
        results.append(dict(mode=mode,parameters=sum(p.numel() for p in model.parameters()),zero_baseline=baseline,final=final,improvement_vs_zero={k:1-final[k]/baseline[k] for k in final},reload='exact'))
        histories[mode]=history
    write_json(run/'history.json',histories)
    write_json(run/'result.json',dict(status='complete',reports=results,scope='One fixed training seed, ten epochs, 20 inner-held demonstration episodes. Episode-balanced native action imitation error only. No use of original held-out data or intervention success labels; no closed-loop success or recovery/generalization claim.'))
    print('COMPLETE '+str(run),flush=True)

if __name__=='__main__':main()

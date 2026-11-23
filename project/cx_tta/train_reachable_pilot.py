"""Fixed paired loss diagnostic on cached observations, not a policy evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .audit import digest
from .direction_adapter import DirectionAdapter
from .paths import new_run, write_json, replace_json
from .train_semantic_pilot import load_features, balanced_weights


def supervised_loss(predicted, base, expert, valid, caps, mode):
    if mode not in ('expert_mse', 'reachable_mse'):
        raise ValueError('Unknown fixed loss control')
    target = expert if mode == 'expert_mse' else base + (expert-base).clamp(-caps, caps)
    error = ((predicted-target)/caps).square().mean(-1)
    return ((error*valid).sum(-1)/valid.sum(-1)).mean()


def metric(values, rows):
    result = {}
    for task in sorted({r['task'] for r in rows}):
        groups = sorted({r['group'] for r in rows if r['task']==task})
        result[task] = float(np.mean([np.mean(values[[i for i,r in enumerate(rows) if r['group']==g]]) for g in groups]))
    return result


def evaluate(model, context, base, target, valid, rows):
    with torch.no_grad():
        predicted = model(context, base)[0]
        feasible = base + (target-base).clamp(-model.caps, model.caps)
        metrics = {}
        for name, prediction, goal in [('expert_mse',predicted,target),('base_expert_mse',base,target),
                ('reachable_mse',predicted,feasible),('base_reachable_mse',base,feasible),
                ('residual_energy',predicted,base)]:
            error = (((prediction-goal)/model.caps).square().mean(-1)*valid).sum(-1)/valid.sum(-1)
            metrics[name] = metric(error.cpu().numpy(),rows)
        return metrics,predicted.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; no silent CPU fallback')
    previous=json.loads((args.source_run/'protocol.json').read_text())
    run=new_run('reachable_loss_pilot',group=('effect_learning',))
    contract=dict(source_run=str(args.source_run.resolve()),source_protocol_sha256=digest(args.source_run/'protocol.json'),
        cache=previous['cache'],seeds=[0,1,2],modes=['expert_mse','reachable_mse'],epochs=10,lr=.0003,
        batch_size=64,device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
        torch_version=torch.__version__,code_sha256=digest(Path(__file__)),model_sha256=digest(Path(__file__).with_name('direction_adapter.py')),
        protocol='All expert rows retained; identical paired initialization and task/episode-balanced minibatches; no direction auxiliary, anchor loss, cap changes, checkpoint selection or new labels. Same original 48 fit/12 inner-held episodes and 5/2 success anchor scenes. Three seeds, final epoch only. Both native expert error and reachable-target error reported; neither is task success.')
    write_json(run/'protocol.json',contract)
    print('RUN '+str(run),flush=True)
    plan,rows,arrays,_,_=load_features(Path(previous['cache']))
    if rows!=previous['rows']:
        raise ValueError('Changed data ordering or partition')
    # Use saved preprocessing exactly, not a newly selected representation.
    with np.load(args.source_run/'semantic_preprocess.npz') as z: pre=dict(z)
    fit_indices=np.array([i for i,r in enumerate(rows) if r['pilot_split']=='fit'])
    if not np.array_equal(pre['fit_indices'],fit_indices):
        raise ValueError('Preprocessing fit split changed')
    visual=(arrays['joint']-pre['pca_mean'])@pre['pca_components'].T
    current=np.c_[visual,arrays['coarse'][:,:14]];past=np.zeros_like(current)
    history=pre['history_indices'];exists=history>=0;past[exists]=current[history[exists]]
    context=((np.c_[current,past,exists.astype(np.float32)]-pre['context_mean'])/pre['context_std']).astype(np.float32)
    partitions={s:np.array([i for i,r in enumerate(rows) if r['pilot_split']==s]) for s in ('fit','inner_held','anchor_fit','anchor_held')}
    if any(partitions[k].tolist()!=previous['partitions'][k] for k in partitions):
        raise ValueError('Partition mismatch')
    write_json(run/'data_contract.json',dict(rows=rows,partitions={k:v.tolist() for k,v in partitions.items()},
        preprocessing_sha256=digest(args.source_run/'semantic_preprocess.npz'),cache_manifest_sha256=digest(Path(previous['cache'])/'manifest.json'),cache_result_sha256=digest(Path(previous['cache'])/'result.json')))
    torch_data={key:torch.from_numpy(val).to(args.device) for key,val in dict(context=context,base=arrays['base'],target=arrays['target'],valid=arrays['valid']).items()}
    weights=balanced_weights([rows[i] for i in fit_indices])
    reports=[]
    for seed in contract['seeds']:
        for mode in contract['modes']:
            torch.manual_seed(seed)
            model=DirectionAdapter(context_dim=context.shape[1],mode='bc').to(args.device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=contract['lr'])
            generator=torch.Generator().manual_seed(seed)
            trace=[]
            for epoch in range(contract['epochs']):
                sampled=torch.multinomial(weights,len(fit_indices),replacement=True,generator=generator).numpy()
                total=0.;count=0
                for start in range(0,len(sampled),contract['batch_size']):
                    ids=fit_indices[sampled[start:start+contract['batch_size']]]
                    c,b,y,v=[torch_data[k][ids] for k in ('context','base','target','valid')]
                    optimizer.zero_grad();pred=model(c,b)[0]
                    loss=supervised_loss(pred,b,y,v,model.caps,mode)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();optimizer.step();total+=float(loss.detach())*len(ids);count+=len(ids)
                trace.append(total/count)
            stem=f'seed{seed}_{mode}'
            torch.save(model.state_dict(),run/(stem+'.pt'))
            restored=DirectionAdapter(context_dim=context.shape[1],mode='bc').to(args.device)
            restored.load_state_dict(torch.load(run/(stem+'.pt'),map_location=args.device,weights_only=True))
            metrics={}
            for split in partitions:
                ids=partitions[split];data=[torch_data[k][ids] for k in ('context','base','target','valid')]
                metrics[split],pred=evaluate(model,*data,[rows[i] for i in ids])
                _,again=evaluate(restored,*data,[rows[i] for i in ids])
                if not np.array_equal(pred,again):raise ValueError('Reload discrepancy')
                base=data[1].cpu().numpy()
                if np.max(np.abs(pred-base)-model.caps.cpu().numpy())>1e-6:raise ValueError('Cap violation')
                np.savez_compressed(run/f'{stem}_{split}.npz',predicted=pred,base=base,target=data[2].cpu().numpy(),valid=data[3].cpu().numpy(),indices=ids)
            reports.append(dict(seed=seed,mode=mode,metrics=metrics,train_loss=trace,reload='exact',weights_sha256=digest(run/(stem+'.pt'))))
            replace_json(run/'progress.json',dict(status='running',reports=reports))
            print(stem,json.dumps(metrics['inner_held']['expert_mse']),flush=True)
    write_json(run/'result.json',dict(status='complete',reports=reports,limits='Fixed exploratory loss diagnostic. Existing developer inner holdout reused; not task holdout, closed-loop efficacy or failure recovery supervision. Anchors used for evaluation only, not training.'))
    print('COMPLETE '+str(run),flush=True)


if __name__=='__main__':main()

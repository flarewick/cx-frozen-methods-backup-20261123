"""Paired raw/bounded FK auxiliaries on reachable BC; reuse audited BC seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .audit import digest
from .direction_adapter import DirectionAdapter, losses
from .paths import new_run, write_json, replace_json
from .train_reachable_pilot import evaluate
from .train_semantic_pilot import balanced_weights


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--dataset',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    index=json.loads((args.dataset/'index.json').read_text());baseline=Path(index['baseline'])
    if index['status']!='complete' or digest(args.dataset/'data.npz')!=index['data_sha256'] or digest(baseline/'result.json')!=index['baseline_result_sha256']:
        raise ValueError('Changed dataset or baseline')
    previous=json.loads((baseline/'result.json').read_text());previous_protocol=json.loads((baseline/'protocol.json').read_text())
    if digest(Path(__file__).with_name('direction_adapter.py'))!=previous_protocol['model_sha256']:
        raise ValueError('Model architecture changed')
    audited=json.loads((baseline/'paired_audit.json').read_text())
    rows=index['rows'];partitions={s:np.array(ids) for s,ids in index['partitions'].items()}
    with np.load(args.dataset/'data.npz') as z:arrays=dict(z)
    run=new_run('consistent_direction_pilot',group=('effect_learning',))
    protocol=dict(dataset=str(args.dataset.resolve()),dataset_index_sha256=digest(args.dataset/'index.json'),
        baseline=str(baseline),seeds=[0,1,2],new_modes=['raw','bounded'],reused_mode='bc',epochs=10,lr=.0003,
        auxiliary_weight=.1,batch_size=64,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        code_sha256=digest(Path(__file__)),model_sha256=previous_protocol['model_sha256'],
        evaluation='Report native expert error, reachable-target error, successful-anchor deviation and realized command FK against both targets. Fixed final epoch; no held-out hyperparameter selection. No physical/task efficacy claim.',
        stop_rule='If no repeatable independent directional benefit, stop this auxiliary-head family and prepare state-matched recovery/progress supervision protocol; no loss-weight sweep.',
        capacity='Identical declared architecture. BC has dormant direction head/conditioning channels; raw versus bounded has identical active capacity. All teachers loss-only; predicted directions condition actions.')
    write_json(run/'protocol.json',protocol);print('RUN '+str(run),flush=True)
    data={k:torch.from_numpy(v).cuda() for k,v in arrays.items()}
    fit=partitions['fit'];weights=balanced_weights([rows[i] for i in fit]);reports=[]
    for seed in protocol['seeds']:
        old=next(r for r in previous['reports'] if r['seed']==seed and r['mode']=='reachable_mse')
        weights_path=baseline/f'seed{seed}_reachable_mse.pt'
        if digest(weights_path)!=old['weights_sha256']:raise ValueError('Changed BC weights')
        model=DirectionAdapter(context_dim=arrays['context'].shape[1],mode='bc').cuda()
        model.load_state_dict(torch.load(weights_path,map_location='cuda',weights_only=True));files={};maximum=0.
        for split,ids in partitions.items():
            path=baseline/f'seed{seed}_reachable_mse_{split}.npz'
            if digest(path)!=audited['predictions_sha256'][path.name]:raise ValueError('Changed BC predictions')
            with np.load(path) as z:cached=dict(z)
            if cached['indices'].tolist()!=ids.tolist() or any(not np.array_equal(cached[k],arrays[k][ids]) for k in ('base','target','valid')):
                raise ValueError('BC pairing failure')
            _,pred=evaluate(model,*[data[k][ids] for k in ('context','base','target','valid')],[rows[i] for i in ids])
            difference=float(np.max(np.abs(pred-cached['predicted'])));maximum=max(maximum,difference)
            if difference>1e-6:raise ValueError('BC reloaded under changed inputs')
            files[split]=dict(path=str(path),sha256=digest(path))
        reports.append(dict(seed=seed,mode='bc',metrics=old['metrics'],files=files,weights=str(weights_path),weights_sha256=old['weights_sha256'],baseline_reload_max_abs=maximum))
        for mode in protocol['new_modes']:
            torch.manual_seed(seed)
            model=DirectionAdapter(context_dim=arrays['context'].shape[1],mode='conditioned').cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr']);generator=torch.Generator().manual_seed(seed);history=[]
            for epoch in range(protocol['epochs']):
                sampled=torch.multinomial(weights,len(fit),replacement=True,generator=generator).numpy();total=0.;count=0
                for start in range(0,len(sampled),64):
                    ids=fit[sampled[start:start+64]];c,b,y,v=[data[k][ids] for k in ('context','base','target','valid')]
                    feasible=b+(y-b).clamp(-model.caps,model.caps);optimizer.zero_grad()
                    loss,_=losses(model,c,b,feasible,v,data[mode+'_direction'][ids],data['direction_valid'][ids],auxiliary_weight=.1)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward();optimizer.step();total+=float(loss.detach())*len(ids);count+=len(ids)
                history.append(total/count)
            stem=f'seed{seed}_{mode}';weights_path=run/(stem+'.pt');torch.save(model.state_dict(),weights_path)
            restored=DirectionAdapter(context_dim=arrays['context'].shape[1],mode='conditioned').cuda()
            restored.load_state_dict(torch.load(weights_path,map_location='cuda',weights_only=True));metrics={};files={}
            for split,ids in partitions.items():
                batch=[data[k][ids] for k in ('context','base','target','valid')];meta=[rows[i] for i in ids]
                metrics[split],pred=evaluate(model,*batch,meta);_,again=evaluate(restored,*batch,meta)
                if not np.array_equal(pred,again):raise ValueError('Reload discrepancy')
                with torch.no_grad():head=model(batch[0],batch[1])[2].cpu().numpy()
                if np.max(np.abs(pred-arrays['base'][ids])-model.caps.cpu().numpy())>1e-6:raise ValueError('Cap violation')
                path=run/f'{stem}_{split}.npz'
                np.savez_compressed(path,predicted=pred,base=arrays['base'][ids],target=arrays['target'][ids],valid=arrays['valid'][ids],indices=ids,predicted_direction=head)
                files[split]=dict(path=str(path),sha256=digest(path))
            reports.append(dict(seed=seed,mode=mode,metrics=metrics,files=files,weights=str(weights_path),weights_sha256=digest(weights_path),train_loss=history,reload='exact'))
            replace_json(run/'progress.json',dict(status='running',reports=reports));print(stem,json.dumps(metrics['inner_held']['expert_mse']),flush=True)
    write_json(run/'result.json',dict(status='complete',reports=reports,
        limits='Six new conditioned fits plus three reused BC fits. Same developer scene split; no final validation, task holdout, new collection, policy rollout or TTA.'))
    print('COMPLETE '+str(run),flush=True)


if __name__=='__main__':main()

"""Matched fixed-budget rotation-demand sampling and preservation pilot."""
import argparse
import hashlib
import json
import signal
import traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .audit import digest
from .audit_pose_capacity import numpy_fk,angles
from .paths import ROOT,new_run,write_json,replace_json
from .pose_adapter import PoseAdapter,TorchFK
from .robot_kinematics import RobotFK
from .train_pose_capacity import fk_endpoints,summarize
from .train_semantic_pilot import balanced_weights

PARENT=ROOT/'outputs/effect_learning/pose_capacity_20261002_045559_185861'
VARIANTS=('demand_bc','demand_pose','demand_pose_preserve')

def read(path):return json.loads(path.read_text())

def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def demand_distribution(rows,rare):
    """Inputs contain fit rows ONLY; group-balanced demand mixture, no task rule."""
    if len(rows)!=len(rare) or not len(rows):raise ValueError('Training rows required')
    if any(r['pilot_split']!='fit' for r in rows):raise ValueError('Held/anchor row in demand sampler')
    normal=balanced_weights(rows).numpy();normal=normal/normal.sum()
    groups=sorted({r['group'] for r,is_rare in zip(rows,rare) if is_rare})
    if not groups:raise ValueError('No existing fit rotation demand')
    counts={g:sum(bool(v) and r['group']==g for r,v in zip(rows,rare)) for g in groups}
    demand=np.array([1/(len(groups)*counts[r['group']]) if v else 0 for r,v in zip(rows,rare)],float)
    weights=.5*normal+.5*demand
    if not np.isclose(weights.sum(),1) or not np.all(weights>0):raise ValueError('Invalid normalized mixture')
    return weights,normal,demand

def checked_parent(path):
    audit=read(path/'audit.json');protocol=read(path/'protocol.json');result=read(path/'result.json')
    if not audit['passed'] or result['status']!='complete':raise ValueError('Audited parent required')
    for f,sha in audit['source_sha256'].items():
        if digest(Path(f))!=sha:raise ValueError('Changed parent artifact '+f)
    for f,sha in protocol['source_sha256'].items():
        if digest(Path(f))!=sha:raise ValueError('Changed parent source '+f)
    prepared=Path(protocol['prepared'])
    if digest(prepared/'data.npz')!=protocol['data_sha256'] or digest(prepared/'manifest.json')!=protocol['prepared_manifest_sha256']:raise ValueError('Changed prepared data')
    return protocol,result

def prepare(parent):
    prior,result=checked_parent(parent);prepared=Path(prior['prepared']);rows=prior['rows'];fit=np.array([i for i,r in enumerate(rows) if r['pilot_split']=='fit'])
    with np.load(prepared/'data.npz') as z:arrays={k:z[k][fit].copy() for k in ('base','target','valid')}
    config=read(Path(prior['config']));bots=[RobotFK(config,s) for s in ('left','right')]
    def endpoints(x):return np.stack([numpy_fk(r,x[:,prior['offsets'],start:start+6].astype(float)) for r,start in zip(bots,(0,7))],2)
    base_angle=angles(endpoints(arrays['base']),endpoints(arrays['target']));valid=np.broadcast_to(arrays['valid'][:,prior['offsets'],None],base_angle.shape)
    rare=np.any(valid&(base_angle>=45),(1,2));fitrows=[rows[i] for i in fit];weights,normal,demand=demand_distribution(fitrows,rare)
    run=new_run('rotation_curriculum',group=('effect_learning',));print(run,flush=True)
    np.savez_compressed(run/'sampling.npz',fit_indices=fit,rare_rows=rare,fit_base_angles=base_angle,fit_valid=valid,weights=weights,normal=normal,demand=demand)
    reused=[]
    for r in result['reports']:
        if r['variant'] not in ('proposal_bc','proposal_pose'):continue
        files=[parent/f"{r['variant']}_s{r['seed']}.pt",*[parent/f"{r['variant']}_s{r['seed']}_{p}.npz" for p in ('inner_held','anchor_held')]]
        reused.append(dict(seed=r['seed'],variant='uniform_bc' if r['variant']=='proposal_bc' else 'uniform_pose',parent_variant=r['variant'],source=str(parent),sha256={str(f):digest(f) for f in files}))
    if len(reused)!=6:raise ValueError('Six reused uniform controls required')
    sourcefiles=[parent/'audit.json',parent/'protocol.json',parent/'result.json',parent/'independent_metrics.json',parent/'fit_capacity.json',prepared/'manifest.json',prepared/'data.npz',prepared/'preprocess.npz',Path(prior['config']),Path(__file__),ROOT/'cx_tta/pose_adapter.py',ROOT/'cx_tta/train_pose_capacity.py',ROOT/'cx_tta/train_semantic_pilot.py']
    write_json(run/'protocol.json',dict(status='frozen_before_training',parent=str(parent),prepared=str(prepared),config=prior['config'],rows=rows,context_dim=157,seeds=[0,1,2],epochs=10,lr=3e-4,batch_size=64,draws_per_epoch=len(fit),variants=VARIANTS,reused=reused,offsets=prior['offsets'],pose_loss_weight=1.,preservation_weight=1.,sampling_sha256=digest(run/'sampling.npz'),source_sha256={str(f):digest(f) for f in sourcefiles},
        sampling='0.5 original task/episode-balanced fit + 0.5 group-balanced fit rows with any valid arm/offset >=45deg; target-derived supervision only; identical demand draws per seed across new variants',
        preservation='Success source anchor_fit rows only, original task/episode balancing, independently seeded draws. Entire original base action retained as preservation target; no anchor_held use. No labels enter forward.',
        gates=dict(fit_large_mean_reduction_min=.5,fit_large_fraction_under_15deg_min=.5,held_large_mean_reduction_min=.2,anchor_held_angle_max_deg=1.,preserve_anchor_mse_ratio_max=.5),
        gate_scope='All supported tasks and all three seeds. Coarse offline learning-interface gates only; even passing does not establish recovery/task success. Existing inner-held is a reused developer split, not a new independent test.',
        comparisons=['uniform_bc vs uniform_pose: reuse parent final weights/outputs','uniform_bc vs demand_bc: sampling change','uniform_pose vs demand_pose: sampling change with SAME original all-valid SO(3) loss','demand_bc vs demand_pose: rotation loss','demand_pose vs demand_pose_preserve: successful-action preservation'],
        forward_inputs='Same shared observable current/causal-history context, base actions and time. No task ID, simulator truth, teacher target, rare flag or future input.',
        selection='Final epoch 10 only; no early stop/search/selection; fixed step and draw count. Proposal output deliberately changes command support; no original-cap claim.',
        scope='Bounded offline learning-interface pilot on existing data, no simulation, collection, backbone LoRA or TTA. Original formal validation groups untouched.'))
    print('PREREGISTERED',run,dict(rare_rows=int(rare.sum()),fit_rows=len(fit),rare_groups=len({r['group'] for r,v in zip(fitrows,rare) if v}),expected_rare_fraction=float(weights[rare].sum())),flush=True)

def train(run,device):
    protocol=read(run/'protocol.json')
    if protocol['status']!='frozen_before_training' or (run/'result.json').exists():raise ValueError('Fresh preregistered run required')
    for f,sha in protocol['source_sha256'].items():
        if digest(Path(f))!=sha:raise ValueError('Changed source '+f)
    if digest(run/'sampling.npz')!=protocol['sampling_sha256']:raise ValueError('Changed sampling plan')
    with np.load(run/'sampling.npz') as z:sampling={k:z[k].copy() for k in z.files}
    prepared=Path(protocol['prepared']);rows=protocol['rows']
    with np.load(prepared/'data.npz') as z:arrays={k:z[k].copy() for k in z.files}
    if arrays['context'].shape[1]!=protocol['context_dim']:raise ValueError('Feature dimension changed')
    parts={s:np.array([i for i,r in enumerate(rows) if r['pilot_split']==s]) for s in ('fit','inner_held','anchor_fit','anchor_held')};fit=parts['fit'];anchors=parts['anchor_fit']
    if not np.array_equal(fit,sampling['fit_indices']):raise ValueError('Fit split mismatch')
    groups={s:{rows[i]['group'] for i in ids} for s,ids in parts.items()}
    if groups['fit']&groups['inner_held'] or groups['anchor_fit']&groups['anchor_held']:raise ValueError('Group overlap')
    bots=[RobotFK(read(Path(protocol['config'])),s) for s in ('left','right')];robots=[TorchFK(r).to(device).float() for r in bots]
    data={k:torch.as_tensor(v,device=device) for k,v in arrays.items()};weights=torch.from_numpy(sampling['weights']);aw=balanced_weights([rows[i] for i in anchors]);reports=[]
    write_json(run/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,device=device,gpu=None if device=='cpu' else torch.cuda.get_device_name(0),threads=torch.get_num_threads(),deterministic=torch.are_deterministic_algorithms_enabled()))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Curriculum pilot exceeded 1200 seconds')));signal.alarm(1200)
    try:
        with torch.no_grad():target_end=fk_endpoints(data['target'],robots)
        for seed in protocol['seeds']:
            for variant in protocol['variants']:
                torch.manual_seed(seed);model=PoseAdapter(protocol['context_dim'],proposal=True).to(device);initial=state_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr']);gen=torch.Generator().manual_seed(seed);agen=torch.Generator().manual_seed(seed+123);history=[];draws=[];anchor_draws=[]
                with torch.no_grad():
                    if not torch.equal(model(data['context'][:2],data['base'][:2]),data['base'][:2]):raise ValueError('Zero head changed base')
                for epoch in range(protocol['epochs']):
                    sampled=torch.multinomial(weights,len(fit),replacement=True,generator=gen);draws.append(fit[sampled.numpy()]);losses=[];bc_values=[];pose_values=[];preserve_values=[];adraw=[];model.train()
                    for start in range(0,len(fit),protocol['batch_size']):
                        ids=torch.as_tensor(fit[sampled[start:start+protocol['batch_size']].numpy()],device=device);pred=model(data['context'][ids],data['base'][ids]);valid=data['valid'][ids]
                        bc=((((pred-data['target'][ids])/model.caps).square().mean(-1)*valid).sum(-1)/valid.sum(-1).clamp(min=1)).mean();pose=bc*0;preserve=bc*0
                        if variant!='demand_bc':
                            pe=fk_endpoints(pred,robots);te=target_end[ids];mask=valid[:,protocol['offsets'],None].expand(-1,-1,2);chord=(pe[...,:3,:3]-te[...,:3,:3]).square().sum((-2,-1))/2
                            pose=((chord/.01*mask).sum((1,2))/mask.sum((1,2)).clamp(min=1)).mean()
                        if variant.endswith('_preserve'):
                            ai=anchors[torch.multinomial(aw,len(ids),replacement=True,generator=agen).numpy()];adraw.extend(ai.tolist());ai=torch.as_tensor(ai,device=device)
                            preserved=model(data['context'][ai],data['base'][ai]);preserve=((preserved-data['base'][ai])/model.caps).square().mean()
                        loss=bc+protocol['pose_loss_weight']*pose+protocol['preservation_weight']*preserve
                        if not torch.isfinite(loss):raise ValueError('Nonfinite curriculum loss')
                        opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()));bc_values.append(float(bc.detach()));pose_values.append(float(pose.detach()));preserve_values.append(float(preserve.detach()))
                    if adraw:anchor_draws.append(adraw)
                    history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),bc=float(np.mean(bc_values)),pose=float(np.mean(pose_values)),preserve=float(np.mean(preserve_values)),rare_draws=int(sampling['rare_rows'][sampled.numpy()].sum())))
                    replace_json(run/'progress.json',dict(seed=seed,variant=variant,epoch=epoch+1,completed=len(reports)));print('EPOCH',seed,variant,epoch+1,history[-1],flush=True)
                np.save(run/f'{variant}_s{seed}_draws.npy',np.stack(draws))
                if anchor_draws:np.save(run/f'{variant}_s{seed}_anchor_draws.npy',np.array(anchor_draws))
                model.eval();checkpoint=run/f'{variant}_s{seed}.pt';torch.save(model.state_dict(),checkpoint);restored=PoseAdapter(protocol['context_dim'],proposal=True).to(device);restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));restored.eval();metrics={};files=[]
                for part,indices in parts.items():
                    pred=[]
                    with torch.no_grad():
                        for start in range(0,len(indices),64):
                            ids=torch.as_tensor(indices[start:start+64],device=device);out=model(data['context'][ids],data['base'][ids]);again=restored(data['context'][ids],data['base'][ids])
                            if not torch.equal(out,again):raise ValueError('Reload mismatch')
                            pred.append(out.cpu().numpy())
                    pred=np.concatenate(pred);base=arrays['base'][indices];target=arrays['target'][indices];valid=arrays['valid'][indices]
                    if not np.isfinite(pred).all():raise ValueError('Invalid predictions')
                    metrics[part],geometry=summarize(pred,base,target,valid,[rows[i] for i in indices],robots)
                    file=run/f'{variant}_s{seed}_{part}.npz';np.savez_compressed(file,predicted=pred,base=base,target=target,valid=valid,indices=indices,**geometry);files.append(dict(file=str(file),sha256=digest(file)))
                write_json(run/f'{variant}_s{seed}_history.json',history)
                reports.append(dict(seed=seed,variant=variant,parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,checkpoint_sha256=digest(checkpoint),files=files,metrics=metrics,reload='exact',updates=protocol['epochs']*int(np.ceil(len(fit)/protocol['batch_size']))))
                print('VARIANT_COMPLETE',seed,variant,json.dumps({p:{t:v['large_angle_deg'] for t,v in metrics[p].items()} for p in ('fit','inner_held')}),flush=True)
        write_json(run/'result.json',dict(status='complete',reports=reports,reused=protocol['reused'],scope=protocol['scope']));replace_json(run/'progress.json',dict(phase='complete',trained_variants=len(reports),reused_variants=6));print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed=reports));raise
    finally:signal.alarm(0)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--parent',type=Path,default=PARENT);p.add_argument('--prepare-only',action='store_true');p.add_argument('--run',type=Path);p.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=p.parse_args();torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if a.prepare_only:prepare(a.parent.resolve())
    elif a.run is None:p.error('--run required after preregistration')
    else:train(a.run.resolve(),a.device)

if __name__=='__main__':main()

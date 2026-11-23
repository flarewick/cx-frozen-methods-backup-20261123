"""Separate finite-budget small-set fitting, interference and observable representation.

This is a capacity diagnostic, not a deployment or generalization experiment.
No privileged information is supplied to the adapter forward pass.
"""
import argparse
import json
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .audit import digest
from .audit_pose_capacity import numpy_fk, angles
from .paths import ROOT, new_run, write_json, replace_json
from .pose_adapter import PoseAdapter, TorchFK
from .robot_kinematics import RobotFK
from .train_pose_capacity import CACHE, OFFSETS, fk_endpoints
from .train_semantic_pilot import load_features, balanced_weights

PARENT = ROOT/'outputs/effect_learning/rotation_curriculum_20261002_062504_700870'
CONDITIONS = {
    'semantic_rare': ('semantic', ['rare'], [140, 700, 1400]),
    'coarse_rare': ('coarse', ['rare'], [140, 700, 1400]),
    'semantic_low': ('semantic', ['low'], [140, 700]),
    'semantic_mixed': ('semantic', ['rare', 'low'], [140, 700]),
}

def read(p):
    return json.loads(p.read_text())

def checked_sources(protocol):
    for file, sha in protocol['source_sha256'].items():
        if digest(Path(file)) != sha:
            raise ValueError('Changed source '+file)

def prepare():
    if not read(PARENT/'audit.json')['passed']:
        raise ValueError('Audited curriculum required')
    for name in ('feature_audit.json', 'checkpoint_audit.json'):
        if read(CACHE/name)['status'] != 'passed':
            raise ValueError('Audited feature cache required')
    prior = read(PARENT/'protocol.json')
    checked_sources(prior)
    plan, rows, arrays, contexts, preprocessing = load_features(CACHE)
    if rows != prior['rows']:
        raise ValueError('Changed data rows')
    prepared = Path(prior['prepared'])
    with np.load(prepared/'data.npz') as z:
        for k in ('base', 'target', 'valid'):
            if not np.array_equal(z[k], arrays[k]):
                raise ValueError('Changed existing data '+k)
        if not np.array_equal(z['context'], contexts['semantic']):
            raise ValueError('Existing semantic preprocessing changed')
    with np.load(PARENT/'sampling.npz') as z:
        fit = z['fit_indices']; rare = fit[z['rare_rows']]
        valid = z['fit_valid']; angle = z['fit_base_angles']
        low_mask = np.all(~valid | (angle < 5), axis=(1,2))
    # Deterministic task/group round robin over fit-only low-demand rows.
    tasks = sorted({rows[i]['task'] for i in rare})
    pools = {}
    for task in tasks:
        ids = [int(i) for i in fit[low_mask] if rows[i]['task'] == task]
        groups = sorted({rows[i]['group'] for i in ids})
        queues = {g:sorted([i for i in ids if rows[i]['group']==g],
                          key=lambda i:(rows[i]['frame'],i)) for g in groups}
        pools[task] = []
        while any(queues.values()):
            for g in groups:
                if queues[g]: pools[task].append(queues[g].pop(0))
    low = []
    while len(low) < len(rare):
        for task in tasks:
            if len(low)==len(rare): break
            if not pools[task]: raise ValueError('Insufficient low-demand controls')
            low.append(pools[task].pop(0))
    low = np.array(sorted(low))
    if len(rare)!=17 or len({rows[i]['group'] for i in rare})!=6:
        raise ValueError('Unexpected existing rotation-demand coverage')
    if any(rows[i]['pilot_split']!='fit' for i in np.r_[rare,low]):
        raise ValueError('Training split leakage')
    run = new_run('fit_budget', group=('effect_learning',)); print(run, flush=True)
    np.savez_compressed(run/'data.npz', semantic=contexts['semantic'], coarse=contexts['coarse'],
                        base=arrays['base'], target=arrays['target'], valid=arrays['valid'],
                        rare=rare, low=low)
    for name, values in preprocessing.items():
        np.savez_compressed(run/f'{name}_preprocess.npz', **values)
    sources = [PARENT/'protocol.json', PARENT/'audit.json', PARENT/'sampling.npz',
               CACHE/'manifest.json', CACHE/'result.json', CACHE/'feature_audit.json',
               CACHE/'checkpoint_audit.json', prepared/'manifest.json', prepared/'data.npz',
               Path(prior['config']), Path(__file__), ROOT/'cx_tta/pose_adapter.py',
               ROOT/'cx_tta/train_pose_capacity.py', ROOT/'cx_tta/train_semantic_pilot.py']
    sources.extend(Path(r['file']) for r in read(CACHE/'result.json')['records'])
    write_json(run/'protocol.json', dict(status='frozen_before_training',
        parent=str(PARENT), cache=str(CACHE), config=prior['config'], rows=rows,
        conditions=CONDITIONS, seeds=[0,1,2], lr=3e-4, offsets=OFFSETS,
        context_dim=157, data_sha256=digest(run/'data.npz'),
        source_sha256={str(f):digest(f) for f in sources},
        preprocessing_sha256={n:digest(run/f'{n}_preprocess.npz') for n in contexts},
        optimizer='AdamW default settings; full deterministic small-set batch per update',
        loss='Same normalized valid-step native BC + all-valid endpoint chordal SO(3)/0.01. '
             'Within each component original task/group balancing. Mixed=rare_loss+low_loss, '
             'so rare coefficient/exposure per update is unchanged, not halved.',
        representations='Frozen RGB/state/instruction joint-prefix PCA64 versus existing RGB-grid PCA64; '
             'same 157-dimensional causal context, same 40718 parameters. PCA/scaler expert fit only. '
             'Comparison tests two observable representations; it does not isolate language, geometry or PCA rank.',
        input_contract='Observed current/past RGB-derived context and command state, base actions, '
             'chunk time only. No task ID, label, sample index, rare flag, future, goal or GT in forward.',
        gates=dict(rare_mean_max_deg=3., rare_fraction_under_3deg_min=.9,
                   rare_fraction_under_15deg_min=.9, low_mean_max_deg=1.),
        selection='All preselected snapshots reported. No held selection, hyperparameter search or promotion. '
                  '140/700/1400 counts compare optimizer budgets, not equal data exposure to previous 856-row epochs.',
        scope='Small-set memorization/finite-budget and representation diagnostic on existing fit groups. '
              'Original inner-held only reported at final snapshot as reused developer data; '
              'no task success, action safety or generalization claim. Proposal changes original action support.'))
    print('PREREGISTERED',run,'rare',rare.tolist(),'low',low.tolist(),flush=True)

def metrics(pred, base, target, valid, indices, rows, bots):
    def ends(x):
        return np.stack([numpy_fk(r,x[:,OFFSETS,s:s+6].astype(float))
                         for r,s in zip(bots,(0,7))],axis=2)
    pp,bp,tp = [ends(x) for x in (pred,base,target)]
    a,b = angles(pp,tp),angles(bp,tp)
    m = np.broadcast_to(valid[:,OFFSETS,None],a.shape); large = m & (b>=45)
    output = {}
    for task in sorted({rows[i]['task'] for i in indices}):
        group_metrics = []
        for group in sorted({rows[i]['group'] for i in indices if rows[i]['task']==task}):
            ids=np.array([j for j,i in enumerate(indices) if rows[i]['group']==group]); v=valid[ids]
            ma=m[ids]; la=large[ids]; aa=a[ids]; bb=b[ids]
            group_metrics.append(dict(angle_deg=float(aa[ma].mean()),
                base_angle_deg=float(bb[ma].mean()),
                native_mse=float(((pred[ids]-target[ids])**2).mean(-1)[v].mean()),
                large_angle_deg=None if not la.any() else float(aa[la].mean()),
                large_base_angle_deg=None if not la.any() else float(bb[la].mean()),
                under_3deg=None if not la.any() else float((aa[la]<3).mean()),
                under_15deg=None if not la.any() else float((aa[la]<15).mean())))
        output[task] = {k:float(np.mean(values)) if
                       (values:=[g[k] for g in group_metrics if g[k] is not None]) else None
                       for k in group_metrics[0]}
    return output,dict(angle=a,base_angle=b,mask=m,large=large)

def train(run, device):
    p=read(run/'protocol.json'); checked_sources(p)
    if (run/'result.json').exists() or digest(run/'data.npz')!=p['data_sha256']:
        raise ValueError('Fresh frozen data required')
    with np.load(run/'data.npz') as z: data_np={k:z[k].copy() for k in z.files}
    data={k:torch.as_tensor(data_np[k],device=device) for k in ('semantic','coarse','base','target','valid')}
    rows=p['rows']; subsets={k:data_np[k] for k in ('rare','low')}
    subsets.update({k:np.array([i for i,r in enumerate(rows) if r['pilot_split']==k])
                    for k in ('inner_held','anchor_held')})
    botdefs=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')]
    bots=[TorchFK(r).to(device).float() for r in botdefs]
    weights={k:balanced_weights([rows[i] for i in subsets[k]]).to(device).float()
             for k in ('rare','low')}
    weights={k:v/v.sum() for k,v in weights.items()}
    ids={k:torch.as_tensor(v,device=device) for k,v in subsets.items()}
    with torch.no_grad(): target_ends=fk_endpoints(data['target'],bots)
    reports=[]; started=time.monotonic()
    write_json(run/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        device=device,gpu=None if device=='cpu' else torch.cuda.get_device_name(0),
        deterministic=torch.are_deterministic_algorithms_enabled(),threads=torch.get_num_threads()))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fit diagnostic exceeded 1200s')))
    signal.alarm(1200)
    try:
        for seed in p['seeds']:
            for condition,(representation,components,snapshots) in p['conditions'].items():
                torch.manual_seed(seed); model=PoseAdapter(p['context_dim'],proposal=True).to(device)
                opt=torch.optim.AdamW(model.parameters(),lr=p['lr']); history=[]
                if sum(x.numel() for x in model.parameters())!=40718:raise ValueError('Parameter budget changed')
                with torch.no_grad():
                    if not torch.equal(model(data[representation][ids['rare']],data['base'][ids['rare']]),data['base'][ids['rare']]):
                        raise ValueError('Zero-initialized head changed base')
                for update in range(1,max(snapshots)+1):
                    opt.zero_grad(); losses={}; total=0
                    for component in components:
                        ci=ids[component]; valid=data['valid'][ci]
                        pred=model(data[representation][ci],data['base'][ci])
                        bc=((((pred-data['target'][ci])/model.caps).square().mean(-1)*valid).sum(-1)
                            /valid.sum(-1).clamp(min=1))
                        ep=fk_endpoints(pred,bots);mask=valid[:,OFFSETS,None].expand(-1,-1,2)
                        chord=(ep[...,:3,:3]-target_ends[ci,...,:3,:3]).square().sum((-2,-1))/2
                        pose=(chord/.01*mask).sum((1,2))/mask.sum((1,2)).clamp(min=1)
                        loss=((bc+pose)*weights[component]).sum();total=total+loss
                        losses[component]=dict(bc=float((bc*weights[component]).sum().detach()),
                                               pose=float((pose*weights[component]).sum().detach()))
                    if not torch.isfinite(total):raise ValueError('Nonfinite diagnostic loss')
                    total.backward(); opt.step()
                    if update==1 or update%70==0:
                        history.append(dict(update=update,components=losses,total=float(total.detach())))
                    if update in snapshots:
                        tag=f'{condition}_s{seed}_u{update}';model.eval()
                        checkpoint=run/f'{tag}.pt';torch.save(model.state_dict(),checkpoint)
                        restored=PoseAdapter(p['context_dim'],proposal=True).to(device)
                        restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));restored.eval()
                        summary={}; saved=[]
                        for part in ('rare','low','inner_held','anchor_held'):
                            if part not in ('rare','low') and update!=max(snapshots):continue
                            ix=subsets[part]; values=[]
                            with torch.no_grad():
                                for start in range(0,len(ix),64):
                                    bi=ids[part][start:start+64]; out=model(data[representation][bi],data['base'][bi])
                                    if not torch.equal(out,restored(data[representation][bi],data['base'][bi])):raise ValueError('Reload mismatch')
                                    values.append(out.cpu().numpy())
                            pred=np.concatenate(values)
                            summary[part],geom=metrics(pred,*[data_np[k][ix] for k in ('base','target','valid')],ix,rows,botdefs)
                            dest=run/f'{tag}_{part}.npz'; np.savez_compressed(dest,indices=ix,predicted=pred,
                                **{k:data_np[k][ix] for k in ('base','target','valid')},**geom)
                            saved.append(dict(path=str(dest),sha256=digest(dest)))
                        reports.append(dict(condition=condition,seed=seed,update=update,parameters=40718,
                            checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),files=saved,metrics=summary,reload='exact'))
                        replace_json(run/'progress.json',dict(condition=condition,seed=seed,update=update,
                            completed_snapshots=len(reports),elapsed_seconds=time.monotonic()-started))
                        print('SNAPSHOT',tag,json.dumps(summary['rare']),flush=True)
                        model.train()
                write_json(run/f'{condition}_s{seed}_history.json',history)
        write_json(run/'result.json',dict(status='complete',reports=reports,elapsed_seconds=time.monotonic()-started,
            scope=p['scope'])); print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed=reports));raise
    finally:signal.alarm(0)

def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('--prepare-only',action='store_true')
    a.add_argument('--run',type=Path);a.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=a.parse_args();torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if args.prepare_only:prepare()
    elif args.run is None:a.error('--run required')
    else:train(args.run.resolve(),args.device)

if __name__=='__main__':main()

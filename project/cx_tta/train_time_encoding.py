"""Matched-capacity relative-time encoding pilot on existing recovery data."""
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
from .paths import ROOT,new_run,write_json,replace_json
from .pose_adapter import TorchFK
from .recovery_temporal import temporal_loss
from .time_encoding import TimeEncodedAdapter
from .check_time_encoding import check_interface
from .robot_kinematics import RobotFK
from .train_rotation_curriculum import state_hash
from .recovery_fit_metrics import metrics,recovery_gate

CONDITIONS={f'{encoding}_{regime}':dict(representation='semantic',components=['recovery','ordinary','anchor_fit'],
    loss='se3',temporal=regime=='combined',envelope=regime=='combined',encoding=encoding)
    for regime in ('se3','combined') for encoding in ('repeat','fourier')}

def read(p):return json.loads(p.read_text())

def end_torch(x,robots):return torch.stack([r(x[...,s:s+6]) for r,s in zip(robots,(0,7))],2)

def prepare(time_probe):
    probe=read(time_probe/'result.json');source=Path(probe['source']);p=read(source/'protocol.json')
    if not probe['passed'] or not read(source/'audit.json')['passed'] or not read(source/'temporal_audit.json')['passed']:
        raise ValueError('Audited source and time localization required')
    for file,sha in {**probe['source_sha256'],**p['source_sha256']}.items():
        if digest(Path(file))!=sha:raise ValueError('Changed source '+file)
    if digest(source/'data.npz')!=p['data_sha256'] or digest(source/'weights.npz')!=p['weights_sha256']:
        raise ValueError('Changed original arrays')
    with np.load(source/'data.npz') as z:data={k:z[k].copy() for k in z.files}
    with np.load(source/'weights.npz') as z:w=z['baseline'].copy()
    run=new_run('time_encoding',group=('effect_learning',));print(run,flush=True)
    np.savez_compressed(run/'data.npz',**data)
    np.savez_compressed(run/'weights.npz',**{c:w for c in CONDITIONS})
    files=[source/'protocol.json',source/'audit.json',source/'temporal_audit.json',source/'result.json',
        source/'data.npz',source/'weights.npz',time_probe/'result.json',Path(__file__),
        ROOT/'cx_tta/time_encoding.py',ROOT/'cx_tta/check_time_encoding.py',ROOT/'cx_tta/probe_recovery_time.py',
        ROOT/'cx_tta/audit_time_encoding.py',ROOT/'cx_tta/audit_time_encoding_dense.py',
        ROOT/'scripts/train_time_encoding.sh']
    protocol={**p,'conditions':CONDITIONS,'parameters':41742,'time_probe':str(time_probe),
        'reference_source':str(source),'reference_conditions':['baseline','combined'],
        'data_sha256':digest(run/'data.npz'),'weights_sha256':digest(run/'weights.npz'),
        'source_sha256':{**p['source_sha256'],**{str(f):digest(f) for f in files}},
        'time_encoding':dict(original_scalar=True,additional_channels=8,frequencies=[1,2,4,8],
            scaling='Each extra channel centered and RMS-normalized on all32 known chunk positions, scaled by1/sqrt(8).',
            initialization='Original common weights retained; eight new input columns zero-initialized. All same-shaped '
                'conditions have identical initial weights per seed; all columns remain trainable.',
            control='Repeat the centered linear clock eight times with matching width/channel RMS. No new raw information; '
                'Fourier is a nonlinear re-expression of the same observable clock. Repeat has redundant columns.',
            validity='Clock always uses full base-chunk length32, never target valid tail, phase, native count or scene ID.'),
        'comparisons':['repeat_se3 vs fourier_se3: nonlinear time representation with same parameter allocation/update budget',
            'repeat_combined vs fourier_combined: same encoding contrast with fixed S1t temporal/envelope ingredients',
            'S1t baseline/combined reused as historical scalar-clock references, not same-width controls'],
        'scope':'S1u developer offline time-representation diagnostic. Same six recovery blocks/two studied scenes,17 ordinary '
            'rows and18 anchors;12 new models,three seeds,140/700/1400 updates. Same41742 allocated parameters across '
            'four new conditions (S1t40718). No scene-independent recovery test, simulation, collection, LoRA, TTA, '
            'deployment or task-success claim. Unbounded arm proposal still changes original residual support.'}
    write_json(run/'protocol.json',protocol)
    print('PREREGISTERED',run,'models',12,'snapshots',36,'parameters',41742,flush=True)

def train(run,device):
    p=read(run/'protocol.json')
    if (run/'result.json').exists():raise ValueError('Fresh run required')
    for f,sha in p['source_sha256'].items():
        if digest(Path(f))!=sha:raise ValueError('Changed source '+f)
    if digest(run/'data.npz')!=p['data_sha256'] or digest(run/'weights.npz')!=p['weights_sha256']:raise ValueError('Changed frozen arrays')
    with np.load(run/'data.npz') as z:arrays={k:z[k].copy() for k in z.files}
    with np.load(run/'weights.npz') as z:weight_np={k:z[k].copy() for k in z.files}
    data={k:torch.as_tensor(v,device=device) for k,v in arrays.items()}
    definitions=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')]
    robots=[TorchFK(r).to(device).float() for r in definitions]
    with torch.no_grad():target_end=end_torch(data['target'],robots)
    reports=[];started=time.monotonic()
    write_json(run/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,device=device,
        gpu=None if device=='cpu' else torch.cuda.get_device_name(0),deterministic=torch.are_deterministic_algorithms_enabled()))
    # Validate the *actual dense* BC/pose loss masking before any optimizer update.
    def per_row_loss(pred,target,valid,te,model,kind):
        bc=(((pred-target)/model.caps).square().mean(-1)*valid).sum(-1)/valid.sum(-1).clamp(min=1)
        rot=bc*0;pos=bc*0
        if kind!='bc':
            pe=end_torch(pred,robots);mask=valid[:,:,None].expand(-1,-1,2);denom=mask.sum((1,2)).clamp(min=1)
            rot=(((pe[...,:3,:3]-te[...,:3,:3]).square().sum((-2,-1))/2/p['rotation_scale_rad']**2)*mask).sum((1,2))/denom
            if kind=='se3':pos=(((pe[...,:3,3]-te[...,:3,3]).square().sum(-1)/p['position_scale_m']**2)*mask).sum((1,2))/denom
        return bc+rot+pos,bc,rot,pos
    write_json(run/'loss_interface_check.json',check_interface(p,data,device,per_row_loss,robots))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Recovery-fit pilot exceeded1800s')));signal.alarm(1800)
    try:
        for seed in p['seeds']:
            for condition,setting in p['conditions'].items():
                torch.manual_seed(seed);model=TimeEncodedAdapter(p['context_dim'],encoding=setting['encoding'],envelope=setting['envelope']).to(device)
                initial=state_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=p['lr']);history=[]
                if sum(v.numel() for v in model.parameters())!=p['parameters']:raise ValueError('Parameter mismatch')
                idx_np=np.flatnonzero(weight_np[condition]);idx=torch.as_tensor(idx_np,device=device)
                weights=torch.as_tensor(weight_np[condition][idx_np],device=device);rep=setting['representation']
                local={int(g):i for i,g in enumerate(idx_np)}
                pairs=torch.tensor([[local[v['before']],v['last'],local[v['after']]] for v in p['seam_pairs']],device=device)
                pair_weights=torch.tensor(p['seam_weights'],device=device,dtype=torch.float32)
                with torch.no_grad():
                    if not torch.equal(model(data[rep][idx],data['base'][idx]),data['base'][idx]):raise ValueError('Zero-head mismatch')
                for update in range(1,max(p['updates'])+1):
                    opt.zero_grad();pred=model(data[rep][idx],data['base'][idx])
                    total,bc,rot,pos=per_row_loss(pred,data['target'][idx],data['valid'][idx],target_end[idx],model,setting['loss'])
                    loss=(total*weights).sum()
                    within=loss*0;seam=loss*0
                    if setting['temporal']:
                        within,seam=temporal_loss(pred,data['target'][idx],data['valid'][idx],model.caps,weights,pairs,pair_weights)
                        loss=loss+within+seam
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward();opt.step()
                    if update==1 or update%70==0:
                        history.append(dict(update=update,loss=float(loss.detach()),bc=float((bc*weights).sum().detach()),
                            rotation=float((rot*weights).sum().detach()),position=float((pos*weights).sum().detach()),
                            temporal_within=float(within.detach()),temporal_seam=float(seam.detach())))
                    if update in p['updates']:
                        tag=f'{condition}_s{seed}_u{update}';checkpoint=run/f'{tag}.pt';model.eval();torch.save(model.state_dict(),checkpoint)
                        restored=TimeEncodedAdapter(p['context_dim'],encoding=setting['encoding'],envelope=setting['envelope']).to(device);restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));restored.eval()
                        summaries={};files=[]
                        for part,ix in p['parts'].items():
                            if part in ('inner_held','anchor_held') and update!=max(p['updates']):continue
                            ix=np.array(ix);values=[]
                            with torch.no_grad():
                                for start in range(0,len(ix),64):
                                    bi=torch.as_tensor(ix[start:start+64],device=device);out=model(data[rep][bi],data['base'][bi])
                                    if not torch.equal(out,restored(data[rep][bi],data['base'][bi])):raise ValueError('Checkpoint reload mismatch')
                                    values.append(out.cpu().numpy())
                            predicted=np.concatenate(values)
                            if not np.isfinite(predicted).all():raise ValueError('Nonfinite prediction')
                            summaries[part],geom=metrics(predicted,arrays['base'][ix],arrays['target'][ix],arrays['valid'][ix],[p['rows'][i] for i in ix],definitions)
                            file=run/f'{tag}_{part}.npz';np.savez_compressed(file,predicted=predicted,indices=ix,
                                **{k:arrays[k][ix] for k in ('base','target','valid')},**geom)
                            files.append(dict(part=part,path=str(file),sha256=digest(file)))
                        fit_pass=recovery_gate(summaries['recovery'])
                        reports.append(dict(condition=condition,seed=seed,update=update,initial_state_sha256=initial,
                            checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),parameters=p['parameters'],
                            metrics=summaries,files=files,reload='exact',recovery_fit_pass=fit_pass))
                        replace_json(run/'progress.json',dict(condition=condition,seed=seed,update=update,snapshots=len(reports),
                            elapsed_seconds=time.monotonic()-started))
                        print('SNAPSHOT',tag,'fit_pass',fit_pass,json.dumps(summaries['recovery']['tasks']),flush=True)
                        model.train()
                write_json(run/f'{condition}_s{seed}_history.json',history)
        write_json(run/'result.json',dict(status='complete',reports=reports,elapsed_seconds=time.monotonic()-started,scope=p['scope']))
        replace_json(run/'progress.json',dict(phase='complete',models=len(p['conditions'])*len(p['seeds']),snapshots=len(reports)))
        print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_snapshots=len(reports)));raise
    finally:signal.alarm(0)

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--time-probe',type=Path);parser.add_argument('--run',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');args=parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if args.prepare_only:
        if args.time_probe is None:parser.error('--time-probe is required to freeze protocol')
        prepare(args.time_probe.resolve())
    elif args.run is None:parser.error('A preregistered --run is required')
    else:train(args.run.resolve(),args.device)

if __name__=='__main__':main()

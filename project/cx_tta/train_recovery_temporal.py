"""Fixed 2x2 command-difference and identity-preserving gripper-envelope pilot."""
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
from .recovery_temporal import TemporalPoseAdapter,temporal_loss
from .check_recovery_temporal import check_interface
from .robot_kinematics import RobotFK
from .train_rotation_curriculum import state_hash
from .recovery_fit_metrics import metrics,recovery_gate

CONDITIONS={name:dict(representation='semantic',components=['recovery','ordinary','anchor_fit'],
    loss='se3',temporal=temporal,envelope=envelope) for name,temporal,envelope in
    [('baseline',False,False),('temporal',True,False),('gripper',False,True),('combined',True,True)]}

def read(p):return json.loads(p.read_text())

def end_torch(x,robots):return torch.stack([r(x[...,s:s+6]) for r,s in zip(robots,(0,7))],2)

def prepare(command_audit):
    report=read(command_audit/'result.json');source=Path(report['source']);p=read(source/'protocol.json')
    if not report['passed'] or not read(source/'audit.json')['passed']:raise ValueError('Audited sources required')
    for file,sha in report['source_sha256'].items():
        if digest(Path(file))!=sha:raise ValueError('Changed command preflight source '+file)
    for file,sha in p['source_sha256'].items():
        if digest(Path(file))!=sha:raise ValueError('Changed prior training source '+file)
    with np.load(source/'data.npz') as z:data={k:z[k].copy() for k in z.files}
    if digest(source/'data.npz')!=p['data_sha256']:raise ValueError('Changed source data')
    if digest(source/'weights.npz')!=p['weights_sha256']:raise ValueError('Changed source weights')
    with np.load(source/'weights.npz') as z:w=z['semantic_mixed_se3'].copy()
    tasks={pair['task'] for pair in report['pairs']}
    pair_weights=[]
    for pair in report['pairs']:
        ng=len({r['group'] for r in report['pairs'] if r['task']==pair['task']})
        count=sum(r['group']==pair['group'] for r in report['pairs'])
        pair_weights.append(1/(len(tasks)*ng*count))
    run=new_run('recovery_temporal',group=('effect_learning',));print(run,flush=True)
    np.savez_compressed(run/'data.npz',**data)
    np.savez_compressed(run/'weights.npz',**{c:w for c in CONDITIONS})
    files=[source/'protocol.json',source/'audit.json',source/'result.json',source/'data.npz',source/'weights.npz',
        command_audit/'result.json',Path(__file__),ROOT/'cx_tta/recovery_temporal.py',
        ROOT/'cx_tta/check_recovery_temporal.py',ROOT/'cx_tta/audit_recovery_temporal.py',
        ROOT/'cx_tta/audit_recovery_fit.py',ROOT/'scripts/train_recovery_temporal.sh']
    files.extend([RobotFK(read(Path(p['config'])),side).urdf for side in ('left','right')])
    protocol={**p, 'conditions':CONDITIONS, 'data_sha256':digest(run/'data.npz'),
        'weights_sha256':digest(run/'weights.npz'),'baseline_source':str(source),
        'command_audit':str(command_audit),'seam_pairs':report['pairs'],'seam_weights':pair_weights,
        'source_sha256':{**p['source_sha256'],**{str(f):digest(f) for f in files}},
        'temporal_loss':'Add within-block and cross-block native command-difference error, each coefficient1; '
            'normalize by original BC .1arm/.2grip. Within blocks use original component/row weights; '
            'cross-block pairs separately task/scene/boundary balanced. Teacher increments need not be zero.',
        'gripper_decoder':'delta -> room*tanh(delta/room), sign chooses available room to min(base,0) or max(base,1). '
            'At zero choose inward derivative. No additional overshoot beyond base and [0,1]. '
            'Zero/disabled returns raw base exactly; not strict [0,1] for originally out-of-range base.',
        'comparisons':['baseline vs temporal: only teacher-difference supervision',
            'baseline vs gripper: only parameter-free gripper-envelope decoder',
            'temporal vs combined and gripper vs combined: interactions of two factors',
            'baseline replay vs original S1s semantic_mixed_se3: exact saved prediction parity check'],
        'scope':'S1t offline 2x2 interface diagnostic, same six recovery blocks/two studied scenes,17 ordinary rows '
            'and18 anchors as S1s. Same40718 parameters,three seeds,140/700/1400 updates. No held tuning, '
            'independent recovery test, simulation, new collection, LoRA, TTA or task success claim.'}
    write_json(run/'protocol.json',protocol)
    print('PREREGISTERED',run,'conditions',len(CONDITIONS),'models',12,'snapshots',36,flush=True)

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
                torch.manual_seed(seed);model=TemporalPoseAdapter(p['context_dim'],envelope=setting['envelope']).to(device)
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
                        restored=TemporalPoseAdapter(p['context_dim'],envelope=setting['envelope']).to(device);restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));restored.eval()
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
    parser.add_argument('--command-audit',type=Path);parser.add_argument('--run',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');args=parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if args.prepare_only:
        if args.command_audit is None:parser.error('--command-audit is required to freeze protocol')
        prepare(args.command_audit.resolve())
    elif args.run is None:parser.error('A preregistered --run is required')
    else:train(args.run.resolve(),args.device)

if __name__=='__main__':main()

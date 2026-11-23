"""Fixed-budget shared recovery imitation with matched loss/preservation controls."""
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
from .pose_adapter import PoseAdapter,TorchFK
from .robot_kinematics import RobotFK
from .train_semantic_pilot import balanced_weights
from .train_rotation_curriculum import state_hash
from .recovery_context_v2 import FIT,SOURCE,load_context
from .recovery_fit_metrics import metrics,recovery_gate

CONDITIONS={**{f'semantic_{part}_{loss}':dict(representation='semantic',components=parts,loss=loss)
    for part,parts in [('control',['ordinary','anchor_fit']),('recovery',['recovery']),('mixed',['recovery','ordinary','anchor_fit'])]
    for loss in ('bc','so3','se3')},
    'coarse_mixed_se3':dict(representation='coarse',components=['recovery','ordinary','anchor_fit'],loss='se3')}

def read(p):return json.loads(p.read_text())

def end_torch(x,robots):return torch.stack([r(x[...,s:s+6]) for r,s in zip(robots,(0,7))],2)

def prepare(context_run):
    ca,recovery=load_context(context_run);prior=read(FIT/'protocol.json')
    if not read(FIT/'audit.json')['passed']:raise ValueError('Audited fit source required')
    if digest(FIT/'data.npz')!=prior['data_sha256']:raise ValueError('Changed fit data')
    with np.load(FIT/'data.npz') as z:expert={k:z[k].copy() for k in z.files}
    rows=prior['rows']+ca['rows'];n=len(prior['rows'])
    parts=dict(ordinary=expert['low'].tolist(),recovery=list(range(n,n+len(ca['rows']))))
    parts.update({s:[i for i,r in enumerate(rows) if r['pilot_split']==s] for s in ('anchor_fit','inner_held','anchor_held')})
    groups={s:{rows[i]['group'] for i in ix} for s,ix in parts.items()}
    for s in ('recovery','ordinary','anchor_fit'):
        if groups[s]&(groups['inner_held']|groups['anchor_held']):raise ValueError('Developer held group overlap')
    if groups['recovery']&(groups['ordinary']|groups['anchor_fit']):raise ValueError('Recovery/control group overlap')
    if len(parts['ordinary'])!=17 or len(parts['anchor_fit'])!=18 or len(parts['recovery'])!=6:raise ValueError('Unexpected fixed small sets')
    data={k:np.concatenate([expert[k],recovery[k]],axis=0) for k in ('semantic','coarse','base','target','valid')}
    weights={}
    for c,setting in CONDITIONS.items():
        w=np.zeros(len(rows),np.float32)
        for part in setting['components']:
            ix=parts[part];v=balanced_weights([rows[i] for i in ix]).numpy();w[ix]=v/v.sum()
        weights[c]=w
    run=new_run('recovery_fit',group=('effect_learning',));print(run,flush=True)
    np.savez_compressed(run/'data.npz',**data);np.savez_compressed(run/'weights.npz',**weights)
    files=[context_run/'audit.json',context_run/'data.npz',FIT/'protocol.json',FIT/'audit.json',FIT/'data.npz',
           Path(prior['config']),Path(__file__),ROOT/'cx_tta/pose_adapter.py',ROOT/'cx_tta/recovery_fit_metrics.py',
           ROOT/'cx_tta/recovery_context_v2.py',ROOT/'cx_tta/dataset.py',ROOT/'cx_tta/train_semantic_pilot.py']
    write_json(run/'protocol.json',dict(status='frozen_before_training',context_source=str(context_run),
        expert_source=str(FIT),config=prior['config'],rows=rows,parts=parts,conditions=CONDITIONS,
        seeds=[0,1,2],updates=[140,700,1400],lr=3e-4,context_dim=157,parameters=40718,
        rotation_scale_rad=.1,position_scale_m=.02,
        data_sha256=digest(run/'data.npz'),weights_sha256=digest(run/'weights.npz'),
        source_sha256={str(f):digest(f) for f in files},
        forward='Only corrected observable current/causal-history context, recorded base actions and chunk time. '
                'No task ID, recovery flag, target, future input, simulator goal or phase label.',
        losses='BC: valid-step native MSE normalized by .1 arm/.2 grip. SO3: BC plus rotation chordal/.1^2. '
               'SE3: SO3 plus squared position distance/.02^2. All valid steps and both arms; '
               'dense pose supervision differs from previous five-offset diagnostic.',
        weighting='Full fixed small set each update. Each component separately task/scene/row balanced; '
                  'sum component losses, so adding recovery never reduces ordinary/anchor coefficients. '
                  'Per-row valid-step averaging; no padded target supervision. No sampling or loss-weight search.',
        labels='Recovery: actually executed native teacher targets; ordinary: existing17 fit expert rows; '
               'anchor_fit:18 rows from5 successful source scenes, targets equal original base. '
               'Ordinary set was selected at five FK offsets, not guaranteed low-demand at all32 steps.',
        comparisons=['control vs mixed per loss: adding recovery with same ordinary/anchor coefficients',
                     'recovery vs mixed per loss: ordinary and success preservation',
                     'bc vs so3 vs se3 within setting: loss additions',
                     'semantic_mixed_se3 vs coarse_mixed_se3: same width/parameter/budget, different observable representation'],
        selection='Report all fixed snapshots; final1400 for decisions. No held selection or deployment. '
                  'Three initialization seeds are not independent scene trials.',
        gates=dict(recovery='Every scene and every arm: mean angle<=3deg, position<=1cm, '
                   '>=90% endpoints satisfy both; native arm RMSE<=.05rad, grip MAE<=.05, '
                   'velocity-error and cross-block seam RMSE<=.03rad, no URDF or gripper[-.02,1.02] violation.',
                   preservation='Every held anchor scene/arm angle<=1deg, position<=1cm, grip MAE<=.02; '
                   'ordinary fit angle<=1deg and position<=1cm. These are developer diagnostics, not success.'),
        scope='Shared small-adapter fitting on6 recovery blocks from2 studied failure scenes,17 ordinary fit rows '
              'and18 success fit anchors. No independent recovery test; old inner-held/anchor-held reused for '
              'development. No simulation, collection, backbone LoRA, TTA or task success claim. '
              'Unbounded proposal deliberately changes original command support.'))
    print('PREREGISTERED',run,'conditions',len(CONDITIONS),'seeds',3,'snapshots',90,flush=True)

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
    torch.manual_seed(0);check=PoseAdapter(p['context_dim'],proposal=True).to(device)
    rid=torch.tensor(p['parts']['recovery'],device=device);target=data['target'][rid];valid=data['valid'][rid]
    def checked_gradient(t):
        check.zero_grad();out=check(data['semantic'][rid],data['base'][rid])
        losses=per_row_loss(out,t,valid,end_torch(t,robots),check,'se3')[0].mean();losses.backward()
        return losses.detach(),torch.cat([v.grad.flatten() for v in check.parameters()]).detach().clone()
    original_loss,original_grad=checked_gradient(target);changed=target.clone();changed[~valid]=1000
    changed_loss,changed_grad=checked_gradient(changed)
    if not torch.equal(original_loss,changed_loss) or not torch.equal(original_grad,changed_grad):raise ValueError('Dense loss padding leak')
    if not torch.isfinite(original_grad).all() or original_grad.norm()==0:raise ValueError('Invalid training gradient')
    write_json(run/'loss_interface_check.json',dict(passed=True,dense_valid_steps=True,invalid_tail_loss_and_gradient_exact=True,
        parameters=p['parameters'],se3_loss=float(original_loss),finite_nonzero_gradient=True))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Recovery-fit pilot exceeded1800s')));signal.alarm(1800)
    try:
        for seed in p['seeds']:
            for condition,setting in p['conditions'].items():
                torch.manual_seed(seed);model=PoseAdapter(p['context_dim'],proposal=True).to(device)
                initial=state_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=p['lr']);history=[]
                if sum(v.numel() for v in model.parameters())!=p['parameters']:raise ValueError('Parameter mismatch')
                idx_np=np.flatnonzero(weight_np[condition]);idx=torch.as_tensor(idx_np,device=device)
                weights=torch.as_tensor(weight_np[condition][idx_np],device=device);rep=setting['representation']
                with torch.no_grad():
                    if not torch.equal(model(data[rep][idx],data['base'][idx]),data['base'][idx]):raise ValueError('Zero-head mismatch')
                for update in range(1,max(p['updates'])+1):
                    opt.zero_grad();pred=model(data[rep][idx],data['base'][idx])
                    total,bc,rot,pos=per_row_loss(pred,data['target'][idx],data['valid'][idx],target_end[idx],model,setting['loss'])
                    loss=(total*weights).sum()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward();opt.step()
                    if update==1 or update%70==0:
                        history.append(dict(update=update,loss=float(loss.detach()),bc=float((bc*weights).sum().detach()),
                            rotation=float((rot*weights).sum().detach()),position=float((pos*weights).sum().detach())))
                    if update in p['updates']:
                        tag=f'{condition}_s{seed}_u{update}';checkpoint=run/f'{tag}.pt';model.eval();torch.save(model.state_dict(),checkpoint)
                        restored=PoseAdapter(p['context_dim'],proposal=True).to(device);restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));restored.eval()
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
        replace_json(run/'progress.json',dict(phase='complete',models=30,snapshots=len(reports)))
        print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_snapshots=len(reports)));raise
    finally:signal.alarm(0)

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--context',type=Path);parser.add_argument('--run',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');args=parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if args.prepare_only:
        context=args.context or Path(read(SOURCE/'context_contract_erratum.json')['replacement']);prepare(context.resolve())
    elif args.run is None:parser.error('A preregistered --run is required')
    else:train(args.run.resolve(),args.device)

if __name__=='__main__':main()

"""Fixed-budget wide expert coverage with matched recovery-supervision controls."""
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
from .check_coverage import check_interface
from .train_semantic_pilot import balanced_weights
from .audit_local_geometry import joint_bounds
from .robot_kinematics import RobotFK
from .train_rotation_curriculum import state_hash
from .recovery_fit_metrics import metrics,recovery_gate

SOURCE=ROOT/'outputs/effect_learning/time_encoding_20261002_084713_570840'
CONDITIONS={f'{encoding}_{regime}':dict(representation='semantic',
    components=['expert_fit','anchor_fit']+(['recovery'] if regime=='recovery' else []),
    loss='se3',temporal=True,envelope=True,encoding=encoding)
    for regime in ('control','recovery') for encoding in ('repeat','fourier')}

def read(p):return json.loads(p.read_text())

def end_torch(x,robots):return torch.stack([r(x[...,s:s+6]) for r,s in zip(robots,(0,7))],2)

def prepare(source):
    p=read(source/'protocol.json');inventory=read(source/'support_inventory.json')
    if not read(source/'audit.json')['passed'] or not read(source/'time_encoding_audit.json')['passed']:
        raise ValueError('Completed audited S1u source required')
    for file,sha in {**p['source_sha256'],**inventory['source_sha256']}.items():
        if digest(Path(file))!=sha:raise ValueError('Changed source '+file)
    if digest(source/'data.npz')!=p['data_sha256'] or digest(source/'weights.npz')!=p['weights_sha256']:
        raise ValueError('Changed source data/weights')
    with np.load(source/'data.npz') as z:data={k:z[k].copy() for k in z.files}
    parts={**p['parts'],'expert_fit':[i for i,r in enumerate(p['rows']) if r['pilot_split']=='fit']}
    if len(parts['expert_fit'])!=856 or not set(parts['ordinary'])<=set(parts['expert_fit']):
        raise ValueError('Unexpected broad expert fit pool')
    groups={part:{p['rows'][i]['group'] for i in ids} for part,ids in parts.items()}
    for name in ('expert_fit','anchor_fit','recovery'):
        if groups[name]&(groups['inner_held']|groups['anchor_held']):raise ValueError('Held leakage')
    if any(groups[a]&groups[b] for a,b in [('expert_fit','recovery'),('expert_fit','anchor_fit'),('recovery','anchor_fit')]):
        raise ValueError('Training component scene overlap')
    wmap={}
    for condition,setting in CONDITIONS.items():
        weights=np.zeros(len(p['rows']),np.float32)
        for component in setting['components']:
            ids=parts[component];values=balanced_weights([p['rows'][i] for i in ids]).numpy()
            weights[ids]=values/values.sum()
        wmap[condition]=weights
    with np.load(source/'weights.npz') as prior:
        for name,w in wmap.items():
            for part in ('anchor_fit','recovery'):
                if part in CONDITIONS[name]['components'] and not np.array_equal(w[parts[part]],prior['fourier_combined'][parts[part]]):
                    raise ValueError('Changed anchor/recovery exposure')
    # Check raw commands without wrapping, clipping or relabeling expert goals.
    definitions=[RobotFK(read(Path(p['config'])),side) for side in ('left','right')]
    native={};arm_indices=[0,1,2,3,4,5,7,8,9,10,11,12]
    for part in ('expert_fit','recovery','anchor_fit'):
        ids=parts[part];valid=data['valid'][ids];target=data['target'][ids];base=data['base'][ids]
        violations=0
        for robot,start in zip(definitions,(0,7)):
            limits=joint_bounds(robot);q=target[:,:,start:start+6][valid]
            violations+=int(((q<limits[:,0])|(q>limits[:,1])).sum())
        if violations:raise ValueError('Native teacher target violates URDF bounds')
        delta=(target-base)[valid][:,arm_indices]
        native[part]=dict(rows=len(ids),scenes=len(groups[part]),valid_actions=int(valid.sum()),
            arm_delta_min=float(delta.min()),arm_delta_max=float(delta.max()),
            delta_components_over_pi=int((np.abs(delta)>np.pi).sum()),target_urdf_violations=violations,
            target_equals_base=bool(np.array_equal(target[valid],base[valid])))
    if not native['anchor_fit']['target_equals_base']:raise ValueError('Success anchors changed labels')
    run=new_run('coverage',group=('effect_learning',));print(run,flush=True)
    np.savez_compressed(run/'data.npz',**data);np.savez_compressed(run/'weights.npz',**wmap)
    files=[source/'protocol.json',source/'audit.json',source/'time_encoding_audit.json',source/'result.json',
        source/'analysis.json',source/'data.npz',source/'weights.npz',source/'support_inventory.json',Path(__file__),
        ROOT/'cx_tta/check_coverage.py',ROOT/'cx_tta/audit_coverage.py',ROOT/'cx_tta/audit_coverage_dense.py',
        ROOT/'cx_tta/summarize_coverage.py',ROOT/'scripts/train_coverage.sh']
    protocol={**p,'parts':parts,'conditions':CONDITIONS,'reference_source':str(source),
        'reference_conditions':['repeat_combined','fourier_combined'],
        'data_sha256':digest(run/'data.npz'),'weights_sha256':digest(run/'weights.npz'),
        'source_sha256':{**p['source_sha256'],**{str(f):digest(f) for f in files}},'native_target_check':native,
        'components':'Full856 expert-fit rows REPLACE17 ordinary control rows; ordinary17 remain evaluation-only subset. '
            'Expert targets are original expert commands, success anchors original base, recovery actual teacher commands.',
        'labels':'Expert fit856: original expert commands; recovery6: actual executed teacher commands; '
            'anchor-fit18: original base commands. Ordinary17 is an evaluation subset of expert fit.',
        'weighting':'Full874/880 rows per update. Each active component separately task/scene/row balanced '
            'and normalized to1. Sum component losses without rescaling when adding recovery. '
            'Per-row valid-step averaging; no padded supervision, sampling or loss-weight search.',
        'gates':{**p['gates'],'expert_fit':'Report every expert scene/arm; angle<=1deg and position<=1cm '
            'as a descriptive fit check. Expert targets differ from base; this is not a noninterference/success gate.'},
        'training':'Each update all874 control or880 recovery rows. Independently task/scene/row-balanced components; '
            'expert/anchor coefficients identical with/without recovery. Original6 recovery and18 anchor row weights unchanged. '
            'No new exposure sampling, wrapping, outlier filtering, new preprocessing or held fitting.',
        'control_isolation':'Recovery targets and four recovery seams excluded from control optimization; '
            'all recovery/held labels may be poisoned without changing control loss or gradient.',
        'evaluation_schedule':'Recovery, original ordinary17 and anchor-fit at140/700/1400; full856 expert-fit '
            'plus both developer-held sets only at1400. Original ordinary17 is not an extra training component.',
        'comparisons':['Within encoding: expert+anchor vs +recovery, same expert/anchor coefficients.',
            'Within regime: repeat vs Fourier time with same41742 parameters.',
            'S1u narrow recovery controls reused; full expert coverage changes tasks and compute, not only row count.'],
        'scope':'S1v fixed offline coverage diagnostic.12 models,three seeds,140/700/1400 updates,41742 parameters. '
            'Same previously studied two recovery scenes; expert fit48 scenes, success anchors5 scenes. No independent '
            'recovery test, simulation, collection, LoRA, TTA, deployment or success claim. Unbounded arm proposal. '
            'Full-batch compute grows with coverage; not an equal-FLOPs comparison to S1u.'}
    write_json(run/'protocol.json',protocol)
    print('PREREGISTERED',run,'models',12,'snapshots',36,'control_rows',874,'recovery_rows',880,flush=True)

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
    write_json(run/'loss_interface_check.json',check_interface(p,data,weight_np,device,per_row_loss,robots))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Coverage pilot exceeded3600s')));signal.alarm(3600)
    try:
        for seed in p['seeds']:
            for condition,setting in p['conditions'].items():
                torch.manual_seed(seed);model=TimeEncodedAdapter(p['context_dim'],encoding=setting['encoding'],envelope=setting['envelope']).to(device)
                initial=state_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=p['lr']);history=[]
                if sum(v.numel() for v in model.parameters())!=p['parameters']:raise ValueError('Parameter mismatch')
                idx_np=np.flatnonzero(weight_np[condition]);idx=torch.as_tensor(idx_np,device=device)
                weights=torch.as_tensor(weight_np[condition][idx_np],device=device);rep=setting['representation']
                local={int(g):i for i,g in enumerate(idx_np)}
                selected_pairs=p['seam_pairs'] if 'recovery' in setting['components'] else []
                pairs=torch.tensor([[local[v['before']],v['last'],local[v['after']]] for v in selected_pairs],device=device,dtype=torch.long).reshape(-1,3)
                pair_weights=torch.tensor(p['seam_weights'] if selected_pairs else [],device=device,dtype=torch.float32)
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
                        replace_json(run/'progress.json',dict(phase='training',condition=condition,seed=seed,update=update,
                            snapshots=len(reports),training_rows=len(idx_np),loss=float(loss.detach()),elapsed_seconds=time.monotonic()-started))
                    if update in p['updates']:
                        tag=f'{condition}_s{seed}_u{update}';checkpoint=run/f'{tag}.pt';model.eval();torch.save(model.state_dict(),checkpoint)
                        restored=TimeEncodedAdapter(p['context_dim'],encoding=setting['encoding'],envelope=setting['envelope']).to(device);restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));restored.eval()
                        summaries={};files=[]
                        for part,ix in p['parts'].items():
                            if part in ('inner_held','anchor_held','expert_fit') and update!=max(p['updates']):continue
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
        write_json(run/'result.json',dict(status='complete',reports=reports,elapsed_seconds=time.monotonic()-started,
            peak_gpu_memory_bytes=None if device=='cpu' else torch.cuda.max_memory_allocated(),scope=p['scope']))
        replace_json(run/'progress.json',dict(phase='complete',models=len(p['conditions'])*len(p['seeds']),snapshots=len(reports)))
        print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_snapshots=len(reports)));raise
    finally:signal.alarm(0)

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--run',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');args=parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if args.prepare_only:
        prepare(args.source.resolve())
    elif args.run is None:parser.error('A preregistered --run is required')
    else:train(args.run.resolve(),args.device)

if __name__=='__main__':main()

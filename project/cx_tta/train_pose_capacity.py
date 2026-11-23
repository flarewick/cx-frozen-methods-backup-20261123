"""Fixed small shared-adapter pose capacity gate on existing episode-held data."""
import argparse
import json
import signal
import traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .audit import digest
from .audit_local_geometry import joint_bounds
from .paths import ROOT,new_run,write_json,replace_json
from .pose_adapter import PoseAdapter,TorchFK,parity
from .robot_kinematics import RobotFK
from .train_semantic_pilot import load_features,balanced_weights

CACHE=ROOT/'outputs/feature_caches/joint_prefix_20260930_141244_699126'
OFFSETS=[0,3,7,15,31]
VARIANTS=('bounded_bc','proposal_bc','proposal_pose')

def fk_endpoints(pred,robots):
    return torch.stack([f(pred[:,OFFSETS,start:start+6]) for f,start in zip(robots,(0,7))],2)

def orientation_degrees(first,second):
    relative=first[...,:3,:3]@second[...,:3,:3].transpose(-1,-2)
    cosine=((torch.diagonal(relative,dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    return torch.rad2deg(torch.acos(cosine))

def summarize(pred,base,target,valid,rows,robots):
    # Accumulate raw endpoint arrays; task and episode balancing happen afterwards.
    def endpoint(values):
        with torch.no_grad():return torch.cat([fk_endpoints(torch.as_tensor(values[i:i+64],device=next(robots[0].buffers()).device),robots).cpu() for i in range(0,len(values),64)])
    pp,bp,tp=[endpoint(x) for x in (pred,base,target)]
    angles=orientation_degrees(pp,tp).numpy();baseline=orientation_degrees(bp,tp).numpy();translations=torch.linalg.vector_norm(pp[...,:3,3]-tp[...,:3,3],dim=-1).numpy()
    mask=np.broadcast_to(valid[:,OFFSETS,None],angles.shape);large=mask&(baseline>=45)
    large_lower=np.maximum(baseline-np.degrees(.6),0) # certified six .1-rad joint box endpoint bound
    result={}
    for task in sorted({r['task'] for r in rows}):
        groups=sorted({r['group'] for r in rows if r['task']==task});stats={k:[] for k in ('angle_deg','base_angle_deg','translation_m','large_angle_deg','large_base_angle_deg','large_reduction_fraction','large_under_15deg_fraction','bounded_lower_bound_deg','native_action_mse')};samples=0
        for group in groups:
            ids=[i for i,r in enumerate(rows) if r['group']==group];m=mask[ids];lm=large[ids]
            stats['angle_deg'].append(float(angles[ids][m].mean()));stats['base_angle_deg'].append(float(baseline[ids][m].mean()));stats['translation_m'].append(float(translations[ids][m].mean()))
            v=valid[ids];stats['native_action_mse'].append(float(((pred[ids]-target[ids])**2).mean(-1)[v].mean()))
            if lm.any():
                samples+=int(lm.sum());stats['large_angle_deg'].append(float(angles[ids][lm].mean()));stats['large_base_angle_deg'].append(float(baseline[ids][lm].mean()));stats['large_reduction_fraction'].append(float((angles[ids][lm]<baseline[ids][lm]).mean()));stats['large_under_15deg_fraction'].append(float((angles[ids][lm]<15).mean()));stats['bounded_lower_bound_deg'].append(float(large_lower[ids][lm].mean()))
        result[task]=dict(episodes=len(groups),large_angle_endpoints=samples,large_angle_episodes=len(stats['large_angle_deg']),**{k:None if not values else float(np.mean(values)) for k,values in stats.items()})
    return result,dict(angle=angles,base_angle=baseline,translation=translations,mask=mask,large=large)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--prepare-only',action='store_true');p.add_argument('--prepared',type=Path);p.add_argument('--planning-run',type=Path);p.add_argument('--target-run',type=Path);a=p.parse_args()
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if a.prepare_only:
        for name in ('feature_audit.json','checkpoint_audit.json'):
            if json.loads((a.cache/name).read_text())['status']!='passed':raise ValueError('Source feature audit failed')
        plan,rows,arrays,contexts,pre=load_features(a.cache)
        run=new_run('pose_capacity_data',group=('effect_learning',));print(run,flush=True)
        # Only legal observed semantic context/base enters forward; targets isolated in loss/evaluation.
        np.savez_compressed(run/'data.npz',context=contexts['semantic'],base=arrays['base'],target=arrays['target'],valid=arrays['valid'])
        np.savez_compressed(run/'preprocess.npz',**pre['semantic'])
        config_paths=[Path(r['collection'])/'resolved_config.json' for r in plan['records'] if r['kind']=='expert'];config_paths=sorted(set(config_paths));configs=[json.loads(x.read_text()) for x in config_paths]
        robots=[RobotFK(configs[0],side) for side in ('left','right')]
        parity_reports=[parity(r) for r in robots]
        # Orientation and target-base translation differences are invariant to task table/root translation.
        for cfg in configs[1:]:
            for i,side in enumerate(('left','right')):
                other=RobotFK(cfg,side)
                if digest(other.urdf)!=digest(robots[i].urdf) or other.names!=robots[i].names or other.tip!=robots[i].tip or not np.allclose(other.root[:3,:3],robots[i].root[:3,:3]):raise ValueError('Task FK convention mismatch')
        files=[a.cache/'manifest.json',a.cache/'result.json',a.cache/'feature_audit.json',a.cache/'checkpoint_audit.json',Path(__file__),ROOT/'cx_tta/pose_adapter.py',ROOT/'cx_tta/train_semantic_pilot.py',*config_paths,*[r.urdf for r in robots]]
        write_json(run/'manifest.json',dict(status='prepared',cache=str(a.cache),rows=rows,config=str(config_paths[0]),fk_parity=parity_reports,source_sha256={str(f):digest(f) for f in files},data_sha256=digest(run/'data.npz'),preprocess_sha256=digest(run/'preprocess.npz'),seeds=[0,1,2],epochs=10,lr=3e-4,variants=VARIANTS,pose_loss_weight=1.,offsets=OFFSETS,large_angle_threshold_deg=45.,selection='fixed final epoch; no tuning or early stopping; same original expert 48 fit/12 inner-held groups; successful anchors evaluated only',scope='Existing audited observable joint features; task identity never supplied to adapter; offline native expert commands, not failure recovery labels or task success. Proposal variants intentionally alter original command support. Original unseen validation untouched.'))
        print('PREPARED',run,flush=True);return
    if a.prepared is None or a.planning_run is None or a.target_run is None:p.error('Prepared data and completed planning/target gate paths required')
    gate=json.loads((a.target_run/'result.json').read_text());planning=json.loads((a.planning_run/'audit.json').read_text())
    if not gate['passed'] or not gate['position_audit_complete'] or not planning['passed']:raise ValueError('Target/repetition gates incomplete')
    manifest=json.loads((a.prepared/'manifest.json').read_text())
    for path,sha in manifest['source_sha256'].items():
        if digest(Path(path))!=sha:raise ValueError('Prepared source changed')
    if digest(a.prepared/'data.npz')!=manifest['data_sha256'] or digest(a.prepared/'preprocess.npz')!=manifest['preprocess_sha256']:raise ValueError('Prepared arrays changed')
    rows=manifest['rows'];config=json.loads(Path(manifest['config']).read_text());robotdefs=[RobotFK(config,s) for s in ('left','right')];robots=[TorchFK(r).to(a.device).float() for r in robotdefs]
    with np.load(a.prepared/'data.npz') as z:arrays={k:z[k].copy() for k in z.files}
    data={k:torch.as_tensor(v,device=a.device) for k,v in arrays.items()};parts={s:np.array([i for i,r in enumerate(rows) if r['pilot_split']==s]) for s in ('fit','inner_held','anchor_fit','anchor_held')}
    fit=parts['fit'];fitrows=[rows[i] for i in fit];weights=balanced_weights(fitrows);run=new_run('pose_capacity',group=('effect_learning',));print(run,flush=True)
    write_json(run/'protocol.json',dict(**manifest,prepared=str(a.prepared),prepared_manifest_sha256=digest(a.prepared/'manifest.json'),device=a.device,gate_sources={str(f):digest(f) for f in [a.target_run/'result.json',a.planning_run/'audit.json']},native_limits=[joint_bounds(r).tolist() for r in robotdefs],full_forward_input='semantic current/causal history context, native base action and chunk time only; no task ID, object pose, target pose, teacher target or plan'))
    reports=[];histories=[]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('capacity probe exceeded 1200 seconds')));signal.alarm(1200)
    try:
        with torch.no_grad():target_end=fk_endpoints(data['target'],robots)
        for seed in manifest['seeds']:
            for variant in manifest['variants']:
                torch.manual_seed(seed);model=PoseAdapter(data['context'].shape[1],proposal=variant!='bounded_bc').to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=manifest['lr']);generator=torch.Generator().manual_seed(seed);history=[]
                with torch.no_grad():
                    if not torch.equal(model(data['context'][:2],data['base'][:2]),data['base'][:2]):raise ValueError('Zero head changed baseline')
                for epoch in range(manifest['epochs']):
                    sampled=torch.multinomial(weights,len(fit),replacement=True,generator=generator);losses=[];model.train()
                    for start in range(0,len(sampled),64):
                        ids=torch.as_tensor(fit[sampled[start:start+64].numpy()],device=a.device);pred=model(data['context'][ids],data['base'][ids]);valid=data['valid'][ids]
                        bc=((((pred-data['target'][ids])/model.caps).square().mean(-1)*valid).sum(-1)/valid.sum(-1).clamp(min=1)).mean();loss=bc
                        if variant=='proposal_pose':
                            pe=fk_endpoints(pred,robots);te=target_end[ids];mask=data['valid'][ids][:,OFFSETS,None].expand(-1,-1,2)
                            # Smooth chordal SO(3) error = 2(1-cos(theta)), no acos derivative singularity.
                            chord=(pe[...,:3,:3]-te[...,:3,:3]).square().sum((-2,-1))/2
                            loss=loss+manifest['pose_loss_weight']*((chord/(.1**2)*mask).sum((1,2))/mask.sum((1,2)).clamp(min=1)).mean()
                        if not torch.isfinite(loss):raise ValueError('Nonfinite capacity loss')
                        opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
                    history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))));replace_json(run/'progress.json',dict(seed=seed,variant=variant,epoch=epoch+1,completed=len(reports)))
                    print('EPOCH',seed,variant,epoch+1,history[-1]['loss'],flush=True)
                model.eval();torch.save(model.state_dict(),run/f'{variant}_s{seed}.pt');restored=PoseAdapter(data['context'].shape[1],proposal=variant!='bounded_bc').to(a.device);restored.load_state_dict(torch.load(run/f'{variant}_s{seed}.pt',map_location=a.device,weights_only=True));restored.eval();metrics={}
                for part in ('inner_held','anchor_held'):
                    ids=parts[part];pred=[]
                    with torch.no_grad():
                        for start in range(0,len(ids),64):
                            bi=torch.as_tensor(ids[start:start+64],device=a.device);out=model(data['context'][bi],data['base'][bi]);again=restored(data['context'][bi],data['base'][bi])
                            if not torch.equal(out,again):raise ValueError('Reload mismatch')
                            pred.append(out.cpu().numpy())
                    pred=np.concatenate(pred);base=arrays['base'][ids];target=arrays['target'][ids];valid=arrays['valid'][ids];delta=pred-base
                    if not np.isfinite(pred).all() or (variant=='bounded_bc' and np.max(np.abs(delta)-model.caps.cpu().numpy())>1e-6):raise ValueError('Prediction/cap violation')
                    summary,endpoint_arrays=summarize(pred,base,target,valid,[rows[i] for i in ids],robots);metrics[part]=summary
                    limit_violation=0
                    for r,start in zip(robotdefs,(0,7)):
                        limits=joint_bounds(r);q=pred[...,start:start+6];limit_violation+=int(((q<limits[:,0])|(q>limits[:,1])).sum())
                    metrics[part+'_support']=dict(max_arm_correction_rad=float(np.abs(delta[...,[0,1,2,3,4,5,7,8,9,10,11,12]]).max()),urdf_limit_violations=limit_violation,gripper_outside_zero_one=int(((pred[...,[6,13]]<0)|(pred[...,[6,13]]>1)).sum()))
                    np.savez_compressed(run/f'{variant}_s{seed}_{part}.npz',predicted=pred,base=base,target=target,valid=valid,indices=ids,**endpoint_arrays)
                reports.append(dict(seed=seed,variant=variant,parameters=sum(p.numel() for p in model.parameters()),metrics=metrics,reload='exact'));histories.append(dict(seed=seed,variant=variant,history=history));print('VARIANT_COMPLETE',seed,variant,json.dumps(metrics['inner_held']),flush=True)
        write_json(run/'history.json',histories);write_json(run/'result.json',dict(status='complete',reports=reports,scope=manifest['scope']));replace_json(run/'progress.json',dict(phase='complete',variants=len(reports)));print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed=reports));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()

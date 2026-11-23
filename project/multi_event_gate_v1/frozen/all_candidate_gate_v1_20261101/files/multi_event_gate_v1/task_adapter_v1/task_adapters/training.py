"""Task-configurable expert initialization: frozen observation cache, shared training, audit.

No success-preservation or event-benefit claims from expert-only supervision.
Targets follow saved expert command i+1..i+32, not certified32 executor actions.
"""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid
import fcntl
import numpy as np

from cx_tta.audit import digest
from cx_tta.paths import WORKSPACE_ROOT,ROOT,new_run,write_json,replace_json,output_path
from cx_tta.collection_jobs import read,build_jobs,validate_complete
from .adapter import SCOPE


def verify(files):
    for path,sha in files.items():
        if digest(Path(path))!=sha:raise ValueError('Frozen dependency changed: '+path)


def expert_targets(h,index):
    from cx_tta.dataset import expert_chunk
    commands=h['joint_action/vector'][:]
    target,valid=expert_chunk(commands,index)
    dt=h['cx_effect/transition_dt_s'][index:index+32]
    valid[:len(dt)] &= dt>0
    if not valid.any():raise ValueError('No positive-time expert transition in this row')
    return target,valid


def prepare(dataset):
    import h5py
    from .protocol import load,TEMPLATE
    from .collection import audit
    protocol=load(dataset);template=read(TEMPLATE)
    records=[];sources=dict(protocol['source_sha256'])
    sources[str(dataset/'protocol.json')]=digest(dataset/'protocol.json')
    for task in protocol['tasks']:
        audit(dataset,protocol,task,'smoke')
        for split in ('adapter_fit','adapter_validation'):
            audit(dataset,protocol,task,split)
            folder=dataset/'collections'/task/split
            sources[str(folder/'attempts.json')]=digest(folder/'attempts.json')
            for row in read(folder/'attempts.json'):
                if not row.get('accepted'):continue
                source=Path(row['hdf5']);sources[str(source)]=row['capture']['source_sha256']
                with h5py.File(source,'r') as h:
                    n=len(h['joint_action/vector']);frames=[];valid_steps=0
                    for i in range(0,n-1,32):
                        if not np.any(h['cx_effect/transition_dt_s'][i:i+32]>0):continue
                        _,valid=expert_targets(h,i);valid_steps+=int(valid.sum());frames.append(i)
                    if not frames:raise ValueError('No usable expert rows: '+str(source))
                    records.append(dict(task=task,split=split,seed=row['seed'],source=str(source),
                        source_sha256=sources[str(source)],frames=frames,valid_steps=valid_steps,
                        instruction=str(h.attrs['instruction']),saved_frames=n))
    # Retain the exact initial VLA/URDF dependencies, never the old training data.
    for name,sha in template['source_sha256'].items():
        file=Path(name)
        if file.suffix=='.urdf' or '/qfw/robotwin/' in name and file.suffix=='.py':
            if digest(file)!=sha:raise ValueError('Frozen base/robot code changed: '+name)
            sources[name]=sha
    p={k:v for k,v in template.items() if k not in ('records','source_sha256','pre_gpu_amendment')}
    p.update(scope=SCOPE,records=records,source_sha256=sources,updates=protocol['updates'],
             training_tasks=protocol['tasks'],dataset=str(dataset),microbatch=128,
             validation='Whole scenes held out; final-budget checkpoint; no EVAL selection')
    run=output_path(dataset/'training');run.mkdir(exist_ok=True);(run/'cache').mkdir(exist_ok=True)
    if (run/'training_plan.json').exists():
        if read(run/'training_plan.json')!=p:raise ValueError('Frozen training plan changed')
    else:write_json(run/'training_plan.json',p)
    print('TRAINING_RUN',run,flush=True);return run


def load_plan(run):
    p=read(run/'training_plan.json')
    if (p.get('scope')!=SCOPE or p.get('history_mode')!='current_only'
        or not 1<=p.get('updates',0)<=100000 or p.get('seed')!=0):
        raise ValueError('Frozen task-adapter plan required')
    verify(p['source_sha256']);return p


def verify_base(p):
    from cx_tta.evaluate_structured_action import base_identity
    actual=base_identity(Path(p['tokenizer']))
    for key in ('checkpoint','base_checkpoint_sha256','normalization_sha256','tokenizer_sha256'):
        if actual[key]!=p['policy_identity'][key]:raise ValueError('Frozen VLA identity changed: '+key)
    return actual


def atomic_npz(path,**values):
    temp=path.with_suffix('.npz.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**values)
    temp.replace(path)


def cache_valid(run,index,row):
    path=run/'cache'/f'scene_{index:03d}.npz';meta=read(path.with_suffix('.json'))
    if not meta:return False
    if (meta['source_sha256']!=row['source_sha256'] or meta['frames']!=row['frames']
        or meta['sha256']!=digest(path) or meta['plan_sha256']!=digest(run/'training_plan.json')):
        raise ValueError('Changed existing scene cache')
    return True


def cache(run,gpu):
    import torch,h5py
    from cx_tta.cache_expert import create_base_policy
    from cx_tta.cache_joint_prefix import encode
    from cx_tta.dataset import read_observation
    from cx_tta.structured_features import pack_observable
    from cx_tta.robot_kinematics import RobotFK
    from cx_tta.evaluate_b2 import seed_all,configure_inference
    from cx_tta.recovery_contract import rng_capture,rng_fingerprint
    from cx_tta.evaluate_structured_low_storage import PrivateEvalStorage
    p=load_plan(run)
    verify_base(p)
    pending=[i for i,r in enumerate(p['records']) if not cache_valid(run,i,r)]
    if not pending:return audit_cache(run,p)
    session=run/('cache_session_'+uuid.uuid4().hex[:12]);session.mkdir();runtime=session/'runtime';runtime.mkdir()
    storage=PrivateEvalStorage();atexit.register(storage.cleanup)
    storage.configure_runtime(session,runtime,Path(p['tokenizer']))
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no VLA cache generated')
    configure_inference('compiled',0);torch.set_num_threads(2)
    policy=create_base_policy();policy._model.eval().requires_grad_(False)
    sampler=policy._sample_actions
    if not hasattr(sampler,'_torchdynamo_orig_callable'):raise ValueError('Compiled frozen sampler required')
    eager=torch.compiler.disable(encode)
    projection=dict(np.load(p['projection'],allow_pickle=False))
    config=read(Path(p['robot_config']));robots=[RobotFK(config,s) for s in ('left','right')]
    write_json(session/'hardware.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),backbone_trainable=sum(v.numel() for v in policy._model.parameters() if v.requires_grad),compiled=True))
    for rid in pending:
        row=p['records'][rid];xs=[];bases=[];targets=[];masks=[]
        with h5py.File(row['source'],'r') as h:
            for frame in row['frames']:
                images,state=read_observation(h,frame)
                sample_seed=int.from_bytes(hashlib.sha256(f'{row["task"]}:{row["seed"]}:{frame}'.encode()).digest()[:4],'little') % (2**31)
                seed_all(sample_seed)
                before=rng_fingerprint(rng_capture());cfg=policy._model.paligemma_with_expert.paligemma.language_model.config
                implementation=cfg._attn_implementation
                try:joint,_=eager(policy,images,state,row['instruction'])
                finally:cfg._attn_implementation=implementation
                if before!=rng_fingerprint(rng_capture()):raise ValueError('Feature extraction consumed sampler RNG')
                x=pack_observable(joint[None],state[None],np.zeros_like(joint)[None],np.zeros_like(state)[None],
                    np.asarray([False]),projection,robots)
                obs=dict(state=state.copy(),prompt=row['instruction'],images={k:v.transpose(2,0,1) for k,v in zip(('cam_high','cam_right_wrist','cam_left_wrist'),images)})
                base=np.asarray(policy.infer(obs,noise=None)['actions'],np.float32)
                if before==rng_fingerprint(rng_capture()):raise ValueError('Internal sampler did not advance RNG')
                if base.shape!=(32,14) or not np.isfinite(base).all():raise ValueError('Invalid frozen base actions')
                if policy._sample_actions is not sampler or torch._dynamo.config.disable:raise ValueError('Sampler changed')
                target,valid=expert_targets(h,frame)
                xs.append(x);bases.append(base);targets.append(target);masks.append(valid)
        path=run/'cache'/f'scene_{rid:03d}.npz'
        arrays={k:np.concatenate([x[k] for x in xs]) for k in xs[0]}
        atomic_npz(path,**arrays,base=np.stack(bases),target=np.stack(targets),valid=np.stack(masks),frame_index=np.asarray(row['frames']))
        replace_json(path.with_suffix('.json'),dict(source_sha256=row['source_sha256'],frames=row['frames'],sha256=digest(path),
            plan_sha256=digest(run/'training_plan.json'),session=str(session),task=row['task'],split=row['split']))
        replace_json(run/'cache_progress.json',dict(completed=sum((run/'cache'/f'scene_{i:03d}.json').exists() for i in range(len(p['records']))),requested=len(p['records']),last_task=row['task'],last_seed=row['seed']))
        print('CACHED',row['task'],row['split'],row['seed'],len(row['frames']),'rows',flush=True)
    audit_cache(run,p)


def audit_cache(run,p=None):
    from cx_tta.structured_action_adapter import FORWARD_KEYS
    p=p or load_plan(run);files={};counts={};saturation=[]
    seen={}
    for i,row in enumerate(p['records']):
        if not cache_valid(run,i,row):raise ValueError('Missing scene cache')
        key=(row['task'],row['seed'])
        if key in seen:raise ValueError('Duplicate scene or split leakage')
        seen[key]=row['split']
        path=run/'cache'/f'scene_{i:03d}.npz';files[str(path)]=digest(path)
        with np.load(path,allow_pickle=False) as z:
            n=len(row['frames'])
            if set(z.files)!=set(FORWARD_KEYS)|{'base','target','valid','frame_index'}:raise ValueError('Cache fields changed')
            if z['base'].shape!=(n,32,14) or z['target'].shape!=(n,32,14) or z['valid'].shape!=(n,32):raise ValueError('Cache chunk shapes')
            if not np.array_equal(z['frame_index'],row['frames']) or z['valid'].dtype!=np.bool_ or not z['valid'].any(1).all():raise ValueError('Bad frames/mask')
            if any(not np.isfinite(z[k]).all() for k in z.files):raise ValueError('Nonfinite cache')
            if any(z[k].any() for k in FORWARD_KEYS if k.startswith('past_') or k=='has_history'):raise ValueError('Invented expert native history')
            delta=(z['target']-z['base'])[..., [0,1,2,3,4,5,7,8,9,10,11,12]]
            saturation.append(dict(task=row['task'],split=row['split'],seed=row['seed'],fraction_outside_cap=float((np.abs(delta[z['valid']])>.1).mean())))
        name=row['task']+'/'+row['split'];counts[name]=counts.get(name,0)+n
    manifest=dict(complete=True,scenes=len(p['records']),rows_by_task_split=counts,files_sha256=files,
        plan_sha256=digest(run/'training_plan.json'),saturation=saturation,history='current_only',
        targets='expert_saved_sequence_only',native_recovery_validated=False,success_preservation_included=False)
    if (run/'cache_manifest.json').exists() and read(run/'cache_manifest.json')!=manifest:raise ValueError('Cache manifest changed')
    if not (run/'cache_manifest.json').exists():write_json(run/'cache_manifest.json',manifest)
    print('CACHE_AUDIT_PASS',counts,flush=True);return manifest


def arrays_for(run,p,split):
    from cx_tta.structured_action_adapter import FORWARD_KEYS
    from .adapter import scene_weights
    values=[];rows=[]
    for i,r in enumerate(p['records']):
        if r['split']!=split:continue
        with np.load(run/'cache'/f'scene_{i:03d}.npz',allow_pickle=False) as z:values.append({k:z[k].copy() for k in z.files})
        rows.extend(dict(task=r['task'],seed=r['seed'],frame=f) for f in r['frames'])
    data={k:np.concatenate([v[k] for v in values]) for k in values[0]}
    return {k:data[k] for k in FORWARD_KEYS},dict(base=data['base'],target=data['target'],valid=data['valid'],weights=scene_weights(rows)),rows


def tensorize(values,device):
    import torch
    return {k:torch.from_numpy(v.copy()).to(device) for k,v in values.items()}


def initial_model(p,device):
    import torch
    from .adapter import StackingAdapter
    model=StackingAdapter(32,p['geometry_mean']).to(device)
    model.load_state_dict(torch.load(p['initial'],map_location=device,weights_only=True)['model'],strict=True)
    model.geometry_head.requires_grad_(False)
    return model


def atomic_torch(path,value):
    import torch
    temp=path.with_suffix('.pt.tmp');torch.save(value,temp);temp.replace(path)


def train(run,device='cuda'):
    import torch
    from .adapter import arm_loss
    # Set before constructing any CUDA linear layers/optimizer; never change the
    # frozen VLA's inference settings in the separate caching process.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=load_plan(run);audit_cache(run,p)
    if (run/'training_result.json').exists():return audit(run)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=device,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        deterministic=True,tf32=False,cublas_workspace=':4096:8')
    if (run/'training_backend.json').exists() and read(run/'training_backend.json')!=backend:
        raise ValueError('Training backend changed; do not silently mix resumed optimizer updates')
    if not (run/'training_backend.json').exists():write_json(run/'training_backend.json',backend)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(p['seed'])
    x,y,_=arrays_for(run,p,'adapter_fit');x=tensorize(x,device);y=tensorize(y,device)
    model=initial_model(p,device)
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['lr'],weight_decay=p['weight_decay'])
    step0=0;history=[];latest=run/'training_latest.pt';plan_sha=digest(run/'training_plan.json');cache_sha=digest(run/'cache_manifest.json')
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=True)
        if saved['plan_sha256']!=plan_sha or saved['cache_sha256']!=cache_sha:raise ValueError('Resume training identity differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);step0=saved['step'];history=saved['history']
        if not 0<=step0<=p['updates']:raise ValueError('Invalid training update count')
    began=time.monotonic()
    for step in range(step0+1,p['updates']+1):
        optimizer.zero_grad(set_to_none=True)
        # Accumulate the SAME full weighted objective; no task/count bias and bounded VRAM.
        loss=torch.zeros((),device=device)
        for start in range(0,len(y['base']),p['microbatch']):
            sl=slice(start,start+p['microbatch'])
            out=model({k:v[sl] for k,v in x.items()},y['base'][sl])
            part=arm_loss(out['delta'],y['base'][sl],y['target'][sl],y['valid'][sl],y['weights'][sl])
            if not torch.isfinite(part):raise ValueError('Nonfinite training loss')
            part.backward();loss+=part.detach()
        optimizer.step()
        if step==1 or step%100==0 or step==p['updates']:
            row=dict(step=step,loss=float(loss.detach()));history.append(row)
            atomic_torch(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,history=history,plan_sha256=plan_sha,cache_sha256=cache_sha))
            replace_json(run/'training_progress.json',dict(**row,requested=p['updates'],elapsed_s=time.monotonic()-began))
            print('TRAIN',row,flush=True)
    model.eval();initial=initial_model(p,device).eval();reports=[]
    from .adapter import scene_weights
    for split in ('adapter_fit','adapter_validation'):
        xn,yn,rows=arrays_for(run,p,split);xt=tensorize(xn,device);yt=tensorize(yn,device)
        with torch.inference_mode():out=model(xt,yt['base']);prior=initial(xt,yt['base'])
        for task in sorted({r['task'] for r in rows}):
            idx=[i for i,r in enumerate(rows) if r['task']==task];weights=yt['weights'][idx];weights=weights/weights.sum()
            reports.append(dict(split=split,task=task,rows=len(idx),direction_loss=float(arm_loss(out['delta'][idx],yt['base'][idx],yt['target'][idx],yt['valid'][idx],weights)),
                initial_loss=float(arm_loss(prior['delta'][idx],yt['base'][idx],yt['target'][idx],yt['valid'][idx],weights)),
                zero_loss=float(arm_loss(torch.zeros_like(out['delta'][idx]),yt['base'][idx],yt['target'][idx],yt['valid'][idx],weights))))
        if split=='adapter_validation':atomic_npz(run/'validation_predictions.npz',delta=out['delta'].cpu().numpy())
    checkpoint=run/'adapter_seed0.pt'
    meta=dict(scope=SCOPE,training_tasks=p['training_tasks'],training_run=str(run),updates=p['updates'],seed=p['seed'],channels=32,geometry_mean=p['geometry_mean'],
        projection_path=p['projection'],projection_sha256=digest(Path(p['projection'])),robot_config_path=p['robot_config'],robot_config_sha256=digest(Path(p['robot_config'])),
        training_plan_sha256=plan_sha,cache_manifest_sha256=cache_sha,history_mode='current_only',arm_cap_rad=.1,gripper_residual=0,
        parameters=sum(v.numel() for v in model.parameters()),trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        stage='expert_initialization',success_preservation_included=False,gate_trained=False)
    atomic_torch(checkpoint,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},metadata=meta))
    verify(p['source_sha256'])
    write_json(run/'training_result.json',dict(status='complete',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),updates=p['updates'],
        history=history,reports=reports,device=device,interpretation='Offline expert initialization only. No task-success or recovery benefit established.'))
    print('TRAIN_COMPLETE',checkpoint,flush=True)


def audit(run):
    import torch
    from .adapter import StackingAdapter,read_checkpoint
    p=load_plan(run);payload=read_checkpoint(run/'adapter_seed0.pt',require_audit=False)
    model=StackingAdapter(32,p['geometry_mean']);model.load_state_dict(payload['model']);model.eval();torch.set_num_threads(2)
    xn,yn,_=arrays_for(run,p,'adapter_validation');x=tensorize(xn,'cpu');y=tensorize(yn,'cpu')
    with torch.inference_mode():
        out=model(x,y['base']);zero=model(x,y['base'],enabled=False)
    saved=np.load(run/'validation_predictions.npz',allow_pickle=False)['delta']
    error=float(np.abs(saved-out['delta'].numpy()).max())
    if error>2e-5 or not torch.equal(zero['actions'],y['base']) or out['delta'][...,[6,13]].any() or out['delta'].abs().max()>.100001:
        raise ValueError('Reload/bypass/ARM cap/GRIP audit failed')
    result=dict(passed=True,cpu_reload_max_error=error,checkpoint_sha256=digest(run/'adapter_seed0.pt'),
        expert_initialization_only=True,closed_loop_validated=False,gate_trained=False)
    replace_json(run/'training_audit.json',result);print('TRAIN_AUDIT',result,flush=True);return result


def main(argv=None):
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('phase',choices=('prepare','cache','train','audit','pipeline','check'))
    ap.add_argument('--run',type=Path);ap.add_argument('--gpu',type=int,default=0)
    a=ap.parse_args(argv)
    if a.gpu<0:ap.error('Nonnegative GPU')
    if a.phase=='prepare':
        if a.run is None:ap.error('--run dataset required')
        prepare(output_path(a.run.resolve()));return
    if a.run is None:ap.error('--run required')
    run=output_path(a.run.resolve());os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),JAX_PLATFORMS='cpu')
    if a.phase=='check':
        p=load_plan(run);print(json.dumps(dict(stage=p['stage'],scenes=len(p['records']),rows=sum(len(r['frames']) for r in p['records']),gpu_started=False)));return
    if a.phase=='pipeline':
        with (run/'pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for phase in ('cache','train','audit'):
                subprocess.run([sys.executable,'-B','-m','task_adapters.training',phase,'--run',str(run),'--gpu',str(a.gpu)],check=True)
        return
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    try:
        if a.phase=='cache':cache(run,a.gpu)
        elif a.phase=='train':train(run)
        else:audit(run)
    except BaseException:
        replace_json(run/(a.phase+'_failure.json'),dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()

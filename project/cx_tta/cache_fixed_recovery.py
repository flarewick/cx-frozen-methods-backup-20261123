"""S2d cache: same-state compiled bases, frozen features, actual teacher labels."""
import argparse,json,os,signal,time,traceback
from pathlib import Path
import numpy as np
from .audit import digest
from .eval_diagnostics import array_hash
from .paths import ROOT,write_json,replace_json
from .cache_recovery import read,verify_hashes,legal_observation,request
from .recovery_context_v2 import project_context

def cache(run,gpu):
    p=read(run/'protocol.json');verify_hashes(p['source_sha256'])
    if (run/'result.json').exists():raise ValueError('Fresh run required')
    weight=Path(p['checkpoint'])/'model.safetensors';st=weight.stat()
    if dict(size=st.st_size,mtime_ns=st.st_mtime_ns)!=p['checkpoint_stat']:
        raise ValueError('Checkpoint changed after verified full hash')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu;os.environ['JAX_PLATFORMS']='cpu'
    from .evaluate_b2 import configure_runtime,seed_all
    runtime=run/'runtime';runtime.mkdir();configure_runtime(run,runtime,Path(p['tokenizer']))
    import torch
    import h5py
    from .cache_expert import create_base_policy
    from .cache_joint_prefix import encode
    from .dataset import read_observation
    from .recovery_contract import rng_capture,rng_restore,rng_fingerprint
    from .collect_recovery import rng_json
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch._dynamo.config.disable:raise RuntimeError('Official compiled CUDA policy required')
    reports=[];started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fixed recovery cache exceeded 1800s')))
    signal.alarm(1800)
    try:
        seed_all(0);policy=create_base_policy();policy._model.eval().requires_grad_(False)
        compiled=policy._sample_actions
        if not hasattr(compiled,'_torchdynamo_orig_callable'):raise ValueError('Sampling callable is not compiled')
        def feature(images,state,prompt):
            before=rng_capture();cfg=policy._model.paligemma_with_expert.paligemma.language_model.config
            previous=cfg._attn_implementation
            try:value,meta=encode(policy,images,state,prompt)
            finally:cfg._attn_implementation=previous
            if rng_fingerprint(before)!=rng_fingerprint(rng_capture()):raise ValueError('Feature extraction changed sampler RNG')
            if policy._sample_actions is not compiled or torch._dynamo.config.disable:raise ValueError('Auxiliary extractor changed compiled sampler')
            return value,meta
        golden=p['golden'];record=golden['source']
        with np.load(golden['file']) as z:frame=int(z['frames'][0]);expected=z['joint'][0].copy()
        with h5py.File(record['source'],'r') as h:ims,state=read_observation(h,frame)
        actual,gmeta=feature(ims,state,record['instruction'])
        if not np.array_equal(actual,expected):raise ValueError('Existing frozen feature convention mismatch')
        write_json(run/'environment.json',dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
            cuda=torch.version.cuda,compiled=True,global_dynamo_disabled=False,model_trainable_parameters=0,
            golden_exact=True,golden_frame=frame,golden_metadata=gmeta,
            matmul_precision=torch.get_float32_matmul_precision(),sample_kwargs=policy._sample_kwargs))
        joint=[];states=[];feature_meta=[]
        for i,row in enumerate(p['feature_rows']):
            ims,state,sig=legal_observation(row);v,meta=feature(ims,state,row['instruction'])
            repeated,_=feature(ims,state,row['instruction'])
            if not np.array_equal(v,repeated):raise ValueError('Auxiliary feature is not repeatable')
            joint.append(v);states.append(state);feature_meta.append(dict(row=i,signature=sig,repeat_exact=True,rng_unchanged=True,**meta))
        np.savez_compressed(run/'features.npz',joint=np.stack(joint),state=np.stack(states))
        write_json(run/'feature_checks.json',feature_meta)
        def infer(row):
            ims,state,_=legal_observation(row)
            actions=np.asarray(policy.infer(request(ims,state,row['instruction']))['actions'],np.float32).copy()
            if actions.shape!=(32,14) or not np.isfinite(actions).all():raise ValueError('Invalid compiled action block')
            return actions
        # Compilation/warm-up occurs before each scene stream starts, without selecting labels.
        warm=[]
        for group in dict.fromkeys(r['group'] for r in p['rows']):
            row=next(r for r in p['rows'] if r['group']==group);seed_all(p['policy_seed'])
            replace_json(run/'progress.json',dict(phase='warming_compiled_policy',group=group))
            start=time.monotonic();action=infer(row)
            warm.append(dict(group=group,seconds=time.monotonic()-start,sha256=array_hash(action)))
        write_json(run/'warmup.json',warm)
        previous_group=None;actions=[];repeats=[]
        for i,row in enumerate(p['rows']):
            if row['group']!=previous_group:seed_all(p['policy_seed']);previous_group=row['group']
            before=rng_capture();start=time.monotonic();first=infer(row);after=rng_capture()
            rng_restore(before);again=infer(row);repeat_after=rng_capture();rng_restore(after)
            if rng_fingerprint(after)!=rng_fingerprint(repeat_after):raise ValueError('Repeated query sampler advancement differs')
            if rng_fingerprint(before)==rng_fingerprint(after):raise ValueError('Internal sampler did not advance RNG')
            write_json(run/f'rng_{i:02d}.json',dict(before=rng_json(before),after=rng_json(after)))
            source_compare=[]
            for old in row['matched_compiled_base']:
                if digest(Path(old['file']))!=old['sha256']:raise ValueError('Changed matched old base')
                with np.load(old['file']) as z:old_actions=z['actions']
                source_compare.append(dict(source=old,exact=bool(np.array_equal(first,old_actions)),
                    max_absolute_difference=float(np.abs(first-old_actions).max()),
                    native_mse=float(np.mean((first-old_actions)**2)),
                    old_rng_before_matches=old['rng']['before']==rng_fingerprint(before)))
            reports.append(dict(row=i,group=row['group'],native_count=row['native_count'],
                observation_signature=row['observation_signature'],base_sha256=array_hash(first),
                repeat_exact=bool(np.array_equal(first,again)),repeat_max_abs=float(np.abs(first-again).max()),
                before_rng=rng_fingerprint(before),after_rng=rng_fingerprint(after),
                repeat_after_rng=rng_fingerprint(repeat_after),seconds=time.monotonic()-start,
                matched_old_base_comparison=source_compare))
            actions.append(first);repeats.append(again)
            replace_json(run/'progress.json',dict(phase='caching',rows=i+1,total=len(p['rows'])));
            print('CACHED',i+1,row['group'],row['native_count'],'repeat_exact',reports[-1]['repeat_exact'],flush=True)
        # Labels first loaded here; they cannot affect any inference above.
        labels=[]
        for row in p['rows']:
            with np.load(row['target']) as z:labels.append({k:z[k].copy() for k in z.files})
        with np.load(p['preprocess']) as z:pre={k:z[k].copy() for k in z.files}
        history=np.array(p['history_indices'])
        fhi=np.array(p['feature_history_indices']); current_ids=np.array(p['feature_current_indices'])
        context=project_context(np.stack(joint),np.stack(states),fhi,pre)[current_ids]
        from .dataset import context_features
        coarse=[]
        for row in p['feature_rows']:
            ims,state,_=legal_observation(row);coarse.append(context_features(ims,state)[14:])
        with np.load(p['coarse_preprocess']) as z:cpre={k:z[k].copy() for k in z.files}
        coarse_context=project_context(np.stack(coarse),np.stack(states),fhi,cpre)[current_ids]
        np.savez_compressed(run/'data.npz',context=context,semantic=context,coarse=coarse_context,feature_current_indices=current_ids,feature_history_indices=fhi,base=np.stack(actions),repeated_base=np.stack(repeats),
            target=np.stack([x['target'] for x in labels]),valid=np.stack([x['valid'] for x in labels]),
            source_native_counts=np.stack([x['source_native_counts'] for x in labels]),
            plan_ids=np.stack([x['plan_ids'] for x in labels]),history_indices=history)
        write_json(run/'result.json',dict(status='complete_pending_audit',rows=reports,observations=len(reports),
            branches=len({r['group'] for r in p['rows']}),independent_scenes=len({r['scene_group'] for r in p['rows']}),valid_teacher_actions=sum(r['valid_actions'] for r in p['rows']),
            data_sha256=digest(run/'data.npz'),features_sha256=digest(run/'features.npz'),
            elapsed_seconds=time.monotonic()-started,training_scope='developer_recovery_fit_interface_only',scope=p['scope']))
        replace_json(run/'progress.json',dict(phase='complete_pending_audit',rows=len(reports)))
        print('COMPLETE_PENDING_AUDIT',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed=reports));raise
    finally:signal.alarm(0)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',default='0');a=p.parse_args()
    if not a.gpu.isdigit():p.error('Numeric GPU required')
    cache(a.run.resolve(),a.gpu)

if __name__=='__main__':main()

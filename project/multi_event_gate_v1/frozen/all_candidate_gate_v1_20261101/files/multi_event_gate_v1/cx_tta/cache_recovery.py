"""Cache legal saved recovery observations without running a simulator.

Teacher commands are loaded only after frozen feature/action inference. The
result is a six-row developer fit interface, not an independent recovery set.
"""
import argparse
import json
import os
import signal
import time
import traceback
from pathlib import Path
import numpy as np
from .audit import digest
from .eval_diagnostics import array_hash
from .paths import ROOT, new_run, write_json, replace_json

INVENTORY = ROOT/'outputs/reports/audits/recovery_inventory_20261002_063604_876790'
FEATURES = ROOT/'outputs/feature_caches/joint_prefix_20260930_141244_699126'
PREPARED = ROOT/'outputs/effect_learning/pose_capacity_data_20261002_045340_093853'
FIT = ROOT/'outputs/effect_learning/fit_budget_20261002_070634_070111'
ROLES = ('verified_held_recovery', 'native_switch_recovery')

def read(p):return json.loads(p.read_text())

def verify_hashes(values):
    for file,sha in values.items():
        if digest(Path(file))!=sha:raise ValueError('Changed source '+file)

def legal_observation(row):
    path=Path(row['observation'])
    if digest(path)!=row['observation_sha256']:raise ValueError('Changed saved observation')
    with np.load(path) as z:
        images=[z[k+'_rgb'].copy() for k in ('head','right','left')]
        state=z['command_state'].copy(); raw=z['command_state_raw'].copy()
    signature=dict(images={key:array_hash(im) for key,im in zip(
        ('head_camera','right_camera','left_camera'),images)},command_state=array_hash(raw))
    if signature!=row['observation_signature']:raise ValueError('Wrong observation predecessor')
    if state.shape!=(14,) or state.dtype!=np.float32 or not np.array_equal(state,raw.astype(np.float32)):
        raise ValueError('Policy command-state convention changed')
    if any(im.dtype!=np.uint8 or im.shape!=(240,320,3) for im in images):
        raise ValueError('Unexpected camera input')
    if not np.isfinite(state).all():raise ValueError('Nonfinite state')
    return images,state,signature

def request(images,state,prompt):
    return dict(state=state.copy(),prompt=prompt,images={k:im.transpose(2,0,1).copy()
        for k,im in zip(('cam_high','cam_right_wrist','cam_left_wrist'),images)})

def context_from_features(joint,state,history,pre):
    projected=(joint.astype(np.float32)-pre['pca_mean'])@pre['pca_components'].T
    current=np.c_[projected,state].astype(np.float32)
    past=np.zeros_like(current);has=history>=0;past[has]=current[history[has]]
    return ((np.c_[current,past,has.astype(np.float32)]-pre['context_mean'])/pre['context_std']).astype(np.float32)

def prepare():
    from .cache_expert import CHECKPOINT,RT
    from .evaluate_b2 import resolve_tokenizer
    if not read(FIT/'audit.json')['passed']:raise ValueError('Complete fit/budget diagnosis first')
    inv=read(INVENTORY/'result.json');audit=read(INVENTORY/'audit.json')
    if not audit['passed'] or not inv['passed'] or inv['status']!='complete':raise ValueError('Audited inventory required')
    verify_hashes(inv['source_sha256']);verify_hashes(audit['source_sha256'])
    rows=[r for r in inv['records'] if r['role'] in ROLES]
    if len(rows)!=6 or len({r['group'] for r in rows})!=2 or sum(r['valid_actions'] for r in rows)!=144:
        raise ValueError('Unexpected candidate inventory')
    history=[];previous={};provenances=[];sources=[INVENTORY/'result.json',INVENTORY/'audit.json',
        FIT/'audit.json',FIT/'result.json',FEATURES/'manifest.json',FEATURES/'result.json',
        FEATURES/'feature_audit.json',FEATURES/'checkpoint_audit.json',PREPARED/'manifest.json',
        PREPARED/'preprocess.npz',Path(__file__),ROOT/'cx_tta/cache_joint_prefix.py',
        ROOT/'cx_tta/cache_expert.py',ROOT/'cx_tta/evaluate_b2.py',ROOT/'cx_tta/recovery_contract.py']
    for i,row in enumerate(rows):
        legal_observation(row)
        if digest(Path(row['target']))!=row['target_sha256']:raise ValueError('Changed teacher target')
        j=previous.get(row['group'],-1);history.append(j)
        expected=None if j<0 else rows[j]['observation']
        if row['past_observation']!=expected:raise ValueError('Recovery history gap or wrong branch')
        if j>=0 and rows[j]['native_count']+32!=row['native_count']:raise ValueError('Not a causal 32-action boundary')
        previous[row['group']]=i
        source=Path(row['source']);sm=read(source/'manifest.json');l0=Path(sm['source'])
        prov=l0/'provenance.json';provenances.append(read(prov));sources.extend([prov,source/'manifest.json',
            source/'result.json',source/'trace.jsonl',Path(row['observation']),Path(row['target'])])
        for key,sha in read(prov)['source_sha256'].items():
            if key.startswith('openpi/'):
                f=RT/'RoboTwin/policy/pi05/src'/key
            elif key=='integration/register_motus_config.py':
                f=RT/'pi05_integration/register_motus_config.py'
            else:continue
            if digest(f)!=sha:raise ValueError('Official policy source changed '+key)
            sources.append(f)
    tokenizer=resolve_tokenizer();weight=CHECKPOINT/'model.safetensors'
    identity=dict(base_checkpoint_sha256=digest(weight),normalization_sha256=digest(
        CHECKPOINT/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'),tokenizer_sha256=digest(tokenizer))
    if identity['base_checkpoint_sha256']!=read(FEATURES/'checkpoint_audit.json')['sha256']:
        raise ValueError('Feature checkpoint differs')
    for prov in provenances:
        if any(prov[k]!=v for k,v in identity.items()):raise ValueError('Source L0 policy identity differs')
    for name in ('feature_audit.json','checkpoint_audit.json'):
        if read(FEATURES/name)['status']!='passed':raise ValueError('Feature cache audit missing')
    encoded=next(r for r in read(FEATURES/'result.json')['records']
                 if r['source']['kind']=='expert' and r['source']['task']=='place_shoe' and r['source']['pilot_split']=='fit')
    sources.extend([Path(encoded['file']),Path(encoded['source']['source']),tokenizer,
                    CHECKPOINT/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'])
    stat=weight.stat();run=new_run('recovery',group=('feature_caches',));print(run,flush=True)
    write_json(run/'protocol.json',dict(status='frozen_before_inference',rows=rows,history_indices=history,
        inventory=str(INVENTORY),feature_cache=str(FEATURES),preprocess=str(PREPARED/'preprocess.npz'),
        fit_diagnostic=str(FIT),checkpoint=str(CHECKPOINT),tokenizer=str(tokenizer),identity=identity,
        checkpoint_stat=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns),golden=encoded,
        source_sha256={str(f):digest(f) for f in sorted(set(sources))},policy_seed=910000,
        noise='Official compiled sample_actions, internal Torch noise. One chronological stream per scene, '
              'reset to policy seed after one warm-up per distinct instruction. No best-of-N.',
        repeat='Restore complete Python/NumPy/Torch CPU/CUDA RNG and repeat every production query once. '
               'Keep the first result; log differences, never select by teacher match. Restore first post-state afterward.',
        feature_contract=read(FEATURES/'manifest.json')['feature'],
        forward_inputs='Current saved 3RGB, command_state float32, instruction only. Teacher commands, '
                       'plan IDs, task/scene identity and success are never passed to VLA/feature extractor.',
        alignment='Observation at native n -> actually executed teacher commands n+1..n+32, masked at terminal; '
                  'same teacher-trajectory observation for each new base query. No reuse after branch divergence.',
        history='Only previous saved teacher observation in the same source and scene; first history absent. '
                'Original expert-fit PCA/scaler reused without refitting on recovery.',
        scope='Six rows / two already studied failure scenes, 144 executed teacher commands. '
              'Interface/fit diagnostic only; no independent test, new collection, task success gain or promotion.'))
    print('PREREGISTERED',run,flush=True)

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
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Recovery cache exceeded 1200s')))
    signal.alarm(1200)
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
        for i,row in enumerate(p['rows']):
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
        history=np.array(p['history_indices']);context=context_from_features(np.stack(joint),np.stack(states),history,pre)
        np.savez_compressed(run/'data.npz',context=context,base=np.stack(actions),repeated_base=np.stack(repeats),
            target=np.stack([x['target'] for x in labels]),valid=np.stack([x['valid'] for x in labels]),
            source_native_counts=np.stack([x['source_native_counts'] for x in labels]),
            plan_ids=np.stack([x['plan_ids'] for x in labels]),history_indices=history)
        write_json(run/'result.json',dict(status='complete_pending_audit',rows=reports,observations=len(reports),
            groups=len({r['group'] for r in p['rows']}),valid_teacher_actions=sum(r['valid_actions'] for r in p['rows']),
            data_sha256=digest(run/'data.npz'),features_sha256=digest(run/'features.npz'),
            elapsed_seconds=time.monotonic()-started,training_scope='developer_recovery_fit_interface_only',scope=p['scope']))
        replace_json(run/'progress.json',dict(phase='complete_pending_audit',rows=len(reports)))
        print('COMPLETE_PENDING_AUDIT',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed=reports));raise
    finally:signal.alarm(0)

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run',type=Path);parser.add_argument('--gpu',default='0');args=parser.parse_args()
    if args.prepare_only:prepare()
    elif args.run is None or not args.gpu.isdigit():parser.error('Prepared --run and numeric --gpu required')
    else:cache(args.run.resolve(),args.gpu)

if __name__=='__main__':main()

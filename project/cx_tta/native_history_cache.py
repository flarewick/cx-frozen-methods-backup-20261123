"""S2n native L0 feature cache; recorded bases are never resampled.

Prepare on CPU, encode only missing prefixes on CUDA, then audit on CPU.
Labels/outcomes/source IDs remain outside the nine deployable input fields.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from .audit import digest
from .paths import ROOT, new_run, output_path, write_json, replace_json
from .structured_action_adapter import FORWARD_KEYS

ACCEPTED = ROOT/'outputs/reports/attribution/expert_history_accepted_20261005_021422_896290'
DESIGN = ACCEPTED/'next_experiments.json'
DESIGN_SHA = '2bb3e168a25d7f69397dc1b217f4593dff69f77222d3a99a8daf884bb7a8f271'
BASE = ROOT/'outputs/effect_learning/structured_action_20261003_095511_385074'
FIXED = ROOT/'outputs/feature_caches/fixed_recovery_20261003_031535_304147'
CURRENT = ('visual','text','state','ee_pose')


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k].copy() for k in data.files}


def require(value,message):
    if not value: raise ValueError(message)


def verify(sources):
    for name,sha in sources.items():
        require(digest(Path(name))==sha,'Changed dependency: '+name)


def feature_key(signature,instruction):
    return json.dumps([signature,instruction],sort_keys=True,separators=(',',':'))


def native_key(row):
    return row['task'],row['seed'],row['chunk']


def choose_rows(support,held):
    """Keep known successes and11 entry states, never all failed states as labels."""
    lookup={native_key(q):q for q in support['queries']}
    require(len(lookup)==len(support['queries']),'Duplicate source query')
    teachers={(t['task'],t['seed'],t['native_count']):t for t in support['first_L0_entry_matches']}
    rows=[];needed={}
    for source in support['queries']:
        identity=source['task'],source['seed'],source['native_count']
        if not source['source_success'] and identity not in teachers: continue
        current=native_key(source);previous=None
        if source['past_chunk'] is not None:
            previous=(source['task'],source['seed'],source['past_chunk'])
            require(previous in lookup,'Missing causal predecessor')
            old=lookup[previous]
            require(old['chunk']+1==source['chunk'] and old['native_count']+old['executed']==source['native_count']
                and old['executed']==32 and old['instruction']==source['instruction'],'Invalid32-action causal history')
            needed[previous]=old
        needed[current]=source
        is_held=(source['task'],source['seed']) in held
        require(not (is_held and identity in teachers),'Held-out teacher entered training candidates')
        rows.append(dict(source,current_key=list(current),past_key=None if previous is None else list(previous),
            split='development_preservation_heldout' if is_held else 'train_candidate',
            target_kind='success_preservation' if source['source_success'] else 'unqualified_teacher_entry',
            teacher_entry=teachers.get(identity),physical_recovery_benefit_qualified=False))
    return rows,[needed[k] for k in sorted(needed)]


def absent_history(inputs):
    require(set(inputs)==FORWARD_KEYS,'Exactly nine legal fields required')
    out={k:v.copy() for k,v in inputs.items()}
    for field in (*('past_'+key for key in CURRENT),'has_history'):out[field].fill(0)
    return out


def prepare():
    require(digest(DESIGN)==DESIGN_SHA,'Sealed S2n design changed')
    design=read(DESIGN);verify(design['source_sha256'])
    support=read(ACCEPTED/'support_audit.json');require(support['passed'],'Native source audit required')
    parent=read(BASE/'protocol.json');fp=read(FIXED/'protocol.json');fr=read(FIXED/'result.json')
    for name in ('protocol.json','features.npz','data.npz'):
        require(digest(FIXED/name)==parent['source_sha256'][str(FIXED/name)],'Changed fixed cache')
    require(read(FIXED/'audit.json')['passed'] and digest(FIXED/'features.npz')==fr['features_sha256'],
        'Audited prefix cache required')
    held={(x['task'],x['seed']) for x in design['whole_scene_holdouts']}
    rows,features=choose_rows(support,held)
    require(len(rows)==59 and len(features)==63,'Preregistered native source selection differs')
    lookup={native_key(row):i for i,row in enumerate(features)}
    for row in rows:
        row['current_feature']=lookup[tuple(row['current_key'])]
        row['past_feature']=-1 if row['past_key'] is None else lookup[tuple(row['past_key'])]
    cached=arrays(FIXED/'features.npz');require(len(cached['joint'])==len(fp['feature_rows'])==34,'Prefix metadata shape')
    old={}
    for i,row in enumerate(fp['feature_rows']):
        key=feature_key(row['observation_signature'],row['instruction'])
        if key in old:require(np.array_equal(cached['joint'][old[key]],cached['joint'][i]),'Conflicting existing prefix')
        old[key]=i
    for row in features:
        row['observation_signature']=row['signature']
        row['reuse_prefix_index']=old.get(feature_key(row['signature'],row['instruction']))
        if row['reuse_prefix_index'] is not None:
            state=arrays(row['observation'])['command_state']
            require(np.array_equal(state,cached['state'][row['reuse_prefix_index']]),'Reused prefix state differs')
    require(sum(x['reuse_prefix_index'] is not None for x in features)==15,'Existing prefix support changed')
    q=read(BASE/'training_protocol.json');config=Path(parent['config']);projection=Path(parent['projection'])
    from .robot_kinematics import RobotFK
    from .prepare_structured_action import fk_contract
    robots=[RobotFK(read(config),side) for side in ('left','right')]
    checked_configs=[]
    sources={}
    def dependency(path,expected=None):
        path=Path(path);sha=digest(path)
        require(expected is None or sha==expected,'Dependency mismatch: '+str(path))
        sources[str(path)]=sha
    for path in (DESIGN,ACCEPTED/'support_audit.json',BASE/'protocol.json',BASE/'training_protocol.json',
        config,projection,FIXED/'protocol.json',FIXED/'features.npz',FIXED/'audit.json',FIXED/'result.json'):
        dependency(path)
    for robot in robots:dependency(robot.urdf)
    for task,source in design['native_source_runs'].items():
        directory=Path(source);cfg=directory/'resolved_config.json';prov=read(directory/'provenance.json')
        require(all(prov[k]==v for k,v in fp['identity'].items()),'Source VLA identity differs')
        live=[RobotFK(read(cfg),side) for side in ('left','right')]
        require([fk_contract(x) for x in live]==[fk_contract(x) for x in robots],'Source robot FK differs')
        checked_configs.append(dict(task=task,path=str(cfg),FK_contract_exact=True))
        dependency(cfg);dependency(directory/'provenance.json')
    for row in features:
        dependency(row['observation'],row['observation_sha256'])
    for row in rows:
        dependency(row['base'],support['source_sha256'][row['base']])
        if row['teacher_entry'] is not None:
            target=row['teacher_entry'];dependency(target['target'],target['target_sha256'])
    golden=fp['golden']
    for path,sha in ((golden['file'],golden['sha256']),
        (golden['source']['source'],golden['source']['source_sha256'])):
        dependency(path,sha)
    dependency(fp['tokenizer'],fp['identity']['tokenizer_sha256'])
    norm=Path(fp['checkpoint'])/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    dependency(norm,fp['identity']['normalization_sha256'])
    for rel in ('cx_tta/native_history_cache.py','scripts/cache_native_history.sh',
        'tests/test_native_history_cache.py',
        'cx_tta/cache_joint_prefix.py','cx_tta/structured_features.py','cx_tta/robot_kinematics.py',
        'cx_tta/cache_expert.py','cx_tta/cache_recovery.py','cx_tta/structured_action_deploy.py',
        'cx_tta/evaluate_structured_low_storage.py'):
        dependency(ROOT/rel)
    run=new_run('native_history',group=('feature_caches',))
    write_json(run/'protocol.json',dict(status='frozen_before_encoding',stage='S2n_native_cache',
        design=str(DESIGN),design_sha256=DESIGN_SHA,rows=rows,feature_rows=features,
        inputs=list(sorted(FORWARD_KEYS)),feature_count=63,reused_prefix_count=15,new_prefix_count=48,
        projection=str(projection),robot_config=str(config),robot_configs_checked=checked_configs,
        golden=golden,checkpoint=fp['checkpoint'],checkpoint_stat=fp['checkpoint_stat'],identity=fp['identity'],
        tokenizer=fp['tokenizer'],source_sha256=sources,time_limit_s=900,maximum_new_run_bytes=20*2**20,
        original_caps=q['caps_native_units'],inherited_missing_source_count=33,
        storage='Projected fields/base/masks only. No raw RGB, full28672 prefix or backbone copy. Temporary caches removed at exit.',
        history='Same-source previous query after32 actual native actions, not32 saved expert frames',
        inference='Current-only deterministic eager auxiliary prefix; compiled sampler unchanged and NEVER CALLED. Recorded actual source bases retained bytewise.',
        labels='48 preservation source candidates;11 cached teacher entries unqualified for bounded recovery;55 failure queries are NOT recovery labels.',
        training_allowed=False,unqualified_teacher_targets_may_train=False))
    print('RUN',run,flush=True)
    return run


def encode_cache(run,gpu):
    run=output_path(run);p=read(run/'protocol.json');verify(p['source_sha256'])
    require(not (run/'result.json').exists(),'Completed cache is immutable')
    require(p['stage']=='S2n_native_cache' and not p['training_allowed'],'Wrong cache stage')
    weight=Path(p['checkpoint'])/'model.safetensors';st=weight.stat()
    require(dict(size=st.st_size,mtime_ns=st.st_mtime_ns)==p['checkpoint_stat'],'Base checkpoint stat changed')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu;os.environ['JAX_PLATFORMS']='cpu'
    from .evaluate_b2 import seed_all
    from .evaluate_structured_low_storage import PrivateEvalStorage
    from .cache_expert import create_base_policy
    from .cache_joint_prefix import encode
    from .cache_recovery import legal_observation
    from .recovery_contract import rng_capture,rng_fingerprint
    from .structured_features import pack_observable
    from .robot_kinematics import RobotFK
    import torch
    import h5py
    from .dataset import read_observation
    import atexit
    runtime=run/'runtime';runtime.mkdir()
    storage=PrivateEvalStorage()
    atexit.register(storage.cleanup)
    storage.configure_runtime(run,runtime,Path(p['tokenizer']))
    torch.set_num_threads(2)
    require(torch.cuda.is_available() and not torch._dynamo.config.disable,'CUDA device required; global compilation must remain enabled')
    started=time.monotonic();reports=[]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Native prefix cache900s bound')))
    signal.alarm(p['time_limit_s'])
    try:
        seed_all(0);policy=create_base_policy();policy._model.eval().requires_grad_(False)
        sampler=policy._sample_actions
        require(hasattr(sampler,'_torchdynamo_orig_callable'),'Official compiled sampler must remain installed')
        feature_calls=0
        eager=torch.compiler.disable(encode)
        def feature(images,state,prompt):
            nonlocal feature_calls
            before=rng_fingerprint(rng_capture());cfg=policy._model.paligemma_with_expert.paligemma.language_model.config
            implementation=cfg._attn_implementation
            try:value,meta=eager(policy,images,state,prompt)
            finally:cfg._attn_implementation=implementation
            require(before==rng_fingerprint(rng_capture()),'Prefix extraction consumed RNG')
            require(policy._sample_actions is sampler and not torch._dynamo.config.disable,'Compiled sampler altered')
            feature_calls+=1
            return value,meta
        golden=p['golden'];stored=arrays(golden['file']);frame=int(stored['frames'][0])
        with h5py.File(golden['source']['source'],'r') as h:images,state=read_observation(h,frame)
        actual,_=feature(images,state,golden['source']['instruction'])
        require(np.array_equal(actual,stored['joint'][0]),'Frozen prefix golden differs')
        fx=arrays(FIXED/'features.npz');joints=[];states=[];tested=set();reused_tested=set()
        for i,row in enumerate(p['feature_rows']):
            images,state,sig=legal_observation(row);reuse=row['reuse_prefix_index']
            if reuse is None:
                value,meta=feature(images,state,row['instruction'])
                if row['task'] not in tested:
                    repeated,_=feature(images,state,row['instruction'])
                    require(np.array_equal(value,repeated),'New prefix not repeatable');tested.add(row['task'])
            else:
                value=fx['joint'][reuse].copy();meta=dict(reused_index=reuse)
                if row['task'] not in reused_tested:
                    repeated,_=feature(images,state,row['instruction'])
                    require(np.array_equal(value,repeated),'Reused native prefix convention differs');reused_tested.add(row['task'])
            joints.append(value);states.append(state)
            reports.append(dict(feature=i,reused=reuse is not None,signature=sig,
                RNG_unchanged=True,**meta))
            replace_json(run/'progress.json',dict(phase='prefix_encoding',encoded=i+1,total=63))
            print('FEATURE',i+1,'/',63,'reused',reuse is not None,flush=True)
        pre=arrays(p['projection']);robots=[RobotFK(read(p['robot_config']),side) for side in ('left','right')]
        # Per-query packing matches deployment's matrix multiplication shape.
        current={key:[] for key in CURRENT};all_inputs={key:[] for key in FORWARD_KEYS}
        for i in range(len(joints)):
            block=pack_observable(joints[i][None],states[i][None],np.zeros_like(joints[i])[None],
                np.zeros_like(states[i])[None],np.asarray([False]),pre,robots)
            for key in CURRENT:current[key].append(block[key][0])
        current={key:np.stack(values) for key,values in current.items()}
        bases=[];executed=[]
        from .eval_diagnostics import array_hash
        for row in p['rows']:
            i,j=row['current_feature'],row['past_feature'];has=j>=0
            block=pack_observable(joints[i][None],states[i][None],
                (joints[j] if has else np.zeros_like(joints[i]))[None],
                (states[j] if has else np.zeros_like(states[i]))[None],np.asarray([has]),pre,robots)
            for key in FORWARD_KEYS:all_inputs[key].append(block[key][0])
            base=arrays(row['base'])['base_action'];require(array_hash(base)==row['base_sha256'],'Recorded base array changed')
            bases.append(base);executed.append(row['executed'])
        np.savez_compressed(run/'current_features.npz',**current)
        np.savez_compressed(run/'inputs.npz',**{key:np.stack(value) for key,value in all_inputs.items()})
        np.savez_compressed(run/'recorded_bases.npz',base=np.stack(bases),executed=np.asarray(executed),
            source_valid=np.arange(32)[None]<np.asarray(executed)[:,None])
        write_json(run/'feature_checks.json',reports)
        write_json(run/'result.json',dict(status='complete_pending_CPU_audit',rows=59,features=63,
            encoded_missing=48,reused=15,actual_prefix_calls=feature_calls,action_sampler_calls=0,
            golden_exact=True,reused_native_prefix_exact_tasks=sorted(reused_tested),
            new_prefix_repeat_exact_tasks=sorted(tested),elapsed_s=time.monotonic()-started,
            gpu=torch.cuda.get_device_name(0),compiled_sampler_preserved=True,
            training_allowed=False,recovery_labels_qualified=False,scope='Native feature/base interface, not model improvement or new physical benefit',
            source_sha256={str(run/name):digest(run/name) for name in
                ('protocol.json','inputs.npz','current_features.npz','recorded_bases.npz','feature_checks.json')}))
        require(sum(path.stat().st_size for path in run.rglob('*') if path.is_file())<=p['maximum_new_run_bytes'],
            'Cache storage bound exceeded')
        print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_features=len(reports)));raise
    finally:
        signal.alarm(0)
        # PrivateEvalStorage removes only this job's short scratch at process
        # exit, after any Torch teardown; original sources remain immutable.


def audit_cache(run):
    run=output_path(run);p=read(run/'protocol.json');r=read(run/'result.json')
    verify(p['source_sha256']);verify(r['source_sha256'])
    x=arrays(run/'inputs.npz');cur=arrays(run/'current_features.npz');base=arrays(run/'recorded_bases.npz')
    require(set(x)==FORWARD_KEYS and set(cur)==set(CURRENT),'Illegal model fields')
    held_train=0;preservation=0;held=0;teacher=0
    for n,row in enumerate(p['rows']):
        i,j=row['current_feature'],row['past_feature'];has=j>=0
        require(bool(x['has_history'][n].item())==has,'History availability mismatch')
        for key in CURRENT:
            require(x[key][n].tobytes()==cur[key][i].tobytes(),'Current field mismatch')
            expected=cur[key][j] if has else np.zeros_like(cur[key][i])
            require(x['past_'+key][n].tobytes()==expected.tobytes(),'Causal past field mismatch')
        require(base['base'][n].tobytes()==arrays(row['base'])['base_action'].tobytes(),'Actual base not retained bytewise')
        require(base['source_valid'][n].tolist()==row['target_tail_mask'],'Tail mask differs from actual execution')
        if row['split']=='development_preservation_heldout':held+=1
        elif row['target_kind']=='success_preservation':preservation+=1
        else:teacher+=1
        require(not row['physical_recovery_benefit_qualified'],'Unexpected recovery qualification')
    control=absent_history(x)
    for key in CURRENT:require(control[key].tobytes()==x[key].tobytes(),'Forbidden control change')
    require((preservation,held,teacher)==(40,8,11),'Native candidate splits changed')
    require(r['action_sampler_calls']==0 and r['actual_prefix_calls']==53,'Unexpected action or feature sampling')
    write_json(run/'audit.json',dict(passed=True,source_rows=59,source_features=63,actual_native_history_rows=int(x['has_history'].sum()),
        train_preservation_candidates=40,whole_scene_heldout_rows=8,unqualified_teacher_entries=11,
        current_past_bytewise_verified=True,actual_recorded_bases_bytewise_retained=True,
        valid_tail_uses_actual_execution=True,nine_legal_fields_only=True,absent_history_control_only_changes_past=True,
        action_sampler_calls=0,training_allowed=False,recovery_benefit_qualified=False,
        original_source_gaps=33,new_source_gaps=0,source_sha256={str(run/name):digest(run/name) for name in
            ('protocol.json','result.json','inputs.npz','current_features.npz','recorded_bases.npz','feature_checks.json')}))
    print('AUDIT_PASSED',run,flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--run',type=Path)
    parser.add_argument('--audit',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--gpu',default='0');args=parser.parse_args()
    if not args.gpu.isdigit():parser.error('Numeric GPU required')
    if args.prepare_only:
        if args.run is not None or args.audit or args.dry_run:parser.error('Prepare mode takes no existing run')
        prepare();return
    if args.run is None:parser.error('--run required')
    run=output_path(args.run)
    if args.dry_run:
        p=read(run/'protocol.json');verify(p['source_sha256'])
        print(json.dumps(dict(run=str(run),features=63,new_features=48,reused=15,action_queries=0,training_allowed=False)));return
    if args.audit:audit_cache(run)
    else:encode_cache(run,args.gpu)


if __name__=='__main__':main()

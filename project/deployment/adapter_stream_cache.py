"""Stage original per-frame training caches while qualified demos are collected."""
import ast,copy,hashlib,importlib.metadata,inspect,json,os,platform,sys,threading,time
from pathlib import Path
D=Path(__file__).resolve().parent
TRAINING_SOURCE=D.parent/"multi_event_gate_v1/task_adapter_v1/task_adapters/training.py"
_EXPERT_TARGETS=None
_FORWARD_KEYS=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def atomic_json(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def bundle_path():
    if os.environ.get('CX_STREAM_BUNDLE'):return Path(os.environ['CX_STREAM_BUNDLE'])
    wanted=os.environ.get('CX_TARGET_GPU_UUID','GPU-2847d2ca-7b83-3b84-b70c-7b44e288c156')
    from compiler_bundle import read_bundle
    matches=[]
    for path in (D/'.work/compiler_bundles').glob('*/*/bundle.json'):
        meta,_=read_bundle(path.parent)
        if meta['fingerprint']['gpu_uuid']==wanted and meta['fingerprint']['gemm_backends']=='ATEN':matches.append(path.parent)
    if len(matches)!=1:raise RuntimeError('Choose one certified sampler bundle explicitly')
    return matches[0]

def make_record(row):
    import h5py
    import numpy as np
    global _EXPERT_TARGETS
    if _EXPERT_TARGETS is None:
        tree=ast.parse(TRAINING_SOURCE.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='expert_targets')
        ns=dict(np=np);exec(compile(ast.Module(body=[node],type_ignores=[]),str(TRAINING_SOURCE),'exec'),ns);_EXPERT_TARGETS=ns['expert_targets']
    expert_targets=_EXPERT_TARGETS
    source=Path(row['hdf5'])
    with h5py.File(source,'r')as h:
        n=len(h['joint_action/vector']);frames=[];valid_steps=0
        for i in range(0,n-1,32):
            if not np.any(h['cx_effect/transition_dt_s'][i:i+32]>0):continue
            _,valid=expert_targets(h,i);valid_steps+=int(valid.sum());frames.append(i)
        if not frames:raise ValueError('No usable original expert rows')
        return dict(task=row['task'],split=row['split'],seed=row['seed'],source=str(source),source_sha256=row['capture']['source_sha256'],frames=frames,valid_steps=valid_steps,instruction=str(h.attrs['instruction']),saved_frames=n)

def rows(run,p):
    for task in p['tasks']:
        for split in ('adapter_fit','adapter_validation'):
            file=run/'collections'/task/split/'attempts.json'
            if not file.exists():continue
            try:data=json.loads(file.read_text())
            except json.JSONDecodeError:continue  # original writer replaces its JSON synchronously
            for row in data:
                if row.get('accepted'):
                    if row['task']!=task or row['split']!=split:raise RuntimeError('Collection row identity changed')
                    yield row

def target(root,record):return root/record['task']/record['split']/('seed_'+str(record['seed'])+'.npz')

def valid(root,record,backend):
    import numpy as np
    path=target(root,record);meta=path.with_suffix('.json')
    if not meta.exists():return False
    data=json.loads(meta.read_text())
    if data['backend_id']!=backend or data['record']!=record or sha(path)!=data['sha256']:raise RuntimeError('Staged cache identity/content changed')
    with np.load(path,allow_pickle=False)as z:
        global _FORWARD_KEYS
        if _FORWARD_KEYS is None:
            source=D.parent/'multi_event_gate_v1/cx_tta/structured_action_adapter.py'
            node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='FORWARD_KEYS'for t in n.targets))
            if not isinstance(node.value,ast.Call)or ast.unparse(node.value.func)!='frozenset'or len(node.value.args)!=1:raise RuntimeError('Original forward key definition changed')
            _FORWARD_KEYS=frozenset(ast.literal_eval(node.value.args[0]))
        FORWARD_KEYS=_FORWARD_KEYS
        if set(z.files)!=set(FORWARD_KEYS)|{'base','target','valid','frame_index'}or not np.array_equal(z['frame_index'],record['frames']):raise RuntimeError('Staged cache fields/frames changed')
        n=len(record['frames'])
        if z['base'].shape!=(n,32,14)or z['target'].shape!=(n,32,14)or z['valid'].shape!=(n,32)or z['valid'].dtype!=np.bool_ or not z['valid'].any(1).all():raise RuntimeError('Original staged target/mask shape changed')
        if any(z[k].any()for k in FORWARD_KEYS if k.startswith('past_')or k=='has_history'):raise RuntimeError('Invented native history in expert cache')
        if any(not np.isfinite(z[k]).all()for k in z.files):raise RuntimeError('Nonfinite staged training cache')
    return True

def original_base_identity(tokenizer):
    # Run the exact original identity function without importing unrelated policy
    # classes that start allocator threads in the CPU-only preparation parent.
    from cx_tta.cache_expert import CHECKPOINT
    from cx_tta.paths import ROOT
    from cx_tta.audit import digest
    source=D.parent/'multi_event_gate_v1/cx_tta/evaluate_structured_action.py'
    tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('json_read','base_identity')or isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='BASE_EVIDENCE'for t in n.targets)]
    if len(nodes)!=3:raise RuntimeError('Original base identity source structure changed')
    ns=dict(Path=Path,json=json,CHECKPOINT=CHECKPOINT,ROOT=ROOT,digest=digest)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    return ns['base_identity'](tokenizer)

def prepare(run,p,pool):
    from task_adapters.protocol import TEMPLATE
    from compiler_bundle import read_bundle
    proof=json.loads((D/'reports/target_adapter_stream_cpu_20261024.json').read_text())
    if not proof['passed']or proof['stream_helper_sha256']!=sha(__file__):raise RuntimeError('Stream query CPU proof stale')
    template=json.loads(TEMPLATE.read_text());base=original_base_identity(Path(template['tokenizer']))
    for k in ('checkpoint','base_checkpoint_sha256','normalization_sha256','tokenizer_sha256'):
        if base[k]!=template['policy_identity'][k]:raise RuntimeError('Original frozen base identity differs: '+k)
    bundle=bundle_path();meta,artifacts=read_bundle(bundle);fingerprint=meta['fingerprint']
    # Verify EVERY source declared by the original sampler bundle; its evaluation
    # result identity is never imported as data. No old source check is disabled.
    for name,expected in fingerprint['frozen_sources'].items():
        if sha(name)!=expected:raise RuntimeError('Original sampler bundle source changed: '+name)
    live=fingerprint['policy_identity']['live_identity']
    for k in ('checkpoint','base_checkpoint_sha256','normalization_sha256','tokenizer_sha256'):
        if live[k]!=template['policy_identity'][k]:raise RuntimeError('Sampler bundle/frozen training model mismatch: '+k)
    backend=dict(schema='qualified_expert_stream_cache_v1',protocol_sha256=sha(run/'protocol.json'),sampler_bundle_artifact_sha256=meta['artifact_sha256'],sampler_bundle_fingerprint_sha256=meta['fingerprint_sha256'],gpu_uuid=fingerprint['gpu_uuid'],query_source_sha256=sha(TRAINING_SOURCE),stream_helper_sha256=sha(__file__),rng_helper_sha256=sha(D/'concurrent_rng.py'),model_service_sha256=sha(D/'model_service.py'),scientific_sources_sha256=hashlib.sha256(canonical(p['source_sha256'])).hexdigest(),projection_sha256=sha(template['projection']),robot_config_sha256=sha(template['robot_config']),policy_identity=template['policy_identity'],precision='original BF16/FP32 mix; compiled max-autotune ATEN; no TTA',history_mode='current_only',original_per_frame_seed=True)
    key=hashlib.sha256(canonical(backend)).hexdigest();root=run/'stream_cache'/key;root.mkdir(parents=True,exist_ok=True)
    manifest=root/'backend.json'
    if manifest.exists()and json.loads(manifest.read_text())!=backend:raise RuntimeError('Staged backend manifest changed')
    if not manifest.exists():atomic_json(manifest,backend)
    records=[make_record(row)for row in rows(run,p)];done=[r for r in records if valid(root,r,key)]
    active=run/'stream_cache_active.json'
    if active.exists():
        old=json.loads(active.read_text())
        if old.get('backend_id')!=key:atomic_json(root/'previous_backend.json',old)
    atomic_json(active,dict(backend_id=key,root=str(root),gpu_uuid=fingerprint['gpu_uuid'],bundle=str(bundle)))
    result=dict(template=template,bundle_meta=meta,bundle_bytes=artifacts,backend=backend,backend_id=key,root=str(root),records=records,done=done)
    atomic_json(pool/'stream_cpu_preparation.json',dict(backend=backend,backend_id=key,root=str(root),initial_qualified_scenes=len(records),initial_cached_scenes=len(done),all_bundle_sources_verified=True))
    return result

def original_query():
    from task_adapters import training
    import numpy as np,torch
    from cx_tta.dataset import read_observation
    from cx_tta.structured_features import pack_observable
    from cx_tta.evaluate_b2 import seed_all
    from cx_tta.recovery_contract import rng_capture,rng_fingerprint
    node=ast.parse(inspect.getsource(training.cache)).body[0]
    frame=next(n for n in ast.walk(node)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='frame')
    body=[];appends=0
    for n in frame.body:
        if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='append':appends+=1
        else:body.append(n)
    if appends!=4:raise RuntimeError('Original cache loop structure changed')
    wrapper=ast.parse('def query(h,frame,row,policy,sampler,eager,projection,robots):\n pass').body[0]
    wrapper.body=body+ast.parse('return x,base,target,valid').body
    ns=dict(np=np,torch=torch,hashlib=hashlib,read_observation=read_observation,pack_observable=pack_observable,seed_all=seed_all,rng_capture=rng_capture,rng_fingerprint=rng_fingerprint,expert_targets=training.expert_targets)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(D/'adapter_stream_cache.py'),'exec'),ns)
    return ns['query']

def initialize(prepared,c,pool):
    import torch
    from compiler_bundle import read_bundle
    f=prepared['bundle_meta']['fingerprint'];prop=torch.cuda.get_device_properties(0)
    actual=dict(schema=1,python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,triton=importlib.metadata.version('triton'),gpu_uuid=os.environ['CX_TARGET_GPU_UUID'],gpu_name=prop.name,capability=[prop.major,prop.minor],driver=os.environ['CX_TARGET_DRIVER_VERSION'],gemm_backends=os.environ.get('TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS','ATEN,TRITON,CPP'),compile_threads=int(os.environ.get('TORCHINDUCTOR_COMPILE_THREADS','0')),compile_mode='max-autotune',float32_matmul_precision='high',loading_helper_sha256=sha(D/'startup_loading.py'))
    for k,v in actual.items():
        if f[k]!=v:raise RuntimeError('Live frozen sampler compiler mismatch: '+k)
    from torch._dynamo.utils import counters
    before=torch.get_rng_state().clone();before_cuda=[s.clone()for s in torch.cuda.get_rng_state_all()]
    info=torch.compiler.load_cache_artifacts(prepared['bundle_bytes'])
    if info is None or not torch.equal(before,torch.get_rng_state())or any(not torch.equal(a,b)for a,b in zip(before_cuda,torch.cuda.get_rng_state_all())):raise RuntimeError('Sampler artifact restore failed/RNG changed')
    from autotune_relocation import AutotuneRelocation
    relocation=AutotuneRelocation(os.environ['TORCHINDUCTOR_CACHE_DIR']);relocation.install()
    from startup_loading import skip_overwritten_random_init
    from cx_tta.cache_expert import create_base_policy
    rt=Path('/data/cj/robo_pi/250010033/qfw/robotwin');sys.path.insert(0,str(rt/'RoboTwin/policy/pi05/src'));sys.path.insert(0,str(rt/'pi05_integration'))
    from openpi.models.model import BaseModelConfig
    old=BaseModelConfig.load_pytorch
    def load(self,*args,**kwargs):
        with skip_overwritten_random_init():return old(self,*args,**kwargs)
    BaseModelConfig.load_pytorch=load
    try:policy=create_base_policy()
    finally:BaseModelConfig.load_pytorch=old
    policy._model.eval().requires_grad_(False)
    if not hasattr(policy._sample_actions,'_torchdynamo_orig_callable'):raise RuntimeError('Original compiled frozen sampler required')
    from shared_weights import versions
    c.base=policy;c.weight_versions=versions(policy._model)
    import numpy as np
    from cx_tta.cache_joint_prefix import encode
    from cx_tta.robot_kinematics import RobotFK
    template=prepared['template'];projection=dict(np.load(template['projection'],allow_pickle=False));config=json.loads(Path(template['robot_config']).read_text());robots=[RobotFK(config,s)for s in ('left','right')]
    atomic_json(pool/'stream_hardware.json',dict(**actual,model_frozen=True,backbone_trainable=sum(v.numel()for v in policy._model.parameters()if v.requires_grad),sampler_artifact_sha256=prepared['bundle_meta']['artifact_sha256'],original_training_query_ast=True))
    c.note('stream_model_ready',initial_cached=len(prepared['done']),initial_qualified=len(prepared['records']))
    return policy,policy._sample_actions,torch.compiler.disable(encode),projection,robots,relocation

def work(run,p,prepared,c,pool,collection_done,model):
    import h5py,numpy as np
    from concurrent_rng import Lane
    from task_adapters.training import atomic_npz
    from cx_tta.recovery_contract import rng_capture,rng_fingerprint
    root=Path(prepared['root']);backend=prepared['backend_id'];query=original_query();lane=Lane('training_cache');known={(r['task'],r['split'],r['seed']):r for r in prepared['records']};done={(r['task'],r['split'],r['seed'])for r in prepared['done']};parity=0
    def status():atomic_json(root/'progress.json',dict(cached=len(done),requested=sum(p['splits'][s]['episodes']for s in ('adapter_fit','adapter_validation'))*len(p['tasks']),backend_id=backend,closed_loop_evaluation=False))
    with c.rng.bind(lane):
        while not c.cancel.is_set():
            for row in rows(run,p):
                key=(row['task'],row['split'],row['seed'])
                if key not in known:known[key]=make_record(row)
                if known[key]['source_sha256']!=row['capture']['source_sha256']:raise RuntimeError('Accepted source changed during streaming')
            todo=[(k,r)for k,r in known.items()if k not in done]
            if not todo:
                status()
                if collection_done.is_set():
                    expected=sum(p['splits'][s]['episodes']for s in ('adapter_fit','adapter_validation'))*len(p['tasks'])
                    if len(done)!=expected:raise RuntimeError('Collection ended without all required qualified training caches')
                    atomic_json(root/'complete.json',dict(complete=True,scenes=len(done),backend_id=backend,original_query=True));return
                c.cancel.wait(.2);continue
            for key,record in todo:
                c.check_cancel();path=target(root,record);path.parent.mkdir(parents=True,exist_ok=True)
                if valid(root,record,backend):done.add(key);continue
                if sha(record['source'])!=record['source_sha256']:raise RuntimeError('Qualified HDF5 changed before frozen query')
                xs=[];bases=[];targets=[];masks=[]
                with h5py.File(record['source'],'r')as h:
                    for frame in record['frames']:
                        def invoke():
                            nonlocal parity
                            result=query(h,frame,record,*model[:5]);c.verify_weights()
                            if parity<3:
                                after=rng_fingerprint(rng_capture());reference=query(h,frame,record,*model[:5]);check=rng_fingerprint(rng_capture())
                                if after!=check or any(not np.array_equal(result[0][k],reference[0][k])for k in result[0])or any(not np.array_equal(a,b)for a,b in zip(result[1:],reference[1:])):raise RuntimeError('Original repeated cache query parity failed')
                                parity+=1;atomic_json(pool/'stream_query_parity.json',dict(passed=True,frames=parity,features_base_targets_masks_exact=True,original_query_sha256=sha(TRAINING_SOURCE),same_source=True))
                            return result
                        invoke._cx_priority=1;invoke._cx_stage='original_training_cache_query'
                        x,base,target_value,valid_value=c.call(lane,invoke);xs.append(x);bases.append(base);targets.append(target_value);masks.append(valid_value)
                arrays={k:np.concatenate([x[k]for x in xs])for k in xs[0]};atomic_npz(path,**arrays,base=np.stack(bases),target=np.stack(targets),valid=np.stack(masks),frame_index=np.asarray(record['frames']))
                atomic_json(path.with_suffix('.json'),dict(backend_id=backend,record=record,sha256=sha(path),session=str(pool),original_query_ast=True))
                if not valid(root,record,backend):raise RuntimeError('New original cache did not validate')
                done.add(key);status();c.note('stream_scene_cached',task=record['task'],split=record['split'],seed=record['seed'],frames=len(record['frames']),cached=len(done))

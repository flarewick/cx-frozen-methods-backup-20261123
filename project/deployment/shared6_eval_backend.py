"""Independent trained-six-task evaluation identities and frozen-model cache checks."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent
FILES=('shared6_eval_backend.py','shared6_eval_worker.py','shared6_eval_core.py','shared6_eval_prepared.py','shared6_planner_pool.py','shared6_screen_reference.py','shared6_render_sync.py','render_sync_materialization.py','adapter_cache_cpu.py','startup_loading.py','native_generic_physics.py','native_generic_control.cpp','native_render.py','observation_materialization.py','concurrent_core.py','concurrent_rng.py','model_service.py','concurrent_queue.py','scene_gate.py','frozen_factory.py','collection_capture_gate.py','collection_gc_schedule.py','collection_optional_open3d.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def sources():return {name:sha(D/name)for name in FILES}
def decorate(raw,uuid,lanes):
    from cx_tta.evaluation_resume import identity
    backend=dict(protocol='trained_shared6_original_closed_loop_v1',gpu_uuid=uuid,lanes=lanes,condition_lanes=lanes//2,sampler_batch=1,compile_mode='max-autotune',gemm_backends='ATEN',private_planner_pairs=False,owner_only_original_expert_pair=True,first_expert_pair_eager_original_constructors=True,policy_mutable_planner_access_forbidden=True,shared_immutable_vla=True,original_success_callback_every_physics_tick=True,physics_visual_sync='defer only camera-pose/render tail until every original observation; all light/RNG prefixes retained',all_original_camera_calls=True,native_binary_sha256=json.loads((D/'.work/native_generic_control/build.json').read_text())['binary_sha256'],sources=sources(),journal='original ordered prefix with validated durable whole out-of-order episodes',compiler_artifacts='source/runtime/UUID checked; process-session cache provenance; no output/data reuse',rng='independent Python/NumPy/Torch per scene, serialized native sampler owner')
    manifest=dict(raw,deployment_shared6=backend);manifest['evaluation_identity']=identity(manifest);return manifest

def make_plan(task,training_run,uuid,lanes=4,episodes=100):
    from task_adapters import eval_resume as original
    from cx_tta.evaluation_resume import load_journal
    if lanes not in(4,6)or episodes!=100:raise ValueError('Formal original100 and bounded even scene count required')
    jobs=[]
    for condition in('Z_zero','M_event_full'):
        args=original.options(['--task',task,'--training-run',str(training_run),'--condition',condition,'--episodes',str(episodes),'--seed','0','--task-config','demo_randomized'])
        protocol,reference,tokenizer,raw=original.preflight(args);manifest=decorate(raw,uuid,lanes);run=original.choose_run(args,manifest);state=load_journal(run,manifest)
        jobs.append(dict(condition=condition,run=str(run),identity=manifest['evaluation_identity'],completed=len(state['episodes']),complete=len(state['episodes'])==episodes,previous_sessions=sorted(p.name for p in(run/'sessions').glob('process_*')),manifest=manifest,protocol=protocol,reference=str(reference),tokenizer=str(tokenizer)))
    return dict(task=task,training_run=str(training_run),gpu_uuid=uuid,lanes=lanes,episodes=episodes,jobs=jobs,source_sha256=sources())

def check_bundle(directory,training_plan,uuid):
    from compiler_bundle import read_bundle
    meta,data=read_bundle(directory);f=meta['fingerprint'];base=f['policy_identity'].get('live_identity',f['policy_identity'])
    if f['gpu_uuid']!=uuid or f['gemm_backends']!='ATEN'or f['compile_mode']!='max-autotune':raise RuntimeError('Compiler artifact GPU/backend mismatch')
    for key in('checkpoint','base_checkpoint_sha256','normalization_sha256','tokenizer_sha256'):
        if base[key]!=training_plan['policy_identity'][key]:raise RuntimeError('Frozen sampler identity differs: '+key)
    for name,expected in f['frozen_sources'].items():
        if sha(name)!=expected:raise RuntimeError('Original compiler source changed: '+name)
    return meta,data

def live_fingerprint(manifest,training_plan):
    import platform,importlib.metadata,torch
    prop=torch.cuda.get_device_properties(0)
    frozen=dict(manifest['source_sha256']);root=Path(manifest['runtime_source'])
    frozen.update({str(root/relative):value for relative,value in manifest['task_runtime_sha256'].items()})
    return dict(schema=1,python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,triton=importlib.metadata.version('triton'),gpu_uuid=os.environ['CX_TARGET_GPU_UUID'],gpu_name=prop.name,capability=[prop.major,prop.minor],driver=os.environ['CX_TARGET_DRIVER_VERSION'],gemm_backends='ATEN',compile_threads=16,compile_mode='max-autotune',float32_matmul_precision='high',policy_identity=training_plan['policy_identity'],frozen_sources=frozen,loading_helper_sha256=sha(D/'startup_loading.py'),model_cache_scope='trained_shared6_frozen_sampler')

def restore_cache(bundle,training_plan,uuid):
    import torch
    from torch._dynamo.utils import counters
    if not bundle:return None
    meta,data=check_bundle(bundle,training_plan,uuid);f=meta['fingerprint'];prop=torch.cuda.get_device_properties(0)
    import platform,importlib.metadata
    actual=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,triton=importlib.metadata.version('triton'),gpu_name=prop.name,capability=[prop.major,prop.minor],driver=os.environ['CX_TARGET_DRIVER_VERSION'],compile_threads=16,float32_matmul_precision='high',loading_helper_sha256=sha(D/'startup_loading.py'))
    if any(f[k]!=v for k,v in actual.items()):raise RuntimeError('Live compiler runtime changed')
    cpu=torch.get_rng_state().clone();cuda=[s.clone()for s in torch.cuda.get_rng_state_all()];info=torch.compiler.load_cache_artifacts(data)
    if info is None or not torch.equal(cpu,torch.get_rng_state())or any(not torch.equal(a,b)for a,b in zip(cuda,torch.cuda.get_rng_state_all())):raise RuntimeError('Compiler restore failed or changed RNG')
    return dict(artifact_sha256=meta['artifact_sha256'],fingerprint_sha256=meta['fingerprint_sha256'],source_closure_verified=True,rng_unchanged=True)

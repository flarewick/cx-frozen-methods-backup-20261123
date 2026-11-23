"""Proven CPU frozen policy preparation; first GPU queries match retained originals."""
import ast,hashlib,json,os,time
from pathlib import Path
D=Path(__file__).resolve().parent
POLICY=None
STATE_SHA=None

def prepare(pool,prepared):
    global POLICY,STATE_SHA
    import torch
    from adapter_cache_cpu import load_cpu_policy,state_digest
    rows=[l for l in(D/'reports/target_adapter_cache_cpu_arrow_nvml_20261024.log').read_text().splitlines()if l.startswith('ADAPTER_CACHE_CPU_PROOF ')]
    proof=json.loads(rows[-1].split(' ',1)[1])
    if not proof['passed']or not proof['cpu_fork_child_passed']or proof['os_threads']!=1 or proof['cuda_initialized']or proof['helper_sha256']!=hashlib.sha256((D/'adapter_cache_cpu.py').read_bytes()).hexdigest():raise RuntimeError('CPU frozen-policy proof stale')
    for key,value in dict(JE_ARROW_MALLOC_CONF='background_thread:false',PYTORCH_NVML_BASED_CUDA_CHECK='1',TORCH_WARM_POOL='0').items():
        if os.environ.get(key)!=value:raise RuntimeError('CPU preload environment differs: '+key)
    began=time.monotonic();POLICY,meta=load_cpu_policy(verify_state=True);state=state_digest(POLICY._model)
    if state!=proof['converted_state_sha256']:raise RuntimeError('Converted original frozen weights differ')
    STATE_SHA=state
    threads=len(list(Path('/proc/self/task').iterdir()))
    if threads!=1 or torch.cuda.is_initialized():raise RuntimeError('CPU frozen policy is not fork safe')
    result=dict(passed=True,cpu_only=True,cuda_initialized=False,os_threads=threads,seconds=round(time.monotonic()-began,3),converted_state_sha256=state,**meta)
    (pool/'cpu_preloaded_policy.json').write_text(json.dumps(result,indent=2)+'\n')
    reused=certified(prepared,state)
    (pool/'cpu_preload_reuse_check.json').write_text(json.dumps(dict(matching_original_gpu_certificate=bool(reused)))+'\n')
    return result

def compare_frame(result,cache,index):
    import numpy as np
    x,base,target,mask=result
    if any(not np.array_equal(value,cache[key][index:index+1])for key,value in x.items())or any(not np.array_equal(value,cache[key][index])for key,value in zip(('base','target','valid'),(base,target,mask))):raise RuntimeError('CPU-preloaded GPU query differs from retained original cache')

def contract_for(prepared,state):
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    tree=ast.parse(Path(__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='initialize');prefix=[]
    for node in fn.body:
        prefix.append(node)
        if isinstance(node,ast.If)and isinstance(node.test,ast.UnaryOp)and isinstance(node.test.operand,ast.Name)and node.test.operand.id=='used':break
    if not isinstance(prefix[-1],ast.If):raise RuntimeError('Device-transfer contract structure changed')
    compare=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='compare_frame');meta=prepared['bundle_meta'];f=meta['fingerprint'];runtime={k:f[k]for k in ('python','torch','cuda','triton','gpu_uuid','driver','gemm_backends','compile_threads','compile_mode','float32_matmul_precision')};runtime['sampler_artifact_sha256']=meta['artifact_sha256']
    return dict(transfer_prefix_sha256=hashlib.sha256(ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False).encode()).hexdigest(),comparator_sha256=hashlib.sha256(ast.dump(compare,include_attributes=False).encode()).hexdigest(),cpu_loader_sha256=sha(D/'adapter_cache_cpu.py'),original_stream_helper_sha256=sha(D/'adapter_stream_cache.py'),original_training_query_sha256=sha(D.parent/'multi_event_gate_v1/task_adapter_v1/task_adapters/training.py'),converted_state_sha256=state,backend_id=prepared['backend_id'],runtime=runtime)

def certified(prepared,state):
    path=D/'reports/target_cpu_preload_gpu_parity_gpu1_20261024.json'
    if not path.exists():return None
    proof=json.loads(path.read_text())
    if proof.get('contract')!=contract_for(prepared,state):return None
    reference=Path(proof['reference']);allowed=Path(prepared['root']).parent.parent/'collection_pools'
    if not reference.resolve().is_relative_to(allowed.resolve())or reference.is_symlink():raise RuntimeError('GPU parity reference escapes retained pools')
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=proof['reference_sha256']:raise RuntimeError('Original GPU parity evidence changed')
    result=json.loads(reference.read_text())
    if proof.get('passed')is not True or proof.get('actual_gpu')is not True or result.get('passed')is not True or len(result['frames'])!=3 or result.get('features_base_targets_masks_exact')is not True or result['reference_backend_id']!=prepared['backend_id']:raise RuntimeError('GPU parity certificate incomplete')
    return proof

def initialize(prepared,c,pool):
    global POLICY
    if POLICY is None:raise RuntimeError('CPU policy was not prepared before GPU admission')
    from cx_tta import cache_expert
    from adapter_cache_cpu import move_to_gpu
    from adapter_stream_cache import initialize as original_initialize,original_query,target,valid
    saved=cache_expert.create_base_policy;policy=POLICY;POLICY=None;used=False
    def ready():
        nonlocal used
        if used:raise RuntimeError('CPU policy consumed twice')
        used=True;return move_to_gpu(policy)
    cache_expert.create_base_policy=ready
    try:model=original_initialize(prepared,c,pool)
    finally:cache_expert.create_base_policy=saved
    if not used:raise RuntimeError('Original initialization did not use the prepared policy')
    certificate=certified(prepared,STATE_SHA)
    if certificate:
        (pool/'cpu_preload_gpu_parity_reused.json').write_text(json.dumps(dict(passed=True,reference=certificate['reference'],contract=certificate['contract'],matching_source_weights_runtime=True),indent=2)+'\n')
        c.note('cpu_preload_original_gpu_parity_reused',passed=True,frames=3)
        return model
    # A finite correctness check against prior original GPU outputs is required
    # for this new device-transfer path; its startup samples all remain counted.
    import h5py,numpy as np
    from concurrent_rng import Lane
    lane=Lane('cpu_preload_original_gpu_parity');query=original_query();checked=[]
    for record in prepared['done']:
        if not valid(Path(prepared['root']),record,prepared['backend_id']):raise RuntimeError('Retained reference cache changed')
        with h5py.File(record['source'],'r')as source,np.load(target(Path(prepared['root']),record),allow_pickle=False)as cache:
            for index,frame in enumerate(record['frames']):
                result=c.call(lane,query,source,frame,record,*model[:5]);compare_frame(result,cache,index);c.verify_weights();checked.append(dict(task=record['task'],seed=record['seed'],frame=frame))
                if len(checked)==3:break
        if len(checked)==3:break
    if len(checked)!=3:raise RuntimeError('Three retained original GPU reference frames required')
    result=dict(passed=True,frames=checked,features_base_targets_masks_exact=True,original_cache_bytes_unchanged=True,separate_rng_lane=True,reference_backend_id=prepared['backend_id'])
    (pool/'cpu_preload_gpu_parity.json').write_text(json.dumps(result,indent=2)+'\n');c.note('cpu_preload_original_gpu_parity',passed=True,frames=3)
    return model

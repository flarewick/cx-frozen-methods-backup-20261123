"""Official compiled sampler, exact source-bound cache; no Python sample_noise replacement."""
import ast,inspect,os,platform,importlib.metadata
import torch
from .common import C,read,write,sha
O=C/'pi05_gt_shared6_v1'
from native_nextchunk_v1.sampler_fork import full_hash,environment_hash,torch_hash,base_hash

def install(base,root):
 from compiler_bundle import read_bundle
 binding=read(O/'original_compiled_sampler_binding_v1.json');directory=__import__('pathlib').Path(binding['bundle']);assert sha(directory/'bundle.json')==binding['bundle_metadata_sha256']
 meta,data=read_bundle(directory);assert meta['artifact_sha256']==binding['artifact_sha256'];f=meta['fingerprint'];prop=torch.cuda.get_device_properties(0)
 actual=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,triton=importlib.metadata.version('triton'),gpu_name=prop.name,capability=[prop.major,prop.minor],driver=__import__('re').search(r'Kernel Module\s+([0-9.]+)',__import__('pathlib').Path('/proc/driver/nvidia/version').read_text()).group(1),float32_matmul_precision=torch.get_float32_matmul_precision())
 assert all(f[k]==v for k,v in actual.items()),actual
 assert not torch._dynamo.config.disable and hasattr(base._sample_actions,'_torchdynamo_orig_callable')
 from torch._inductor import config
 config.compile_threads=16;config.max_autotune_gemm_backends='ATEN'
 from autotune_relocation import AutotuneRelocation
 relocation=AutotuneRelocation(os.environ['TORCHINDUCTOR_CACHE_DIR']);relocation.install();base._GT_autotune_relocation=relocation
 before=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();info=torch.compiler.load_cache_artifacts(data)
 assert info is not None and torch.equal(before,torch.get_rng_state())and torch.equal(gpu,torch.cuda.get_rng_state())
 warm=read(O/'compiled_sampler_warm_binding_v1.json');payload=__import__('pathlib').Path(warm['artifact_file']);assert sha(payload)==warm['artifact_sha256']and warm['original_sampler_source_and_weights_unchanged']and warm['actual_no_action_steady_exact']
 warm_info=torch.compiler.load_cache_artifacts(payload.read_bytes());assert warm_info is not None and torch.equal(before,torch.get_rng_state())and torch.equal(gpu,torch.cuda.get_rng_state())

 write(root/'official_compiled_sampler_restore.json',dict(passed=True,actual_GPU_uuid=os.environ['CUDA_VISIBLE_DEVICES'],compiler_origin_GPU_uuid=f['gpu_uuid'],original_metadata_not_relabelled=True,artifact_sha256=meta['artifact_sha256'],source_closure_CPU_verified=True,RNG_unchanged=True,actual_runtime=actual))
 return base._sample_actions

def audited_draw(router,lane,model,ledger,operation,*,query,tick):
 # The original compiled callable contains exactly one conditional sample_noise call.
 # Python noise interception would change its graph/random algorithm, so audit source and actual RNG boundary.
 import textwrap
 body=model.sample_actions
 while hasattr(body,'_torchdynamo_orig_callable'):body=body._torchdynamo_orig_callable
 tree=ast.parse(textwrap.dedent(inspect.getsource(body)));calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='sample_noise'];assert len(calls)==1
 assert 'sample_noise'not in model.__dict__ and not model.training
 ledger.permit_query(query,tick);before=router.capture(lane);result=operation();after=router.capture(lane)
 assert environment_hash(before)==environment_hash(after)and torch_hash(before)!=torch_hash(after)
 digest=base_hash(result['base']);ledger.record_draw(query,tick,digest,full_hash(before),full_hash(after),actual_sampler_call=True)
 return result,after,dict(query=query,tick=tick,base_sha256=digest,rng_before=full_hash(before),rng_after=full_hash(after),environment_unchanged=True,before=torch_hash(before),after=torch_hash(after),environment_before=environment_hash(before),environment_after=environment_hash(after),shape=[1,32,32],device_type='cuda',noise_witness_kind='frozen_compiled_source_single_noise_call_and_actual_infer_RNG_boundary',Python_noise_hook_installed=False,noise_values_not_individually_intercepted=True)

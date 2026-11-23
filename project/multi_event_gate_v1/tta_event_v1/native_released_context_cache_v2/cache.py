"""Same-device project context; recorded driver mounts for unseeded history."""
from native_released_context_cache_v1.cache import Context,read,digest,identity,safe_file
from pathlib import Path
import os,json,hashlib,shutil,tempfile

def export(result_file,out):
 from native_policy_execution_v1.common import write
 from native_onpolicy_recovery_v3.common import mapped,verify
 import user_container
 from finish_scratch import scratch_target
 result_file=Path(result_file);r=read(result_file);verify(r)
 assert r['release']['own_gpu_released']and not r['release']['remaining_own_gpu_pids']and r['new_scientific_cases_qualified']==0
 root=mapped(r['root']);actual=r['actual_host_exit'];assert actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
 spec=read(root/'protocol.json');verify(spec);gpu=spec['required_gpu_uuid'];plan=read(root/'device_plan.json');assert gpu==plan['info']['uuid']
 sessions=list((root/'sessions').glob('*/storage.json'));assert len(sessions)==1
 storage=read(sessions[0]);scratch=scratch_target(storage['scratch'],user_container.WORK/'run-tmp');cache=scratch/'xdg_cache_home'
 assert storage['environment']['XDG_CACHE_HOME']==storage['scratch']+'/xdg_cache_home'and cache.is_dir()and not cache.is_symlink()
 seed_file=root/'CPU_same_device_driver_cache_seed.json'
 if seed_file.exists():
  old_seed=read(seed_file);assert old_seed['actual_gpu_uuid']==gpu and old_seed['seeded_before_CUDA_permit']
  driver=old_seed['driver'];driver_proof=seed_file
 else:
  import re
  driver_proof=root/'attempt/launch.json';launch=read(driver_proof);assert launch['gpu']['uuid']==gpu
  versions=set(re.findall(r'libnvidia-(?:glcore|rtcore|gpucomp|glvkspirv)\.so\.(\d+\.\d+\.\d+)',str(launch['command'])))
  assert len(versions)==1,versions
  driver=versions.pop()

 payload={};snap={}
 for top in ['nvidia/GLCache','torch/kernels']:
  for p in sorted((cache/top).rglob('*')):
   if not p.is_file():continue
   safe_file(p,cache);stat=p.stat();data=p.read_bytes();name=str(p.relative_to(cache));payload[name]=data;snap[name]=(stat.st_size,stat.st_mtime_ns,digest(p));assert len(data)==stat.st_size and hashlib.sha256(data).hexdigest()==snap[name][2]
 assert payload and sum(map(len,payload.values()))<=32*2**20 and any(n.startswith('nvidia/GLCache/')for n in payload)
 hashes={n:hashlib.sha256(b).hexdigest()for n,b in payload.items()}
 ctx=dict(actual_gpu_uuid=gpu,driver=driver,files=hashes)
 proof=dict(passed=True,CPU_only_export=True,source_actual_result=str(result_file.resolve()),source_GPU_protocol=str((root/'protocol.json').resolve()),source_cache_before_after_stable=True,context_identity_sha256=identity(ctx),**ctx,original_workload_root=str(root),original_GPU_release_verified=True,resource_failure_preserved_NOT_qualified_science=True,no_parent_compiled_sampler_import_or_fingerprint_claim=True)
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 for n,data in payload.items():p=out/'files'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 for n,expected in snap.items():
  p=cache/n;s=p.stat();assert (s.st_size,s.st_mtime_ns,digest(p))==expected,'Context cache changed during snapshot'
 files=[result_file,root/'protocol.json',root/'attempt/resources.jsonl',driver_proof,sessions[0],*list((out/'files').rglob('*'))]
 proof['input_sha256']={str(p.resolve()):digest(p)for p in files if p.is_file()};write(out/'manifest.json',proof)
 return out/'manifest.json'

def available(deployment,gpu_uuid,driver):
 from native_policy_execution_v1.common import T
 from native_onpolicy_recovery_v3.common import mapped,verify
 p=T/'outputs/native_released_context_cache_v2/catalog_v1.json';v=read(p);verify(v)
 assert gpu_uuid in v['devices']
 row=v['devices'][gpu_uuid];manifest=mapped(row['manifest']);assert digest(manifest)==row['sha256']and row['driver']==driver
 return manifest

def restore(root,gpu_uuid):
 from native_policy_execution_v1.common import T,write
 from native_onpolicy_recovery_v3.common import verify
 import user_container,gpu_doctor
 root=Path(root);barrier=root/'attempt/barrier';assert not(barrier/'permit.json').exists()
 ready=read(barrier/'ready.json');spec=read(root/'protocol.json');assert ready['passed']and ready['os_threads']==1 and not ready['cuda_initialized']and ready['cpu_fork_child_passed']and ready['source_sha256']==spec['source_sha256']and ready['protocol_identity']==spec['identity']and spec['required_gpu_uuid']==gpu_uuid
 sessions=list((root/'sessions').glob('*/storage.json'));assert len(sessions)==1
 versions=gpu_doctor.query(['--query-gpu=driver_version']).splitlines();assert versions and len(set(x.strip()for x in versions))==1;driver=versions[0].strip()
 manifest=available(user_container.HERE,gpu_uuid,driver);context=Context(manifest,gpu_uuid,driver);seed=context.seed(sessions[0].parent)
 assert not(barrier/'permit.json').exists()
 info=dict(passed=True,actual_gpu_uuid=gpu_uuid,driver=driver,original_seed_result=seed,context_identity_sha256=context.record['context_identity_sha256'],seeded_before_CUDA_permit=True,actual_source_bound_CPU_preload_checked=True,CPU_ready_sha256=digest(barrier/'ready.json'),restores_context_files_only_NOT_parent_compiled_sampler=True,original_generic_ContextCacheSeed_copy_algorithm_reused=True,NN_compiler_fingerprint_explicitly_null=True,GPU_qualification_NOT_assumed=True,source_sha256={str(p):digest(p)for p in[Path(__file__),user_container.HERE/'context_cache_seed.py',user_container.HERE/'finish_scratch.py']},input_sha256={str(manifest):digest(manifest),**context.record['input_sha256']})
 write(root/'CPU_same_device_driver_cache_seed.json',info);return info

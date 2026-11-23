"""Export only genuine same-GPU NVRTC/GL caches after verified GPU release."""
import hashlib,json,os,shutil,tempfile,time
from pathlib import Path
import user_container as container
from compiler_bundle import read_bundle
from context_cache_seed import ContextCacheSeed
from finish_scratch import scratch_target
from evaluate import host
D=Path(__file__).resolve().parent;P=D/'reports'

def export_context(plan,bundle,telemetry,previous=None):
    rows=[json.loads(l)for l in Path(telemetry).read_text().splitlines()if l.strip()]
    if not rows or rows[-1].get('own_gpu_released')is not True or rows[-1].get('remaining_own_gpu_pids'):
        raise RuntimeError('Context export requires verified GPU release')
    meta,_=read_bundle(bundle)
    if meta['fingerprint']['gpu_uuid']!=plan['gpu_uuid']:raise RuntimeError('Context export GPU differs')
    master=None
    for job in plan['jobs']:
        for session in(host(job['run'])/'sessions').glob('process_*'):
            if session.name in job['previous_sessions']or not(session/'deployment_model_cache.json').is_file():continue
            record=json.loads((session/'deployment_model_cache.json').read_text())
            if record['gpu_uuid']!=plan['gpu_uuid']or record['sources']!=plan['source_sha256']:raise RuntimeError('Actual GPU cache provenance differs')
            imported=record.get('imported')or{}
            if imported.get('artifact_sha256')!=meta['artifact_sha256']or imported.get('fingerprint_sha256')!=meta['fingerprint_sha256']:raise RuntimeError('Actual sampler import differs')
            if master is not None:raise RuntimeError('Multiple cache owners')
            master=session
    if master is None:raise RuntimeError('No actual original sampler context')
    storage=json.loads((master/'storage.json').read_text());scratch=scratch_target(storage['scratch'],container.WORK/'run-tmp');cache=scratch/'xdg_cache_home'
    if storage['environment']['XDG_CACHE_HOME']!=storage['scratch']+'/xdg_cache_home'or cache.is_symlink():raise RuntimeError('Unexpected private cache location')
    files={}
    for top in ['torch/kernels','nvidia/GLCache']:
        for f in sorted((cache/top).rglob('*')):
            if f.is_symlink()or any(a.is_symlink()for a in f.parents if a.is_relative_to(cache)):raise RuntimeError('Symlink in actual context cache')
            if f.is_file():files[str(f.relative_to(cache))]=f.read_bytes()
    if not files or sum(map(len,files.values()))>32*2**20:raise RuntimeError('Context cache empty or too large')
    hashes={n:hashlib.sha256(v).hexdigest()for n,v in files.items()};digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();root=D/'.work/shared6_context_caches'/plan['gpu_uuid']/digest;root.parent.mkdir(parents=True,exist_ok=True)
    if not root.exists():
        stage=Path(tempfile.mkdtemp(prefix='export_',dir=root.parent))
        try:
            for name,data in files.items():f=stage/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
            os.rename(stage,root)
        finally:
            if stage.exists():shutil.rmtree(stage)
    before={n:hashlib.sha256(v).hexdigest()for n,v in(previous or{}).items()};added=[n for n,h in hashes.items()if before.get(n)!=h]
    proof=dict(passed=True,artifact_sha256=meta['artifact_sha256'],fingerprint_sha256=meta['fingerprint_sha256'],root_relative=str(root.relative_to(D)),files=hashes,source_gpu_uuid=plan['gpu_uuid'],actual_sampler_session=str(master),actual_sampler_manifest_sha256=hashlib.sha256((master/'deployment_model_cache.json').read_bytes()).hexdigest(),source_sha256=plan['source_sha256'],verified_gpu_release=True,new_or_changed_files=added,created_ns=time.time_ns())
    file=P/('target_context_cache_shared6_'+plan['gpu_uuid']+'_'+digest[:12]+'.json');file.write_text(json.dumps(proof,indent=2)+'\n');ContextCacheSeed(file,bundle)
    return dict(proof=str(file),bundle=str(bundle),files=len(files),additional_entries=len(added),bytes=sum(map(len,files.values())))

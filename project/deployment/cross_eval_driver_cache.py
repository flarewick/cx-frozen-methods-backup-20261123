"""Host-only SAME-physical-GPU driver caches with explicit parent compiler provenance."""
import hashlib,json,os,shutil,tempfile,time
from pathlib import Path
import user_container as container
from compiler_bundle import read_bundle
from context_cache_seed import ContextCacheSeed
from finish_scratch import scratch_target
from evaluate import host
D=Path(__file__).resolve().parent;P=D/'reports'
class SameDeviceSeed(ContextCacheSeed):
 def __init__(self,file,bundle,actual_uuid):
  proof=json.loads(Path(file).read_text())
  if proof.get('source_gpu_uuid')!=actual_uuid or not proof.get('verified_gpu_release')or not proof.get('actual_sampler_import_verified'):raise RuntimeError('Driver cache is not from this physical GPU')
  super().__init__(file,bundle);self.path=Path(file);self.actual_uuid=actual_uuid
 def seed(self,session):
  result=super().seed(session);result.update(actual_gpu_uuid=self.actual_uuid,proof=str(self.path),same_physical_gpu_only=True)
  (session/'cross_driver_cache_seed.json').write_text(json.dumps(result,indent=2)+'\n');return result

def select(actual_uuid,bundle):
 meta,_=read_bundle(bundle);choices=[]
 for file in P.glob('cross_driver_cache_*.json'):
  value=json.loads(file.read_text())
  if value.get('source_gpu_uuid')!=actual_uuid or value.get('artifact_sha256')!=meta['artifact_sha256']or value.get('fingerprint_sha256')!=meta['fingerprint_sha256']:continue
  choices.append((len(value.get('files',{})),value.get('created_ns',0),file))
 return SameDeviceSeed(max(choices)[2],bundle,actual_uuid)if choices else None

def export(plan,bundle,telemetry):
 rows=[json.loads(line)for line in Path(telemetry).read_text().splitlines()if line.strip()]
 if not rows or rows[-1].get('own_gpu_released')is not True or rows[-1].get('remaining_own_gpu_pids'):raise RuntimeError('Driver cache export requires confirmed release')
 meta,_=read_bundle(bundle);actual=plan['gpu_uuid'];master=None
 for job in plan['jobs']:
  for session in(host(job['run'])/'sessions').glob('process_*'):
   file=session/'deployment_model_cache.json'
   if session.name in job['previous_sessions']or not file.exists():continue
   record=json.loads(file.read_text());imported=record.get('imported')or{}
   if record['gpu_uuid']!=actual or record['sources']!=plan['source_sha256']or imported.get('artifact_sha256')!=meta['artifact_sha256']or imported.get('fingerprint_sha256')!=meta['fingerprint_sha256']or imported.get('actual_gpu_uuid')!=actual:raise RuntimeError('Actual sampler parent/target/source differs')
   if master is not None:raise RuntimeError('Multiple context owners')
   master=session
 if master is None:raise RuntimeError('Missing actual sampler owner')
 storage=json.loads((master/'storage.json').read_text());scratch=scratch_target(storage['scratch'],container.WORK/'run-tmp');cache=scratch/'xdg_cache_home'
 if storage['environment']['XDG_CACHE_HOME']!=storage['scratch']+'/xdg_cache_home'or cache.is_symlink():raise RuntimeError('Unexpected same-device context path')
 files={}
 for top in ['torch/kernels','nvidia/GLCache']:
  for f in sorted((cache/top).rglob('*')):
   if f.is_symlink()or any(p.is_symlink()for p in f.parents if p.is_relative_to(cache)):raise RuntimeError('Symlink in driver cache')
   if f.is_file():files[str(f.relative_to(cache))]=f.read_bytes()
 if not files or sum(map(len,files.values()))>32*2**20:raise RuntimeError('Empty/oversized driver cache')
 hashes={name:hashlib.sha256(data).hexdigest()for name,data in files.items()};digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();root=D/'.work/cross_driver_caches'/actual/digest;root.parent.mkdir(parents=True,exist_ok=True)
 if not root.exists():
  stage=Path(tempfile.mkdtemp(prefix='export_',dir=root.parent))
  try:
   for name,data in files.items():target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
   os.rename(stage,root)
  finally:
   if stage.exists():shutil.rmtree(stage)
 receipt=dict(passed=True,source_gpu_uuid=actual,parent_compiler_gpu_uuid=meta['fingerprint']['gpu_uuid'],artifact_sha256=meta['artifact_sha256'],fingerprint_sha256=meta['fingerprint_sha256'],root_relative=str(root.relative_to(D)),files=hashes,verified_gpu_release=True,actual_sampler_import_verified=True,actual_sampler_manifest=str(master/'deployment_model_cache.json'),actual_sampler_manifest_sha256=hashlib.sha256((master/'deployment_model_cache.json').read_bytes()).hexdigest(),source_sha256=plan['source_sha256'],created_ns=time.time_ns())
 file=P/('cross_driver_cache_'+actual+'_'+digest[:12]+'.json');file.write_text(json.dumps(receipt,indent=2)+'\n');SameDeviceSeed(file,bundle,actual)
 return dict(proof=str(file),files=len(files),actual_gpu_uuid=actual,parent_compiler_gpu_uuid=meta['fingerprint']['gpu_uuid'])

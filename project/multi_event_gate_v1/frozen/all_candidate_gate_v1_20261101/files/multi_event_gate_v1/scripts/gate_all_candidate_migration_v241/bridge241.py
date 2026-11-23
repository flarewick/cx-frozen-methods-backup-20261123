"""Operational whole-G import to a truthful original-GPU3 scientific identity.

The numerical entry is the archived entry218 used on originalGPU3. OriginGPU0
is never relabeled: immutable per-episode receipts and cache ownership remain.
"""
import copy,json,os,hashlib,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
HOST='/data/qiufw/cx_migration/cx_new_direction_guide_TTA';CANON='/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA'
VERIFIED=set();INSTALLED=False
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def locate(f):
 f=Path(f)
 if f.exists():return f
 if str(f).startswith(CANON+'/'):return Path(HOST+str(f)[len(CANON):])
 if str(f).startswith(HOST+'/'):return Path(CANON+str(f)[len(HOST):])
 return f
def atomic(f,v):
 f=Path(f);t=f.with_suffix('.tmp');t.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');t.replace(f)
def comparable(m):
 m=copy.deepcopy(m);m.pop('evaluation_identity',None);extra=m.pop('gpu_migration241',None) or m.pop('gpu_migration231',None)
 if extra:
  for k in extra['registered_sources']:m['source_sha256'].pop(k,None)
 m['deployment_shared6']['gpu_uuid']='<physical-GPU>'
 return m

def verify_origin(job):
 run=locate(job['run'])
 for name,h in job['files'].items():assert sha(run/name)==h,('Origin changed',run,name)
 m=read(run/'manifest.json');q=read(run/'concurrency_queue.json');assert m==job['manifest'] and q['identity']==m['evaluation_identity']
 for e in q['completed'].values():
  assert e['complete']is True and type(e['success'])is bool
  for f,h in e['evidence_sha256'].items():assert sha(locate(f))==h
 return run,m,q

def copy_exact(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert sha(src)==sha(dst);return
 try:os.link(src,dst)
 except OSError:shutil.copyfile(src,dst)
 assert sha(src)==sha(dst)

def import_G(run,manifest):
 import benefit_followup_protocol_v2 as p
 if manifest.get('pilot',{}).get('branch')!='G' or manifest['deployment_shared6']['gpu_uuid']!='GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a':return
 regpath=p.ROOT/'migration241/origin_G_registry.json';reg=read(regpath);key=manifest['task']
 if key not in reg['jobs']:return
 job=reg['jobs'][key]
 cache=(str(run),manifest['evaluation_identity'],sha(regpath))
 if cache in VERIFIED:return
 run=Path(run);oldrun,old,q=verify_origin(job);assert run!=oldrun and comparable(old)==comparable(manifest)
 receipt=run/'resume_import241_receipt.json'
 if receipt.exists():
  v=read(receipt);assert v['passed'] and v['registry_sha256']==sha(regpath) and v['new_identity']==manifest['evaluation_identity']
  for name,h in v['copied_files'].items():assert sha(run/name)==h
  assert(run/'journal.json').exists()and(run/'concurrency_queue.json').exists();VERIFIED.add(cache);return
 assert not(run/'journal.json').exists()and not(run/'concurrency_queue.json').exists(),'No import over a live target'
 imported={};proof=[];files={}
 for seed,e in q['completed'].items():
  source=oldrun/'sessions'/e['process_session'];newname='import241_'+e['process_session'];dest=run/'sessions'/newname
  # Copy only SHA-certified private Python sources, immutable session metadata
  # and this complete episode's evidence. Runtime asset links stay in origin.
  for f in source.iterdir():
   if f.is_symlink()or not f.is_file():continue
   dst=dest/f.relative_to(source);copy_exact(f,dst);files[str(dst.relative_to(run))]=sha(dst)
  provenance=read(source/'provenance.json')
  for name,h in provenance['runtime_sha256'].items():
   relative=Path(name);assert not relative.is_absolute()and '..'not in relative.parts
   f=source/'robotwin'/relative;assert not f.is_symlink()and sha(f)==h
   dst=dest/'robotwin'/relative;copy_exact(f,dst);files[str(dst.relative_to(run))]=h
  for name,h in e['evidence_sha256'].items():
   f=locate(name);relative=f.relative_to(source);assert not f.is_symlink()and sha(f)==h
   dst=dest/relative;copy_exact(f,dst);files[str(dst.relative_to(run))]=h
  new=copy.deepcopy(e);new['process_session']=newname;new['evidence_sha256']={}
  for f,h in e['evidence_sha256'].items():
   dst=dest/locate(f).relative_to(source);assert sha(dst)==h;new['evidence_sha256'][str(dst)]=h
  imported[seed]=new;proof.append(dict(seed=e['seed'],origin_session=e['process_session'],new_session=newname,actual_origin_GPU_UUID=old['deployment_shared6']['gpu_uuid'],only_session_and_evidence_paths_rebased=True))
 prefix=[];eps=[]
 for r in q['attempts']:
  prefix.append(copy.deepcopy(r))
  if r['eligible']:
   if str(r['seed'])not in imported:break
   eps.append(copy.deepcopy(imported[str(r['seed'])]))
 value=dict(passed=True,original_controls88_not_replaced=True,origin_run=job['run'],origin_identity=old['evaluation_identity'],new_identity=manifest['evaluation_identity'],actual_origin_GPU_UUID=old['deployment_shared6']['gpu_uuid'],actual_target_GPU_UUID=manifest['deployment_shared6']['gpu_uuid'],registry_sha256=sha(regpath),imported_whole_episodes=len(imported),episodes=proof,copied_files=files,source_sha256={str(f):sha(f)for f in HERE.glob('*.py')},scientific_manifest_equal_except_physical_GPU_and_explicit_origin231_provenance=True,original_scientific_audit_and_cross_GPU_exact_replay_still_required=True)
 atomic(receipt,value);atomic(run/'concurrency_queue.json',dict(identity=manifest['evaluation_identity'],attempts=q['attempts'],completed=imported));atomic(run/'journal.json',dict(identity=manifest['evaluation_identity'],attempts=prefix,episodes=eps));atomic(run/'episodes.json',eps);VERIFIED.add(cache)

def install():
 global INSTALLED
 if INSTALLED:return
 INSTALLED=True
 import benefit_growth_runtime as runtime
 prior=runtime.install_sources
 def sources():
  prior()
  import shared6_eval_backend as b,benefit_followup_protocol_v2 as p
  if not getattr(b,'_migration241_sources',False):
   previous=b.decorate;registry=p.ROOT/'migration241/origin_G_registry.json';target=p.ROOT/'migration241/target.json'
   def decorate(*a,**kw):
    raw=previous(*a,**kw)
    registered={str(f):sha(f)for f in [*HERE.glob('*.py'),registry,target]}
    raw['source_sha256'].update(registered)
    raw['gpu_migration241']=dict(registry_sha256=sha(registry),registered_sources=registered,actual_target_GPU_UUID='GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a',origin_results_keep_actual_GPU0=True,all_candidate_scientific_scope_unchanged=True)
    from cx_tta.evaluation_resume import identity
    raw['evaluation_identity']=identity(raw);return raw
   b.decorate=decorate;b._migration241_sources=True
  if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':return
  import cx_tta.evaluation_resume as resume
  if getattr(resume,'_import241',False):return
  original=resume.load_journal
  def load(run,manifest):import_G(Path(run),manifest);return original(run,manifest)
  resume.load_journal=load;resume._import241=True
 runtime.install_sources=sources

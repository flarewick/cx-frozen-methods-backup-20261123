"""Register truthful originalGPU3 caches and immutable whole-G0 source queues."""
import sys,os,json,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;W=HERE.parents[1];R=W.parent;D=R/'deployment';H=W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4';sys.path.insert(0,str(D))
from compiler_bundle import read_bundle
from shared6_native_cache_seed import Shared6NativeCacheSeed
from cross_eval_driver_cache import select

def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def locate(f):return Path(str(f).replace('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA',str(R)))
def write(f,v):
 if f.exists():assert read(f)==v
 else:f.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
for cohort in (0,1):
 q=H/('candidate_veto_all_events_v237_'+str(cohort))/'dev0';m=q/'migration241';m.mkdir(exist_ok=True)
 old=read(H/('candidate_veto_cup_bowls_v200_'+str(cohort))/'dev0/cache_preference_v96.json');bundle=R/old['bundle_relative'];meta,_=read_bundle(bundle)
 assert meta['fingerprint']['gpu_uuid']=='GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a'and meta['artifact_sha256']=='2fc67e7d2e5c2d38879001f085dd107080df49c35b649532087c0672f615af2b'
 native=Shared6NativeCacheSeed(R/old['native_proof_relative'],bundle);assert native.proof['source_count']==48 and len(native.files)==963 and not native.proof['unselected'];driver=select(old['gpu_uuid'],bundle);assert driver
 cfg=dict(index=3,uuid=old['gpu_uuid'],bundle_relative=old['bundle_relative'],bundle_metadata_sha256=sha(bundle/'bundle.json'),native_proof_relative=old['native_proof_relative'],native_proof_sha256=old['native_proof_sha256'],compiler_origin_and_actual_same_GPU3=True,original_same_card_driver_seed_read_only=True)
 write(m/'target.json',cfg)
 jobs={};total=0;plan=q/'validate_plan.json'
 if plan.exists():
  v=read(plan);assert v['gpu_uuid']=='GPU-fa87bc41-1919-efe1-73f9-11f38008763e'
  for j in v['jobs']:
   r=locate(j['run']);man=read(r/'manifest.json');queue=read(r/'concurrency_queue.json');assert man==j['manifest'] and queue['identity']==man['evaluation_identity'];total+=len(queue['completed'])
   for e in queue['completed'].values():
    for f,h in e['evidence_sha256'].items():assert sha(locate(f))==h
   jobs[j['task']]=dict(run=str(r),manifest=man,files={n:sha(r/n)for n in ['manifest.json','concurrency_queue.json','journal.json','episodes.json']})
  write(m/'origin_validate_plan.json',v)
 reg=dict(passed=True,origin_GPU_UUID='GPU-fa87bc41-1919-efe1-73f9-11f38008763e',target_GPU_UUID=cfg['uuid'],origin_whole_G=total,jobs=jobs,all_candidate_scope_and_original118_frozen=True,explicit_import_only_no_relabel=True,full_original_science_audit_required=True,prospective_source_sha256={str(f):sha(f)for f in HERE.glob('*.py')})
 write(m/'origin_G_registry.json',reg);print(cohort,total)

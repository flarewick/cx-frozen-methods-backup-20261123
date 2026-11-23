"""Pin actual certified GPU3 sampler/native/driver bytes to independent protocol."""
import json
from pathlib import Path
import benefit_followup_protocol_v2 as p
def bind():
 from shared6_native_cache_seed import Shared6NativeCacheSeed
 from compiler_bundle import read_bundle
 from cross_eval_driver_cache import select
 f=p.ROOT.parent/'cache_selection_CPU_v156.json';s=json.loads(f.read_text());bundle=p.R/s['bundle_relative'];meta,_=read_bundle(bundle);native=Shared6NativeCacheSeed(p.R/s['native_proof_relative'],bundle)
 assert s['gpu']==3 and meta['fingerprint']['gpu_uuid']==s['gpu_uuid'] and meta['artifact_sha256']==s['artifact_sha256']
 assert p.sha(bundle/'bundle.json')==s['bundle_metadata_sha256'] and p.sha(p.R/s['native_proof_relative'])==s['native_proof_sha256']
 assert len(native.files)==963 and len(native.proof['kernels'])==48 and not native.proof['unselected']
 assert p.sha(p.R/s['driver_proof_relative'])==s['driver_proof_sha256'] and select(s['gpu_uuid'],bundle) is not None
 proof=dict(passed=True,CPU_only=True,new_protocol_sha256=p.sha(p.ROOT/'protocol.json'),selection_sha256=p.sha(f),original_actual_artifact_sha256=meta['artifact_sha256'],same_physical_GPU3=True,original_native_files_verified=len(native.files),all48_official_requests_exact=True,no_old_episodes_copied=True,all_new_data_requires_strict_AA_and_full88_audit=True,binder_sha256=p.sha(__file__))
 target=p.ROOT/'cache_provenance_CPU_v96.json'
 if target.exists():assert json.loads(target.read_text())==proof
 else:p.atomic(target,proof)
 cfg=dict(bundle_relative=s['bundle_relative'],parent_bundle_relative=s['bundle_relative'],native_proof_relative=s['native_proof_relative'],native_proof_sha256=s['native_proof_sha256'],new_metadata_sha256=s['bundle_metadata_sha256'],gpu_uuid=s['gpu_uuid'],original_host_sha256=p.sha(p.D/'benefit_growth_host_v3.py'),protocol_sha256=proof['new_protocol_sha256'],provenance_sha256=p.sha(target))
 target=p.ROOT/'cache_preference_v96.json'
 if target.exists():assert json.loads(target.read_text())==cfg
 else:p.atomic(target,cfg)
 return cfg

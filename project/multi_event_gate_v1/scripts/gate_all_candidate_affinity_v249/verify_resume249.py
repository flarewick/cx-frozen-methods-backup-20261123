"""Original import verifier with explicit validation of already completed native scenes."""
import ast,copy,json,sys
from pathlib import Path
import torch
assert not torch.cuda.is_initialized()and not list(Path('/dev').glob('nvidia*'))
import benefit_followup_protocol_v2 as p,benefit_growth_runtime as runtime
runtime.install_sources()
from bridge241 import read,sha
native=[]
def verify_native(job,queue):
 run=Path(job['run']);assert read(run/'manifest.json')==job['manifest']
 # EpisodeQueue has already run the original whole-evidence, identity and journal validation.
 assert queue.state['identity']==job['identity']==job['manifest']['evaluation_identity']
 for e in queue.state['completed'].values():
  assert not e['process_session'].startswith('import241_')
  session=run/'sessions'/e['process_session'];storage=read(session/'storage.json');master=Path(storage.get('shared_process_scratch_owner',str(session)))
  m=read(master/'deployment_model_cache.json');n=read(master/'native_cache_seed.json');d=read(master/'cross_driver_cache_seed.json')
  assert m['gpu_uuid']==d['actual_gpu_uuid']=='GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a'and d['same_physical_gpu_only']
  assert m['imported']['artifact_sha256']==n['artifact_sha256']=='2fc67e7d2e5c2d38879001f085dd107080df49c35b649532087c0672f615af2b'
  assert m['imported']['source_closure_verified']and m['imported']['rng_unchanged']and n['official_requests_exact']and n['unique_selected_sources']==48
  for f,h in e['evidence_sha256'].items():assert sha(f)==h
  native.append(dict(task=job['task'],seed=e['seed'],session=e['process_session'],cache_owner=str(master),original_identity=job['identity']))
file=Path(__file__).resolve().parent.parent/'gate_all_candidate_migration_v241/verify241.py';tree=ast.parse(file.read_text());hits=[]
for n in ast.walk(tree):
 if isinstance(n,ast.Assert)and ast.unparse(n.test)=="not queue.state['completed']":
  hits.append((n,copy.deepcopy(n)));replacement=ast.Expr(value=ast.Call(func=ast.Name(id='_verify_native249',ctx=ast.Load()),args=[ast.Name(id='job',ctx=ast.Load()),ast.Name(id='queue',ctx=ast.Load())],keywords=[]));n.__class__=ast.Expr;n.__dict__.clear();n.__dict__.update(replacement.__dict__)
assert len(hits)==1
ns=dict(__name__='resumed_verifier249',__file__=str(file),_verify_native249=verify_native)
exec(compile(ast.fix_missing_locations(tree),str(file),'exec'),ns)
p.atomic(p.ROOT/'migration241/CPU_resumed249.json',dict(passed=True,no_GPU=True,native_whole=len(native),native=native,original241_import_checks_retained=True,original_episode_queue_evidence_identity_and_journal_checks_passed=True,old_empty_only_precondition_replaced_by_full_native_result_validation=True,helper_sha256=p.sha(__file__),original_verifier_sha256=p.sha(file)))
print('249_CPU_RESUME_NATIVE_VERIFIED',len(native),flush=True)

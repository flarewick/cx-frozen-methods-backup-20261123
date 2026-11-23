"""Release the CUDA worker after verified scenes; run the unchanged final loop on CPU."""
import argparse,ast,copy,hashlib,json,os,time
from pathlib import Path
from benefit_growth_protocol import ROOT,D,sha,atomic
TRANSFORMS=[]

def final_loop(node):
 loops=[n for n in ast.walk(node)if isinstance(n,ast.For)and any(isinstance(c,ast.Call)and ast.unparse(c.func)=='original.final_result'for c in ast.walk(n))]
 if len(loops)!=1:raise RuntimeError('Original final-result loop changed')
 return loops[0]

def rewrite_gpu_tail(node):
 loop=final_loop(node)
 lists=[getattr(n,f)for n in ast.walk(node)for f in ('body','orelse','finalbody')if isinstance(getattr(n,f,None),list)and loop in getattr(n,f)]
 if len(lists)!=1:raise RuntimeError('Ambiguous GPU finalization boundary')
 body=lists[0];i=body.index(loop)
 if i==0 or ast.unparse(body[i-1])!='c.verify_weights()':raise RuntimeError('Frozen-weight verification must precede CPU intent')
 before=copy.deepcopy(node);removed=copy.deepcopy(loop)
 body[i]=ast.parse("__import__('benefit_growth_deferred_finalize').ready_for_cpu(plan)").body[0]
 strings=[n for n in ast.walk(node)if isinstance(n,ast.Constant)and n.value=='SHARED6_ORIGINAL_TWO_CONDITION_ALL_COMPLETE']
 if len(strings)!=1:raise RuntimeError('Original completion marker changed')
 strings[0].value='SHARED6_GPU_PHASE_COMPLETE_CPU_FINALIZATION_PENDING'
 # Independently restore the two scheduling edits to prove every other AST node unchanged.
 restored=copy.deepcopy(node)
 def undo(parent):
  for field,value in ast.iter_fields(parent):
   if isinstance(value,list):
    for j,item in enumerate(value):
     if isinstance(item,ast.Expr)and isinstance(item.value,ast.Call)and ast.unparse(item.value.func)=="__import__('benefit_growth_deferred_finalize').ready_for_cpu":value[j]=copy.deepcopy(removed)
     elif isinstance(item,ast.AST):undo(item)
   elif isinstance(value,ast.AST):undo(value)
  if isinstance(parent,ast.Constant)and parent.value=='SHARED6_GPU_PHASE_COMPLETE_CPU_FINALIZATION_PENDING':parent.value='SHARED6_ORIGINAL_TWO_CONDITION_ALL_COMPLETE'
 undo(restored)
 if ast.dump(restored,include_attributes=False)!=ast.dump(before,include_attributes=False):raise RuntimeError('GPU body beyond finalization was modified')
 receipt=dict(GPU_body_before_finalization_AST_unchanged=True,original_final_loop_AST_sha256=hashlib.sha256(ast.dump(removed,include_attributes=False).encode()).hexdigest(),worker_source_sha256=sha(D/'shared6_eval_worker.py'))
 TRANSFORMS.append(receipt);return receipt

def install():
 import benefit_stage_runtime as stage
 if getattr(stage,'_growth_deferred_finalize',False):return
 before=stage.transform
 def transform(function,mutate):
  def edit(node):
   mutate(node)
   if node.name=='run_gpu':rewrite_gpu_tail(node)
  return before(function,edit)
 stage.transform=transform;stage._growth_deferred_finalize=True

def ready_for_cpu(plan):
 from shared6_eval_backend import sources
 assert plan['source_sha256']==sources()
 audits={}
 for job in plan['jobs']:
  for session in(Path(job['run'])/'sessions').glob('process_*'):
   if session.name in job['previous_sessions']:continue
   f=session/'source_runtime_audit.json'
   if not f.exists():raise RuntimeError('Original per-scene source audit missing before GPU release')
   audits[str(f)]=sha(f)
 assert audits and len(TRANSFORMS)==1
 barrier=Path(os.environ['CX_BENEFIT_RECYCLE_FILE']).parent
 atomic(barrier/'cpu_finalize_intent.json',dict(schema='verified_GPU_phase_then_original_CPU_finalization_v1',plan=plan,gpu_uuid=os.environ['CX_TARGET_GPU_UUID'],source_sha256=sources(),helper_sha256=sha(__file__),source_runtime_audits_sha256=audits,worker_AST_receipt=TRANSFORMS[0],frozen_weights_verified_by_original_worker=True,all_original_scene_threads_finished=True,GPU_namespace_pid=os.getpid(),created_unix=time.time()))

def execute_original_final_loop(plan,writer=None):
 # This is the exact original AST, including all original final_result calls,
 # result/progress writes, prints and failure propagation. No placeholder result.
 source=ast.parse((D/'shared6_eval_worker.py').read_text());node=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name=='run_gpu');loop=copy.deepcopy(final_loop(node))
 from task_adapters import eval_resume as original
 from cx_tta.paths import replace_json
 ns=dict(original=original,Path=Path,replace_json=replace_json if writer is None else writer,plan=plan,json=json)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),__file__,'exec'),ns)
 return hashlib.sha256(ast.dump(loop,include_attributes=False).encode()).hexdigest()

def finalize(intent_path):
 import torch
 if list(Path('/dev').glob('nvidia*'))or torch.cuda.is_initialized():raise RuntimeError('Finalization requires a CPU-only container')
 # Preserve archived mechanical proof binding before the explicit new-cohort bind.
 import benefit_success_shortcircuit
 import benefit_growth_runtime as growth
 growth.bind();growth.install()
 import benefit_memory_batches
 benefit_memory_batches.install()
 from shared6_eval_backend import sources
 intent=json.loads(Path(intent_path).read_text());assert intent['helper_sha256']==sha(__file__)
 release_file=Path(intent_path).parent/'host_gpu_release_receipt.json';release=json.loads(release_file.read_text())
 assert release['GPU_job_released']and release['release']['own_gpu_released']and not release['release'].get('remaining_own_gpu_pids')
 assert release['intent_sha256']==sha(intent_path)and release['telemetry_sha256']==sha(release['telemetry'])
 assert intent['source_sha256']==sources()==intent['plan']['source_sha256']
 assert intent['frozen_weights_verified_by_original_worker']and intent['all_original_scene_threads_finished']
 for f,h in intent['source_runtime_audits_sha256'].items():assert sha(f)==h
 expected=Path(intent_path).parent/'memory_recycle.json';assert os.environ['CX_BENEFIT_RECYCLE_FILE']==str(expected)
 loop_hash=execute_original_final_loop(intent['plan'])
 assert loop_hash==intent['worker_AST_receipt']['original_final_loop_AST_sha256']
 assert not torch.cuda.is_initialized()and sources()==intent['source_sha256']
 for f,h in intent['source_runtime_audits_sha256'].items():assert sha(f)==h
 receipt=dict(passed=True,CPU_only=True,cuda_initialized=False,intent_sha256=sha(intent_path),helper_sha256=sha(__file__),original_final_loop_AST_sha256=loop_hash,result_sha256={j['run']:sha(Path(j['run'])/'result.json')for j in intent['plan']['jobs']},source_sha256=sources(),completed_unix=time.time())
 atomic(Path(intent_path).parent/'cpu_finalize_receipt.json',receipt)
 print('ORIGINAL_CPU_FINALIZATION_PASSED',json.dumps(dict(jobs=len(intent['plan']['jobs']),cuda_initialized=False)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--intent',type=Path,required=True);args=p.parse_args();finalize(args.intent)

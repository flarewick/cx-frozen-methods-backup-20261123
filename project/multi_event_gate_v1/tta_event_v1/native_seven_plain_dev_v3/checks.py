"""Same numeric engine, real10-lane gate and original render/owner/RNG controls."""
import ast,threading,time,random,types
import numpy as np,torch
from .common import C,T,P,read,sha,verify
from native_policy_execution_v1.common import write

def restore_runtime(s):
 import json
 for before,after in reversed(read(C/'reuse_edits.json')):
  assert s.count(after)==1,(after,s.count(after));s=s.replace(after,before)
 return s

def lane_gate():
 from .startup import StagedLanes
 cancel=threading.Event();rows=[];gate=StagedLanes(cancel,rows.append,lanes=8);started=set();lock=threading.Lock()
 def worker(i):
  if gate.wait(i):
   with lock:started.add(i)
 threads=[threading.Thread(target=worker,args=(i,))for i in range(8)]
 for t in threads:t.start()
 end=time.monotonic()+2
 while len(started)<4:assert time.monotonic()<end;time.sleep(.005)
 assert started==set(range(4))
 assert gate.release('actual_once_policy_query_complete',lane=0,query=0,tick=0)
 for t in threads:t.join(2);assert not t.is_alive()
 assert started==set(range(8))and rows[0]['released_lanes']==list(range(4,8))
 return True

def overlap_checks():
 from native_render_overlap_v4 import checks as old
 before=old.O;old.O=T/'outputs'/C.name/'CPU_overlap_fixture_v2';old.O.mkdir(parents=True,exist_ok=False)
 def seeded(*a,**kw):torch.manual_seed(17);np.random.seed(17);random.seed(17);return old.trial(*a,**kw)
 try:
  for flags in({},dict(controls=True),dict(controls=True,unsafe_first=True)):
   a=seeded(False,**flags);b=seeded(True,**flags)
   assert a[-1]is None and b[-1]is None and a[2]==b[2]
   assert np.array_equal(a[0][0],b[0][0])and np.array_equal(a[0][2],b[0][2])
   assert all(x[0]==y[0]and np.array_equal(x[1],y[1])for x,y in zip(a[1],b[1],strict=True))
   assert b[5]==(6 if flags.get('controls')else 2)
  assert seeded(True,error=True)[-1]=='ValueError'
  assert seeded(True,cancel=True)[-1]in(None,'AssertionError')
  assert len(old.queue_checks())==2
 finally:old.O=before
 # Actual production runtime code objects, not name-only whitelisting.
 from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
 from native_expert_request_priority_v1.fairness import install
 from native_render_overlap_v4.scheduler import NativeOverlap
 from native_render_overlap_v4.codes import original_controls
 from .runtime import execute
 from .owner_precision import install as precision
 c=Coordinator();c.rng.install()
 class Base:
  def infer(self):pass
 base=Base();check=precision(base,torch)
 try:
  install(c,T/'outputs'/C.name/'CPU_production_code_fairness_v2.jsonl')
  bridge=NativeOverlap(c,torch,object(),lambda:False,control_codes=original_controls(execute,check.__code__))
  sample=next(code for code in bridge.safe_codes if code.co_name=='sample')
  def cell(v):return(lambda:v).__closure__[0]
  fn=types.FunctionType(sample,{},closure=tuple(cell(None)for _ in sample.co_freevars))
  from concurrent_rng import Lane
  from concurrent.futures import Future
  item=(0,0,Lane('actual_sampler'),fn,(),{},Future());bridge.pumping=True
  assert not bridge.allowed(item)
  bridge.graph_ready=lambda:True;assert bridge.allowed(item)
  def impostor():pass
  impostor.__name__='sample';assert not bridge.allowed((0,1,Lane('actual_sampler'),impostor,(),{},Future()))
  bridge.pumping=False;bridge.close();assert bridge.closed
 finally:c.rng.uninstall()
 return True

def main():
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 old=T/'native_seven_plain_dev_v2'
 changed={'common.py','collection_plan.py','checks.py','runtime.py','work.py','semantic_worker.py','branch_checks.py','branch_queue.py','prepared_recovery.py','closed_audit.py','AGENTS.md'}
 exact=[]
 for p in C.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 for p in old.rglob('*'):
  if p.is_file()and p.suffix in('.py','.sh')and p.name not in changed:
   q=C/p.relative_to(old);assert q.read_text()==p.read_text().replace(old.name,C.name),str(q);exact.append(str(p.relative_to(old)))
 runtime=(C/'runtime.py').read_text()
 assert runtime==(old/'runtime.py').read_text().replace(old.name,C.name)
 assert lane_gate()and overlap_checks()
 from .component_checks import pixels,prefix,source_exact,typechecks
 assert source_exact()and pixels()and prefix()and typechecks()
 from .branch_checks import check as check_branches
 assert check_branches()
 from native_calibrated_validation_v24.collection_plan import calibrated_qualification
 from native_visible_response_bridge_v1.bridge import load_model
 qualification,q=calibrated_qualification();model,trained=load_model(qualification)
 assert trained['actual_updates']==1000 and trained['minimum_distinct_FIT_cases']==7
 from cx_tta_feedback_confirm5.query import build_head
 from adapter_cache_cpu import state_digest
 rng=torch.random.get_rng_state().clone()
 spec=read(T/'outputs/native_eval_throughput_v4/branch12/study.json')
 with torch.random.fork_rng(devices=[]):heads=[build_head(spec['training'])[0]for _ in range(8)]
 assert torch.equal(rng,torch.random.get_rng_state())and len({state_digest(h)for h in heads})==1
 from .prepared_recovery import cache
 recovered=cache();assert recovered['cached_valid_cases']==6 and not recovered['primary_policy_results_imported']and not recovered['old_masks_or_teacher_outputs_imported']
 from .closed_audit import audit as actual_closed
 assert actual_closed._source_before!=actual_closed._source_after
 repair=read(T/'outputs/resource_repair_audits_v1/CPU_audit_repair_v1.json');assert repair['passed']
 from native_governor180_v2 import checks as policy
 assert len(policy.checks)>=35 and not torch.cuda.is_initialized()
 from native_seven_task_gate_expansion_v1.contract import O as EXP
 cpu=read(EXP/'CPU_compatibility_v1.json');actual=read(EXP/'CPU_actual_host_exit_v1.json')
 assert cpu['passed']and actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
 assert cpu['plan']['plain_tasks']==list(__import__(C.name+'.common',fromlist=['TASKS']).TASKS)
 source={str(p):sha(p)for p in C.rglob('*')if p.is_file()and p.suffix in('.py','.sh','.md','.json')}
 for folder in('native_owner_profile_v2','native_render_overlap_v4','native_governor180_v2','native_guide_reuse_v1','native_image_readback_gil_v1','native_batched_prefix_graph_v1','native_direct_frames_v1','native_pot_mic_semantics_v1','declared_reference_audit_v1','gated_handoff_compare_v1','resource_repair_audits_v1'):
  source.update({str(p):sha(p)for p in(T/folder).glob('*')if p.is_file()})
 write(T/'outputs'/C.name/'CPU_contract_checks_v1.json',dict(passed=True,no_GPU_or_model_update=True,exact_scientific_sources=exact,only_task_scope_language_route_changed_runtime_inverse_exact=True,actual8_identical_heads_and_staged_gate=True,actual27_whole_branches_serial_parallel_arrays_RNG_exact=True,original_pixel_prefix_typechecking_components_and_negative_controls=True,actual_runtime_code_whitelist_and_first_graph_render_exclusion=True,policy_checks=policy.checks,checkpoint=q['training_checkpoint_sha256'],source_sha256=source,input_sha256={str(p):sha(p)for p in(qualification,EXP/'CPU_compatibility_v1.json',EXP/'CPU_actual_host_exit_v1.json',EXP/'language_coverage_v1.json')}))
 print('PREPARED_EIGHT_LANE_THREE_TASK_TRAIN_CPU_PASS',flush=True)
if __name__=='__main__':main()

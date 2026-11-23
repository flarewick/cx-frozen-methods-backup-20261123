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
 cancel=threading.Event();rows=[];gate=StagedLanes(cancel,rows.append,lanes=12);started=set();lock=threading.Lock()
 def worker(i):
  if gate.wait(i):
   with lock:started.add(i)
 threads=[threading.Thread(target=worker,args=(i,))for i in range(12)]
 for t in threads:t.start()
 end=time.monotonic()+2
 while len(started)<4:assert time.monotonic()<end;time.sleep(.005)
 assert started==set(range(4))
 assert gate.release('actual_once_policy_query_complete',lane=0,query=0,tick=0)
 for t in threads:t.join(2);assert not t.is_alive()
 assert started==set(range(12))and rows[0]['released_lanes']==list(range(4,12))
 return True

def overlap_checks():
 from native_render_overlap_v4 import checks as old
 before=old.O;old.O=T/'outputs'/C.name/'CPU_overlap_fixture_v4';old.O.mkdir(parents=True,exist_ok=False)
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
  install(c,T/'outputs'/C.name/'CPU_production_code_fairness_v4.jsonl')
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
 import json
 from .configuration import GROUPS,bounds
 from .adoption import check as check_adoption
 adoption_inputs=check_adoption()
 from .qualification import qualified_stage
 from .common import mapped
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 for p in C.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 runtime=(C/'runtime.py').read_text()
 for before,after in reversed(read(C/'early_runtime_edits.json')):
  assert runtime.count(after)==1;runtime=runtime.replace(after,before)
 for before,after in reversed(read(C/'formal_runtime_edits.json')):
  assert runtime.count(after)==1,(after,runtime.count(after));runtime=runtime.replace(after,before)
 assert runtime==(T/'native_seven_plain_dev_v3/runtime.py').read_text().replace('native_seven_plain_dev_v3',C.name)
 from .budget import check as budget_check
 assert budget_check()
 queue_proof=T/'outputs/native_seven_early_queue_scope_v1/CPU_queue_checks_v1.json';queue_value=read(queue_proof);verify(queue_value)
 assert queue_value['passed']and queue_value['all_outputs_and_RNG_bytes_exact']and queue_value['official_six_scenes_per_task']
 queue_actual=T/'outputs/native_seven_early_queue_scope_v1/CPU_actual_exit_v2.json';qa=read(queue_actual)
 assert qa['exit_code']==0 and qa['actual_Popen_wait_return']and qa['proc_absent_after_wait']
 assert lane_gate()and overlap_checks()
 from .component_checks import pixels,prefix,source_exact,typechecks
 assert source_exact()and pixels()and prefix()and typechecks()
 from .branch_checks import check
 queue=check();assert len(queue)==20
 from .episode import execute,original_execute,_plain,_gated
 from gated_arm_context_v1.episode import execute as gated
 from native_arm_phase_feedback_v1.episode import execute as plain
 assert execute._q0_original is original_execute and _plain._q0_original is plain and _gated._q0_original is gated
 from .method import build_head,AuditedActor,PlainActor,PairedActor
 from native_seven_task_gate_expansion_v1.contract import plan
 from adapter_cache_cpu import state_digest
 from native_calibrated_validation_v24.collection_plan import calibrated_qualification
 from native_visible_response_bridge_v1.bridge import load_model
 qualification,q=calibrated_qualification();model,trained=load_model(qualification);assert trained['actual_updates']==1000 and trained['minimum_distinct_FIT_cases']==7
 p=plan();rng=torch.random.get_rng_state().clone();headproof=[]
 for group,cfg in GROUPS.items():
  training=p['gate_training']if group=='gated'else p['plain_training']
  with torch.random.fork_rng(devices=[]):heads=[build_head(training)[0]for _ in range(cfg['lanes'])]
  assert len({state_digest(h)for h in heads})==1 and torch.equal(rng,torch.random.get_rng_state())
  from native_seven_task_gate_expansion_v1.paired_heads import ARM
  assert all(torch.equal(h.action_head[2].weight[ARM],model.arm_weight)for h in heads)
  headproof.append(dict(group=group,heads=len(heads),qualified_auxiliary_weight_exact=True));del heads
 from .reference_order import before_tail
 for controls in(False,True):
  for reference in(False,True):
   branches=['M_event_full','TTA']+(['zero_update','shadow_update']if controls else[])+(['handoff_reference']if reference else[])
   actual=before_tail(branches);assert set(actual)==set(branches)and actual[0]=='M_event_full'
   if reference:assert actual[1]=='handoff_reference'
 root,spec=qualified_stage(T/'outputs/native_seven_plain_early_v1/plain_000');assert spec['dataset_split']=='TRAIN'
 acceptance_path=T/'outputs/native_seven_early_acceptance_v1/acceptance.json';acceptance=read(acceptance_path);verify(acceptance)
 assert acceptance['passed']and acceptance['matched_native_branches']==15 and acceptance['all_actual_RGB_noise_policy_actions_physics_and_fast_updates_exact']
 acceptance_actual=T/'outputs/native_seven_early_acceptance_driver_v1/actual_exit_v1.json';a=read(acceptance_actual)
 assert a['actual_exit_code']==0 and a['actual_wait_observed']and a['process_absent']
 # Formal stage mechanics: all100 indices exactly once, official starting seed,
 # no TRAIN cache at stage000, and each-session whole controls remain required.
 assert [j for i in range(17)for j in range(*bounds(i))]==list(range(100))
 from .prepared_recovery import collect
 assert collect(None,{t:dict(next_seed=100000,accepted=0)for t in GROUPS['plain']['tasks']},3)is None
 from native_governor180_v5 import checks as policy
 assert len(policy.checks)==39 and read(T/'outputs/native_governor180_v5/CPU_checks_v1.json')['passed']
 assert read(T/'outputs/native_governor180_v5/CPU_actual_exit_v1.json')['exit_code']==0
 assert read(T/'outputs/native_rule8_startup_v1/CPU_checks_v1.json')['passed']
 assert read(T/'outputs/native_rule8_startup_v1/CPU_actual_exit_v1.json')['exit_code']==0
 from .adoption import failed_coordination
 assert failed_coordination()[2]['cleanup_seconds_after_start']==963.156
 from .future_checks import check as check_future
 assert check_future()
 from .carry import check as carry_check
 assert carry_check()
 from .qualification import TRAIN_prerequisites
 train=TRAIN_prerequisites();assert len(train)==2
 for package in('native_gpu_drained_guard_v4',):
  guard=read(T/'outputs'/package/'CPU_checks_v1.json');verify(guard);assert guard['passed']
  assert read(T/'outputs'/package/'CPU_actual_exit_v1.json')['exit_code']==0
 source={str(p):sha(p)for p in C.rglob('*')if p.is_file()and p.suffix in('.py','.sh','.md','.json')}
 for folder in('native_rule8_startup_v1','native_owner_profile_v2','native_render_overlap_v4','native_governor180_v5','native_guide_reuse_v1','native_image_readback_gil_v1','native_batched_prefix_graph_v1','native_direct_frames_v1','native_pot_mic_semantics_v1','native_seven_task_gate_expansion_v1','gated_native_audit_v1','gated_arm_context_v1','declared_reference_audit_v1','gated_handoff_compare_v1','resource_repair_audits_v1','native_gpu_drained_guard_v4','gated_postexit_recovery_v2'):
  source.update({str(p):sha(p)for p in(T/folder).glob('*')if p.is_file()})
 write(T/'outputs'/C.name/'CPU_contract_checks_v1.json',dict(passed=True,no_GPU_or_model_update=True,exact_runtime_inverse_to_qualified_plain_engine=True,queue_checks=queue,head_proofs=headproof,policy_checks=policy.checks,checkpoint=q['training_checkpoint_sha256'],source_sha256=source,input_sha256={**adoption_inputs,**{str(p):sha(p)for p in(T/'outputs/native_rule8_startup_v1/CPU_checks_v1.json',T/'outputs/native_rule8_startup_v1/CPU_actual_exit_v1.json',qualification,queue_proof,queue_actual,acceptance_path,acceptance_actual,root/'terminal_audit.json',T/'outputs/gated_postexit_recovery_v2/qualified_TRAIN/qualification.json',T/'outputs/native_gpu_drained_guard_v4/CPU_checks_v1.json',T/'outputs/native_gpu_drained_guard_v4/CPU_actual_exit_v1.json',T/'outputs/native_governor180_v5/CPU_checks_v1.json',T/'outputs/native_governor180_v5/CPU_actual_exit_v1.json')}}))
 print('SEVEN_TASK_OFFICIAL_CPU_PASS',flush=True)
if __name__=='__main__':main()

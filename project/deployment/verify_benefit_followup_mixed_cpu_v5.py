"""Actual CPU runtime assembly, planner dispatch, and fixed partial-scope checks."""
import ast,copy,json,os,sys,threading,time
from pathlib import Path
import torch
import benefit_followup_protocol_v2 as p
from benefit_followup_stream_v3 import atomic

def planner_proof(deferred=False):
 from benefit_followup_mixed_manager_v3 import planner_class,PREFIX
 if deferred:from deferred_planner import make_class
 else:from shared6_planner_pool import make_class
 from concurrent_core import Coordinator
 from concurrent_rng import Lane
 from model_service import ModelService
 calls=[]
 class Original:
  def __init__(self,key):self.key=key;self.mutable=[];calls.append(key)
  def plan_grippers(self,a,b):return[a,b]
  def plan_path(self,value):self.mutable.append(value);return self.key,list(self.mutable)
  def plan_batch(self,value):return self.plan_path(value)
 c=Coordinator(cuda=False);c.service=ModelService(c)
 pooled,_=make_class(Original,c);policy=Lane('policyA');policy.phase='discarded_probe'if deferred else'expert_screen'
 with c.rng.bind(policy),c.rng.torch_scope(policy):old=pooled('left');old2=pooled('right')
 mixed=planner_class(pooled,c)
 assert isinstance(old,mixed)and isinstance(old2,mixed)
 expert=Lane(PREFIX+'lift_pot');expert.phase='expert_screen'
 peer=Lane(PREFIX+'dump_bin_bigbin');peer.phase='expert_screen'
 with c.rng.bind(expert):private=mixed('private-left');private2=mixed('private-right');assert private.plan_grippers(1,2)==[1,2]
 with c.rng.bind(peer):other=mixed('second-private');assert other.plan_path(3)==('second-private',[3])
 with c.rng.bind(expert):assert private.plan_path(7)==('private-left',[7]);assert private2.plan_batch(8)==('private-right',[8])
 policy.phase='policy'
 with c.rng.bind(policy):
  try:value=old.mutable
  except RuntimeError:pass
  else:raise AssertionError('Pooled mutable-state denial lost')
  assert old.plan_grippers(3,4)==[3,4]
 with c.rng.bind(peer):
  try:private.plan_path(99)
  except RuntimeError:pass
  else:raise AssertionError('Private planner crossed expert lanes')
 result=[];errors=[]
 def worker():
  try:
   with c.rng.bind(expert):result.append(private.plan_batch(11))
  except BaseException as e:errors.append(repr(e))
 t=threading.Thread(target=worker);t.start()
 while t.is_alive():c.service.process_one(.01);t.join(.001)
 assert not errors and result==[('private-left',[7,11])]
 assert calls==([]if deferred else['left','right'])+['private-left','private-right','second-private']
 return dict(existing_pooled_instances_remain_valid=True,original_constructor_arguments_and_order_exact=True,private_expert_state_isolated=True,policy_mutable_state_denial_retained=True,actual_ModelService_owner_routing_verified=True)

def main():
 torch.set_num_threads(1)
 assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized()
 result=planner_proof();result['actual_deferred_planner_architecture_proof']=planner_proof(deferred=True)
 import benefit_followup_mixed_manager_v3 as manager
 # Task/Curobo modules have CUDA-valued import defaults, so actual class loading
 # remains in the real GPU child, before any expert setup. CPU checks bind the
 # exact files and single import edit without executing those CUDA defaults.
 import types
 from task_adapters import evaluate
 prepared=dict(lanes=[True]);manager.prepare_cpu(prepared)
 for row in prepared['mixed_experts']:
  tree=ast.parse((row['runtime']/'envs'/(row['task']+'.py')).read_text())
  before=ast.dump(tree,include_attributes=False)
  hits=[x for x in tree.body if isinstance(x,ast.ImportFrom)and x.level==1 and x.module=='_base_task'];assert len(hits)==1
  hits[0].module='_cpu_fixture_private_base';compile(ast.fix_missing_locations(tree),'<CPU_structural_only>','exec');hits[0].module='_base_task'
  assert ast.dump(tree,include_attributes=False)==before
  assert row['manifest']['task_runtime_sha256']['envs/_base_task.py']==json.loads((p.D/'reports/target_collection_control_cpu_20261024.json').read_text())['instrumented_base_sha256']
  dummy=types.ModuleType('CPU_ONLY_NOT_A_REAL_TASK');row['private_task_module']=dummy
  actual=manager.private_execute(evaluate.execute,row)
  assert actual.__code__ is evaluate.execute.__code__
  assert actual.__globals__['importlib'].import_module('envs.'+row['task'])is dummy
 assert 'envs._base_task'not in sys.modules
 result['expert_module_import_AST_and_original_bytecode_verified_on_CPU']=True
 result['actual_private_module_source_guard_runs_in_GPU_before_expert_setup']=True
 import benefit_growth_runtime as runtime
 runtime.bind()
 import benefit_memory_batches as batches,benefit_growth_deferred_finalize as deferred
 batches.install();deferred.install();runtime.install()
 import benefit_followup_mixed_core_v3 as core
 from shared6_eval_backend import sources,make_plan
 original=json.loads((p.PRIOR/'analysis_preregistration.json').read_text())['source_sha256'];live=sources()
 assert len(live)==83 and all(live[n]==v for n,v in original.items())
 value=core.current_qualification();assert not value['complete']and core.validate_snapshot(value)
 receipt=json.loads((p.ROOT/'finite_bootstrap_receipt_v6.json').read_text())
 assert receipt['passed']and receipt['expert_only']and receipt['complete40']is False
 assert receipt['helper_sha256']==p.sha(p.D/'benefit_followup_bootstrap_prefix_v6.py')
 assert receipt['snapshot_sha256']==p.sha(p.ROOT/'stream_prepare/parallel_screen_completed.json')
 assert {t:r['qualified']for t,r in value['tasks'].items()}==receipt['actual_qualified']
 assert all(2<=r['qualified']<20 for r in value['tasks'].values())
 assert not(p.ROOT/'qualified.json').exists()
 bad=copy.deepcopy(value);bad['tasks']['lift_pot']['attempts'][0]['seed']+=1;assert not core.validate_snapshot(bad)
 old=copy.deepcopy(value);plan=make_plan('benefit40_collect',p.TRAINING,os.environ['CX_TARGET_GPU_UUID'],4,20)
 assert plan['episodes']==88 and [j['manifest']['episodes']for j in plan['jobs']]==[20,4,20,20,4,20]
 assert [j['lane_count']for j in plan['jobs']]==[3,1,3,3,1,3]
 for j in plan['jobs']:
  m=j['manifest'];assert m['pilot']['qualification_stream_scope_sha256']==p.sha(core.scope_path())
  assert str(core.scope_path())in m['source_sha256']and str(p.ROOT/'qualified.json')not in m['source_sha256']and str(p.ROOT/core.EVENTS)not in m['source_sha256']
 # Add an actual historical next expert failure only in a private ledger clone,
 # never to production journal. Plan identity must depend solely on frozen scope.
 ledger=core.state();clone=copy.copy(ledger)
 from benefit_followup_stream_v3 import StreamingEligibility
 clone=StreamingEligibility(p.TASKS,ledger.snapshot,ledger.protocol_sha256)
 task='lift_pot';seed=clone.take(task)
 clone.submit(task,dict(seed=seed,episode_index=0,eligible=False,expert_plan_success=False,expert_success=False,reason='CPU_FIXTURE_NOT_A_REAL_EXPERT_DECISION'))
 core._LEDGER=clone
 repeated=make_plan('benefit40_collect',p.TRAINING,os.environ['CX_TARGET_GPU_UUID'],4,20)
 assert[j['identity']for j in plan['jobs']]==[j['identity']for j in repeated['jobs']]
 core._LEDGER=ledger;make_plan('benefit40_collect',p.TRAINING,os.environ['CX_TARGET_GPU_UUID'],4,20)
 assert core.current_qualification()==old
 from concurrent_queue import EpisodeQueue
 for j in plan['jobs']:
  queue=EpisodeQueue(Path(j['run']),j['manifest']);core.synchronize_queue(queue,core.branch_rows(j['task'],j['branch']))
  assert not queue.state['completed']
  assert core.ExpectedRows(j['task'],j['branch']).get(-1)is None
 assert len(deferred.TRANSFORMS)==1 and deferred.TRANSFORMS[0]['GPU_body_before_finalization_AST_unchanged']
 assert any(r.get('GPU_body_except_two_scheduling_provenance_calls_AST_unchanged')for r in core.RECEIPTS)
 assert any(r.get('original_per_scene_setup_and_hash_guard_AST_unchanged')for r in core.RECEIPTS)
 assert __import__('benefit_prefix_original_reference_v11').RECEIPT['actual_policy_and_original_error_AST_unchanged']
 assert not torch.cuda.is_initialized()and not(p.ROOT/'qualified.json').exists()
 result.update(passed=True,cpu_only=True,cuda_initialized=False,source_sha256=live,original61_unchanged=True,actual_partial_expert_prefix_honest=True,append_only_membership_identity_stable=True,no_production_expert_decisions_added=True,original_per_scene_physical_hash_guard_preserved=True,actual_original_GPU_AST_verified=True,requested_episodes=88,partial_counts={t:r['qualified']for t,r in value['tasks'].items()},scope_sha256=p.sha(core.scope_path()),checker_sha256=p.sha(__file__),runtime_receipts=core.RECEIPTS)
 atomic(p.ROOT/'mixed_runtime_cpu_proof.json',result);print('MIXED_CPU_PROOF',json.dumps({k:v for k,v in result.items()if k!='source_sha256'}),flush=True)
if __name__=='__main__':main()

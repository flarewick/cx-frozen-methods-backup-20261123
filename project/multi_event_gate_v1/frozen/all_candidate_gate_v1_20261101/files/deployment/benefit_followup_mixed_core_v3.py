"""Explicit partial-membership protocol and unchanged per-case scientific checks."""
import ast,copy,inspect,json,linecache,os,threading
from pathlib import Path
import benefit_followup_protocol_v2 as p
from benefit_followup_stream_v3 import StreamingEligibility,extend_queue,digest,atomic
_LEDGER=None
_BEFORE=False
_AFTER=False
RECEIPTS=[]
SCOPE='qualification_stream_contract_v3.json'
EVENTS='qualification_stream_events_v3.json'

def scope_path():return p.ROOT/SCOPE

def prepare_scope():
 p.prepare();file=scope_path();snapshot=p.ROOT/'stream_prepare/parallel_screen_completed.json'
 if not snapshot.exists():raise RuntimeError('A real expert prefix is required before productive mixed work')
 raw=json.loads(snapshot.read_text());ledger=StreamingEligibility(p.TASKS,raw,p.sha(p.ROOT/'protocol.json'))
 if any(r['qualified']==0 for r in ledger.ledger.tasks.values()):raise RuntimeError('Each task needs a finite real expert bootstrap prefix')
 value=dict(schema='fixed_scope_append_only_original_expert_v3',tasks=p.TASKS,target_per_task=20,maximum_per_task=256,protocol_sha256=p.sha(p.ROOT/'protocol.json'),initial_expert_snapshot_sha256=p.sha(snapshot),canonical_initial_expert_snapshot_sha256=ledger.snapshot_sha256,initial_expert_snapshot=str(snapshot),expert_membership='first20 original expert eligible seeds in chronological order, before each corresponding A0',policy_outcomes_never_enter_membership=True,qualification_not_pretended_complete=True,original_per_scene_A0_A1_B_order=True,max_real_scene_slots=6,policy_slots_while_expert_pending=5,active_expert_slots=1,stream_ledger_source_sha256=p.sha(p.D/'benefit_followup_stream_v3.py'))
 if file.exists():
  if json.loads(file.read_text())!=value:raise RuntimeError('Immutable stream scope changed')
 else:atomic(file,value)
 return value

def state():
 global _LEDGER
 if _LEDGER is None:
  contract=prepare_scope();snapshot=json.loads(Path(contract['initial_expert_snapshot']).read_text());log=p.ROOT/EVENTS
  _LEDGER=StreamingEligibility.restore(p.TASKS,snapshot,contract['protocol_sha256'],log)if log.exists()else StreamingEligibility(p.TASKS,snapshot,contract['protocol_sha256'],contract['canonical_initial_expert_snapshot_sha256'])
 return _LEDGER

def current_qualification():
 q=state()
 with q.lock:return dict(complete=q.ledger.complete(),policy_episodes=0,protocol_sha256=q.protocol_sha256,tasks=copy.deepcopy(q.ledger.tasks),source='actual_expert_prefix_no_policy_outcomes')

def validate_snapshot(value):
 if value['protocol_sha256']!=p.sha(p.ROOT/'protocol.json')or value['policy_episodes']!=0:return False
 if set(value['tasks'])!=set(p.TASKS):return False
 for task,start in p.TASKS.items():
  index=0
  for i,row in enumerate(value['tasks'][task]['attempts']):
   state().validate_row(task,row)
   if row['seed']!=start+i or row['episode_index']!=index:return False
   index+=int(row['eligible'])
  if index!=value['tasks'][task]['qualified']or not 0<=index<=20:return False
 return True

def branch_rows(task,branch):return state().branch_rows(task,branch,p.REPEAT,p.VAL)
class ExpectedRows:
 def __init__(self,task,branch):self.task,self.branch=task,branch
 def get(self,seed,default=None):return next((r for r in branch_rows(self.task,self.branch)if r['eligible']and r['seed']==seed),default)
 def values(self):return [r for r in branch_rows(self.task,self.branch)if r['eligible']]

def probe_first(task):
 # A historical expert-qualified TRAIN seed is used only by the original
 # discarded L0/zero parity probe, never as a new policy scene or label.
 source=json.loads((p.PRIOR/'train0/qualified.json').read_text())
 return next(r for r in source['tasks'][task]['attempts']if r['eligible'])

def synchronize_queue(queue,rows):return extend_queue(queue,rows)

def compile_function(function,node,label,namespace=None):
 source=ast.unparse(ast.fix_missing_locations(node))+chr(10);filename='<followup_mixed_'+label+'>'
 linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
 if namespace is None:namespace=dict(function.__globals__)
 namespace.update({n:c.cell_contents for n,c in zip(function.__code__.co_freevars,function.__closure__ or ())})
 exec(compile(source,filename,'exec'),namespace);return namespace[function.__name__]

def edit_install(node):
 before=copy.deepcopy(node);changes=0
 for x in ast.walk(node):
  if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='qualified'for t in x.targets):x.value=ast.Constant(None);changes+=1
  if isinstance(x,ast.FunctionDef)and x.name=='prepare_cpu':
   for i,y in enumerate(x.body):
    if isinstance(y,ast.For):
     for k,z in enumerate(y.body):
      if isinstance(z,ast.If)and ast.unparse(z.test)=="not queue.state['attempts']":
       y.body[k]=ast.parse("__import__('benefit_followup_mixed_core_v3').synchronize_queue(queue,rows)").body[0];changes+=1
  if isinstance(x,ast.FunctionDef)and x.name=='configure':
   for y in x.body:
    if isinstance(y,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='expected'for t in y.targets):y.value=ast.parse("__import__('benefit_followup_mixed_core_v3').ExpectedRows(name,branch)",mode='eval').body;changes+=1
    if isinstance(y,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='first'for t in y.targets):y.value=ast.parse("__import__('benefit_followup_mixed_core_v3').probe_first(name)",mode='eval').body;changes+=1
 if changes!=5:raise RuntimeError('Original installation boundaries changed: '+str(changes))
 # The native setup/replay check itself is preserved, including row/index/hash.
 def setup(tree):return next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name=='setup')
 if ast.dump(setup(before),include_attributes=False)!=ast.dump(setup(node),include_attributes=False):raise RuntimeError('Per-scene physical hash guard changed')
 RECEIPTS.append(dict(original_per_scene_setup_and_hash_guard_AST_unchanged=True,explicit_partial_membership_install_edits=changes))

def edit_plan(node):
 count=0
 for x in ast.walk(node):
  if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='qualification'for t in x.targets):x.value=ast.parse("__import__('benefit_followup_mixed_core_v3').scope_path()",mode='eval').body;count+=1
  if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='qualified'for t in x.targets):x.value=ast.parse("__import__('benefit_followup_mixed_core_v3').current_qualification()",mode='eval').body;count+=1
  if isinstance(x,ast.If)and "qualified['complete']"in ast.unparse(x.test):x.test=ast.parse("not __import__('benefit_followup_mixed_core_v3').validate_snapshot(qualified)",mode='eval').body;count+=1
  if isinstance(x,ast.keyword)and x.arg=='qualification_sha256':x.arg='qualification_stream_scope_sha256';count+=1
 if count!=4:raise RuntimeError('Original immutable plan boundaries changed: '+str(count))
 RECEIPTS.append(dict(new_identity_binds_immutable_expert_scope=True,partial_expert_prefix_explicit_not_fake_complete=True))

def edit_gpu(node):
 # Called before original deferred-tail rewrite; preserve c.verify_weights()
 # adjacent to its original final loop for the original invariant checker.
 before=copy.deepcopy(node);inserted=[]
 for parent in ast.walk(node):
  if not isinstance(parent,(ast.FunctionDef,ast.With,ast.Try)):continue
  body=parent.body
  for i in range(len(body)-1,-1,-1):
   x=body[i]
   if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='threads'for t in x.targets):
    addition=ast.parse("threads.extend(__import__('benefit_followup_mixed_manager_v3').expert_threads(c,prepared,plan,errors))").body[0];body.insert(i+1,addition);inserted.append(addition)
   if isinstance(x,ast.Expr)and isinstance(x.value,ast.Call)and ast.unparse(x.value.func)=='c.verify_weights':
    addition=ast.parse("__import__('benefit_followup_mixed_manager_v3').checkpoint(c,plan)").body[0];body.insert(i,addition);inserted.append(addition)
 if len(inserted)!=2:raise RuntimeError('Original GPU scheduling boundaries changed')
 restored=copy.deepcopy(node)
 for parent in ast.walk(restored):
  for field in ('body','orelse','finalbody'):
   value=getattr(parent,field,None)
   if isinstance(value,list):value[:]=[x for x in value if not(isinstance(x,ast.Expr)and 'benefit_followup_mixed_manager_v3'in ast.unparse(x))]
 if ast.dump(before,include_attributes=False)!=ast.dump(restored,include_attributes=False):raise RuntimeError('Original GPU action/physics/math body changed')
 RECEIPTS.append(dict(GPU_body_except_two_scheduling_provenance_calls_AST_unchanged=True))

def before_runtime():
 global _BEFORE
 if _BEFORE:return
 import benefit_stage_runtime as stage,benefit_stage_protocol as protocol
 prepare_scope();protocol.branch_rows=branch_rows
 original=stage.install;node=ast.parse(inspect.getsource(original));fn=next(n for n in node.body if isinstance(n,ast.FunctionDef));edit_install(fn);stage.install=compile_function(original,fn,'stage_install',stage.__dict__)
 previous=stage.transform
 def transform(function,mutate):
  def edit(node):
   mutate(node)
   if node.name=='make_plan':edit_plan(node)
   if node.name=='run_gpu':edit_gpu(node)
  return previous(function,edit)
 stage.transform=transform;_BEFORE=True

def after_runtime():
 global _AFTER
 if _AFTER:return
 import benefit_memory_batches as batches
 import shared6_eval_worker as worker,shared6_screen_reference as screen,benefit_stage_runtime_v4 as v4
 original_prepare=worker.prepare_cpu
 def prepare(args):
  prepared=original_prepare(args);__import__('benefit_followup_mixed_manager_v3').prepare_cpu(prepared);return prepared
 worker.prepare_cpu=prepare
 screen.load_references=lambda plan:dict(expected={},sources=[dict(qualification_stream_scope=str(scope_path()),sha256=p.sha(scope_path()),no_policy_results=True)])
 original_init=v4.initialize;node=ast.parse(inspect.getsource(original_init)).body[0]
 targets=[x for x in ast.walk(node)if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='q'for t in x.targets)]
 if len(targets)!=1:raise RuntimeError('Discarded constructor qualification boundary changed')
 targets[0].value=ast.parse("__import__('benefit_followup_mixed_core_v3').current_qualification()",mode='eval').body
 v4.initialize=compile_function(original_init,node,'discarded_constructor_prefix')
 from concurrent_queue import EpisodeQueue
 batches.MAX_CLAIMED=6 if state().ledger.complete()else 5
 old=EpisodeQueue.acquire
 def acquire(queue,screen,cancel):
  try:task,branch=p.decode(queue.manifest['condition'])
  except ValueError:return old(queue,screen,cancel)
  with queue.lock:
   while True:
    if cancel.is_set():raise RuntimeError('Mixed work cancelled')
    if batches.requested():return None
    with state().lock:row_count=len(state().ledger.tasks[task]['attempts'])
    if len(queue.state['attempts'])!=row_count:synchronize_queue(queue,branch_rows(task,branch))
    eligible=[r for r in queue.state['attempts']if r['eligible']]
    if state().ledger.complete():return batches.acquire(queue,screen,cancel)
    # The original acquisition accepts a frozen list; while it grows, apply the
    # exact original readiness/claimed/completed checks without an eager return.
    with batches._LOCK:full=len(batches._INFLIGHT)>=batches.MAX_CLAIMED
    if not full:
     for row in eligible:
      if str(row['seed'])in queue.state['completed']or row['seed']in queue.claimed:continue
      if not batches.predecessor(queue,row):continue
      with batches._LOCK:
       if len(batches._INFLIGHT)>=batches.MAX_CLAIMED:break
       key=(str(queue.run),row['seed'])
       if key in batches._INFLIGHT:raise RuntimeError('Duplicate original scene claim')
       batches._INFLIGHT.add(key);queue.claimed.add(row['seed']);return copy.deepcopy(row)
    queue.lock.wait(.05)
 EpisodeQueue.acquire=acquire
 import benefit_growth_deferred_finalize as deferred
 ready=deferred.ready_for_cpu
 def ready_for_cpu(plan):
  folder=Path(os.environ['CX_BENEFIT_RECYCLE_FILE']).parent;intent=folder/'cpu_finalize_intent.json';checkpoint_file=folder/'mixed_expert_checkpoint.json'
  if not checkpoint_file.exists():raise RuntimeError('Mixed expert checkpoint missing before GPU release')
  verify_checkpoint(checkpoint_file);ready(plan);data=json.loads(intent.read_text())
  data['mixed_expert_checkpoint_sha256']=p.sha(checkpoint_file);data['qualification_stream_scope_sha256']=p.sha(scope_path());atomic(intent,data)
 deferred.ready_for_cpu=ready_for_cpu
 finalize=deferred.finalize
 def finalize_checked(path):
  data=json.loads(Path(path).read_text());cp=Path(path).parent/'mixed_expert_checkpoint.json';v=json.loads(cp.read_text())
  verify_checkpoint(cp)
  if data['mixed_expert_checkpoint_sha256']!=p.sha(cp)or v['event_journal_sha256']!=p.sha(p.ROOT/EVENTS)or data['qualification_stream_scope_sha256']!=p.sha(scope_path()):raise RuntimeError('Sealed mixed checkpoint changed')
  return finalize(path)
 deferred.finalize=finalize_checked;_AFTER=True


def verify_checkpoint(path):
 from shared6_eval_backend import sources
 v=json.loads(Path(path).read_text())
 if not v['all_expert_threads_finished']or v['source_sha256']!=sources()or v['qualification_stream_scope_sha256']!=p.sha(scope_path())or v['event_journal_sha256']!=p.sha(p.ROOT/EVENTS):raise RuntimeError('Mixed expert checkpoint provenance changed')
 # Re-read the entire append-only chain, not just trust its mutable hash field.
 contract=prepare_scope();snapshot=json.loads(Path(contract['initial_expert_snapshot']).read_text())
 ledger=StreamingEligibility.restore(p.TASKS,snapshot,contract['protocol_sha256'],p.ROOT/EVENTS)
 if v['complete']!=ledger.ledger.complete()or v['expert_counts']!={t:r['qualified']for t,r in ledger.ledger.tasks.items()}:raise RuntimeError('Expert checkpoint counts differ from strict replay')
 for file,h in v['expert_runtime_audits_sha256'].items():
  if p.sha(Path(file))!=h:raise RuntimeError('Private expert source audit changed')
 if v['complete']:
  q=json.loads((p.ROOT/'qualified.json').read_text())
  if v['qualified_sha256']!=p.sha(p.ROOT/'qualified.json')or q['tasks']!=ledger.ledger.tasks or q['eligibility_event_chain_head']!=ledger.head or q['source_sha256']!=sources():raise RuntimeError('Final expert seal differs from original eligibility chain')
 elif not v['normal_memory_maintenance']:raise RuntimeError('Incomplete experts outside original maintenance')
 return v

"""Real private expert scenes beside unchanged productive policy rollouts."""
import ast,functools,importlib,json,os,sys,threading,time,traceback,types,uuid
from pathlib import Path
import benefit_followup_protocol_v2 as p
import benefit_followup_mixed_core_v3 as core
from benefit_followup_stream_v3 import atomic
KEEP=[]
PREFIX='mixed_original_expert/'
_ACTIVE=threading.BoundedSemaphore(1)
_CONTEXT=None

def planner_class(pooled,c):
 """Keep existing pooled instances valid; experts own original private pairs."""
 original=pooled.__mro__[1]
 def lane():
  value=c.rng.current()
  if value is None:raise RuntimeError('Planner call outside a genuine scene lane')
  return value
 class Meta(type(pooled)):
  def __instancecheck__(cls,value):return isinstance(value,pooled)
 class Mixed(pooled,metaclass=Meta):
  def __init__(self,*a,**kw):
   current=lane();private=current.name.startswith(PREFIX)
   object.__setattr__(self,'_cx_mixed_private',private)
   object.__setattr__(self,'_cx_mixed_owner',current.name)
   if private:
    if threading.get_ident()!=c.service.owner:raise RuntimeError('Original private planner constructor escaped owner')
    original.__init__(self,*a,**kw)
   else:pooled.__init__(self,*a,**kw)
  def __getattribute__(self,name):
   if name.startswith('_cx_')or name in('__class__','__dict__'):return object.__getattribute__(self,name)
   if object.__getattribute__(self,'_cx_mixed_private'):
    if lane().name!=object.__getattribute__(self,'_cx_mixed_owner'):raise RuntimeError('Private expert planner crossed lanes')
    return object.__getattribute__(self,name)
   return pooled.__getattribute__(self,name)
  def __setattr__(self,name,value):
   if name.startswith('_cx_'):return object.__setattr__(self,name,value)
   if object.__getattribute__(self,'_cx_mixed_private'):
    if lane().name!=object.__getattribute__(self,'_cx_mixed_owner'):raise RuntimeError('Private expert state crossed lanes')
    return object.__setattr__(self,name,value)
   return pooled.__setattr__(self,name,value)
 for name in('plan_path','plan_batch'):
  fn=getattr(original,name)
  def make(function):
   @functools.wraps(function)
   def invoke(self,*a,**kw):
    current=lane()
    if not object.__getattribute__(self,'_cx_mixed_private'):return getattr(pooled,function.__name__)(self,*a,**kw)
    if current.name!=object.__getattribute__(self,'_cx_mixed_owner'):raise RuntimeError('Private expert plan crossed lanes')
    def execute():return function(self,*a,**kw)
    execute._cx_priority=0;execute._cx_stage='expert_'+function.__name__
    return c.call(current,execute)
   return invoke
  setattr(Mixed,name,make(fn))
 return Mixed

def prepare_cpu(prepared):
 if not prepared['lanes']or core.state().ledger.complete():prepared['mixed_experts']=[];return
 from cx_tta.native_entry_storage import NativeEntryStorage
 from cx_tta.recording import instrument_runtime
 from task_adapters import eval_resume
 entries=[]
 for task in p.TASKS:
  if core.state().ledger.tasks[task]['qualified']==20:continue
  args=eval_resume.options(['--task',task,'--training-run',str(p.TRAINING),'--condition','Z_zero','--episodes','20','--seed-start',str(p.TASKS[task]),'--max-attempts','256'])
  protocol,reference,tokenizer,manifest=eval_resume.preflight(args)
  session=p.ROOT/task/('mixed_screen_'+uuid.uuid4().hex[:12]);session.mkdir(parents=True)
  for name in('queries','onsets'):(session/name).mkdir()
  storage=NativeEntryStorage(runtime_source=Path(manifest['runtime_source']));runtime=storage.prepare_runtime(session);instrument_runtime(runtime)
  actual={str(f.relative_to(runtime)):p.sha(f)for f in runtime.rglob('*.py')}
  changed=[n for n,h in actual.items()if manifest['task_runtime_sha256'].get(n)!=h]
  proof=json.loads((p.D/'reports/target_collection_control_cpu_20261024.json').read_text())
  if changed!=['envs/_base_task.py']or actual['envs/_base_task.py']!=proof['instrumented_base_sha256']:raise RuntimeError('Original expert private runtime changed')
  manifest['original_task_runtime_sha256']=manifest['task_runtime_sha256'];manifest['task_runtime_sha256']=actual
  atomic(session/'qualified_private_runtime_manifest.json',manifest);args.dataset_run=session;args.initial_completed=0
  entries.append(dict(task=task,args=args,protocol=protocol,reference=reference,tokenizer=tokenizer,manifest=manifest,session=session,storage=storage,runtime=runtime))
 # These import/load only the same archived CPU-native extensions; no task setup.
 from collection_native_step import load as step
 from collection_native_control import load as control
 prepared['mixed_step']=step();prepared['mixed_control']=control();prepared['mixed_experts']=entries

def expert_threads(c,prepared,plan,errors):
 global _CONTEXT
 import envs.robot.robot as robot
 from shared6_eval_worker import capture
 from concurrent_rng import Lane
 from cx_tta.native_entry_storage import NativeEntryStorage
 from task_adapters import evaluate
 from collection_pool import prepare_task
 from benefit_followup_screen_v5 import route_seed
 from benefit_success_shortcircuit import install as exact_success
 from benefit_stage_bounds import install as bounds
 import benefit_memory_batches as batches
 from shared6_eval_backend import sources
 if _CONTEXT is not None:raise RuntimeError('Duplicate mixed expert manager')
 ledger=core.state();ledger.persist(p.ROOT/core.EVENTS)
 entries=prepared.get('mixed_experts',[]);master=prepared['master'];callbacks=[];planners={}
 robot.CuroboPlanner=planner_class(robot.CuroboPlanner,c)
 c.collection_step=prepared.get('mixed_step');c.collection_control=prepared.get('mixed_control')
 for row in entries:
  row['private_task_module']=load_private_task(row)
  class Storage(NativeEntryStorage):
   def __init__(self,row):self.__dict__.update(row['storage'].__dict__);self.row=row
   def prepare_runtime(self,run):
    if Path(run)!=self.row['session']:raise RuntimeError('Expert CPU/GPU runtime scope differs')
    return self.row['runtime']
   def configure_runtime(self,run,runtime,tokenizer):
    self.run=Path(run);self.scratch=master.scratch
    record=json.loads((master.run/'storage.json').read_text());record['shared_process_scratch_owner']=str(master.run);atomic(self.run/'storage.json',record)
   def cleanup(self):pass
  storage=Storage(row);lane=Lane(PREFIX+row['task']);lane.phase='expert_screen'
  with c.rng.bind(lane),c.rng.torch_scope(lane):part=capture(private_execute(evaluate.execute,row),row['args'],row['protocol'],row['reference'],row['tokenizer'],row['manifest'],storage,row['session'])
  task=part['task'];bounds(task,row['task']);original_setup=task.setup_demo
  def zero(*a,_setup=original_setup,_task=task,**kw):
   if a:raise RuntimeError('Original setup positional API changed')
   kw['now_ep_num']=0;_task._cx_tick=0;_task._cx_dense=[];return _setup(**kw)
  task.setup_demo=zero
  prepare_task(task,lane,c,c.native.native)
  setup=task.setup_demo
  def private_setup(*a,_setup=setup,_task=task,_lane=lane,**kw):
   value=_setup(*a,**kw)
   for planner in(_task.robot.left_planner,_task.robot.right_planner):
    owner=planners.setdefault(id(planner),_lane.name)
    if owner!=_lane.name or not object.__getattribute__(planner,'_cx_mixed_private'):raise RuntimeError('Expert planner pair is not private')
   exact_success(_task);return value
  task.setup_demo=private_setup;route_seed(part['screen'],lane,c)
  callbacks.append((row,lane,part,storage));KEEP.extend([row,lane,part,storage])
 _CONTEXT=dict(coordinator=c,prepared=prepared,plan=plan,callbacks=callbacks,sources=sources())
 def seal():
  if ledger.ledger.complete():
   batches.MAX_CLAIMED=6;value=ledger.seal(sources());value.update(qualification_stream_scope_sha256=p.sha(core.scope_path()),gpu_uuid=os.environ['CX_TARGET_GPU_UUID'],expert_runtime_audits={str(r['session']/'source_runtime_audit.json'):p.sha(r['session']/'source_runtime_audit.json')for r,_,_,_ in callbacks if(r['session']/'source_runtime_audit.json').exists()})
   path=p.ROOT/'qualified.json'
   # Membership/provenance is sealed once; audit receipts for later sessions live
   # in their per-batch checkpoint, never rewrite this immutable expert artifact.
   if path.exists():
    old=json.loads(path.read_text())
    for key in('tasks','protocol_sha256','source_sha256','initial_snapshot_sha256','eligibility_event_chain_head','qualification_stream_scope_sha256'):
     if old[key]!=value[key]:raise RuntimeError('Sealed expert qualification changed')
   else:atomic(path,value)
 def worker(row,lane,part,storage):
  try:
   with c.rng.bind(lane):
    while not(master.run/'deployment_compiler_bundle/bundle.json').exists():
     c.check_cancel()
     if batches.requested():break
     time.sleep(.05)
    while not c.cancel.is_set()and not batches.requested():
     if not _ACTIVE.acquire(timeout=.05):continue
     try:
      if batches.requested():break
      seed=ledger.take(row['task'])
      if seed is None:break
      decision=part['screen'](seed,0);decision['deployment_screen_origin']='live_original_screen_private_mixed_planner'
      ledger.submit(row['task'],decision,p.ROOT/core.EVENTS,origin='mixed_original_expert_no_policy_outcomes')
      print('MIXED_ORIGINAL_EXPERT',json.dumps(dict(task=row['task'],seed=seed,eligible=decision['eligible'],qualified=ledger.ledger.tasks[row['task']]['qualified'])),flush=True)
     finally:_ACTIVE.release()
    for name,value in plan['source_sha256'].items():
     if p.sha(p.D/name)!=value:raise RuntimeError('Mixed scientific source changed during work')
    atomic(row['session']/'source_runtime_audit.json',storage.verify_source())
    with ledger.lock:seal()
   if not batches.requested()and not ledger.ledger.complete():
    # A completed task can exit while its peer still works. Exhausted windows fail.
    if ledger.ledger.tasks[row['task']]['qualified']!=20:raise RuntimeError('Original expert seed window exhausted')
  except BaseException:
   message=traceback.format_exc();errors.append(message);c.cancel.set();atomic(row['session']/'failure.json',dict(complete=False,traceback=message));print('MIXED_EXPERT_FAILURE',message,flush=True)
 return[threading.Thread(target=worker,args=entry,name=entry[1].name,daemon=True)for entry in callbacks]

def checkpoint(c,plan):
 import benefit_memory_batches as batches
 from shared6_eval_backend import sources
 if _CONTEXT is None or _CONTEXT['coordinator']is not c or _CONTEXT['sources']!=sources():raise RuntimeError('Mixed manager provenance changed')
 ledger=core.state();ledger.persist(p.ROOT/core.EVENTS)
 audits={}
 for row,_,_,_ in _CONTEXT['callbacks']:
  file=row['session']/'source_runtime_audit.json'
  if not file.exists():raise RuntimeError('Original expert runtime audit missing')
  audits[str(file)]=p.sha(file)
 if not ledger.ledger.complete()and not batches.requested():raise RuntimeError('Incomplete expert scope without normal maintenance')
 barrier=Path(os.environ['CX_BENEFIT_RECYCLE_FILE']).parent
 atomic(barrier/'mixed_expert_checkpoint.json',dict(schema='mixed_original_expert_checkpoint_v3',event_journal_sha256=p.sha(p.ROOT/core.EVENTS),qualification_stream_scope_sha256=p.sha(core.scope_path()),complete=ledger.ledger.complete(),source_sha256=sources(),qualified_sha256=p.sha(p.ROOT/'qualified.json')if(p.ROOT/'qualified.json').exists()else None,expert_runtime_audits_sha256=audits,expert_counts={t:r['qualified']for t,r in ledger.ledger.tasks.items()},all_expert_threads_finished=True,normal_memory_maintenance=batches.requested()))


def load_private_task(row):
 """Separate only the already-CPU-certified instrumented expert base class.

 Policy envs modules stay original. Task methods and imports other than their
 base-class binding are AST-identical, including their original super() cells.
 """
 suffix=row['session'].name.replace('-','_');base_name='envs._cx_mixed_base_'+suffix
 module=importlib.import_module('envs._base_task')
 original_file=Path(module.__file__);before=p.sha(original_file)
 spec=importlib.util.spec_from_file_location(base_name,row['runtime']/'envs/_base_task.py')
 base=importlib.util.module_from_spec(spec);sys.modules[base_name]=base;spec.loader.exec_module(base)
 file=row['runtime']/'envs'/(row['task']+'.py');tree=ast.parse(file.read_text());original=ast.dump(tree,include_attributes=False)
 hits=[n for n in tree.body if isinstance(n,ast.ImportFrom)and n.level==1 and n.module=='_base_task']
 if len(hits)!=1:raise RuntimeError('Original expert base-class import boundary changed')
 hits[0].module=base_name.split('.',1)[1]
 name='envs._cx_mixed_task_'+suffix;task=types.ModuleType(name);task.__file__=str(file);task.__package__='envs';sys.modules[name]=task
 exec(compile(ast.fix_missing_locations(tree),str(file),'exec'),task.__dict__)
 hits[0].module='_base_task'
 if ast.dump(tree,include_attributes=False)!=original or sys.modules['envs._base_task']is not module or p.sha(original_file)!=before:raise RuntimeError('Policy/task scientific source or shared module binding changed')
 cls=getattr(task,row['task']);obj=cls()
 if cls.__mro__[1]is not base.Base_Task:raise RuntimeError('Private expert base-class identity differs')
 proof=json.loads((p.D/'reports/target_collection_control_cpu_20261024.json').read_text())
 import inspect
 if p.sha(Path(inspect.getsourcefile(obj.take_dense_action)))!=proof['instrumented_base_sha256']:raise RuntimeError('Expert uses wrong dense-control recording base')
 atomic(row['session']/'expert_module_isolation_runtime_proof.json',dict(passed=True,verified_before_any_expert_setup=True,task_class_name=cls.__name__,policy_base_module_unchanged=True,policy_base_source_sha256=before,instrumented_expert_base_sha256=p.sha(Path(inspect.getsourcefile(obj.take_dense_action))),original_task_AST_except_base_import_unchanged=True,task_source_sha256=p.sha(file)))
 return task

def private_execute(function,row):
 # Identical original bytecode; replace only the task import result in this
 # callback's private globals, without changing global importlib/sys.modules.
 def load(name,*a,**kw):
  return row['private_task_module']if name=='envs.'+row['task']else importlib.import_module(name,*a,**kw)
 proxy=types.SimpleNamespace(import_module=load,util=importlib.util)
 namespace=dict(function.__globals__,importlib=proxy)
 value=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
 if value.__code__ is not function.__code__:raise RuntimeError('Original evaluate bytecode changed')
 return value

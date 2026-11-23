"""Eight genuine expert scenes; private planners and original GPU owner routing.

Only ordered expert eligibility selects membership. Retained first decisions never
change; complete out-of-order expert results are durable before prefix promotion.
"""
import ast,copy,inspect,json,os,sys,threading,time,traceback,uuid
from pathlib import Path
from benefit_growth_protocol import D,ROOT,TRAINING,TASKS,sha,atomic

FILES=('benefit_growth_parallel_screen.py','benefit_stage_bounds.py','collection_pool.py','collection_capture_gate.py','collection_gc_schedule.py','collection_native_step.py','collection_native_control.py','concurrent_core.py','concurrent_rng.py','model_service.py','native_render.py','native_physics.py','scene_gate.py','collection_step.cpp','collection_control.cpp','collection_optional_open3d.py','benefit_growth_protocol.py','benefit_growth_parallel_screen_v2.py','benefit_growth_parallel_screen_v3.py','benefit_success_shortcircuit.py','benefit_growth_parallel_screen_v4.py','benefit_followup_protocol_v2.py','benefit_followup_bootstrap_v2.py','benefit_followup_screen_v5.py')
KEEP=[]

def source_maps():
 sources={str(D/n):sha(D/n)for n in FILES}
 excluded={str(D/n)for n in ('benefit_followup_protocol_v2.py','benefit_followup_bootstrap_v2.py','benefit_followup_screen_v5.py')}
 previous={k:v for k,v in sources.items()if k not in excluded}
 return sources,previous

def lane_allocation(tasks, target=20, slots=8):
 unfinished=[task for task,row in tasks.items() if row['qualified'] < target]
 counts={task:0 for task in unfinished}
 for i in range(slots if unfinished else 0):counts[unfinished[i%len(unfinished)]]+=1
 return counts

class OrderedEligibility:
 def __init__(self,tasks,prior,maximum=256,target=20):
  self.lock=threading.RLock();self.starts=dict(tasks);self.maximum=maximum;self.target=target
  self.tasks=copy.deepcopy(prior);self.pending={t:{}for t in tasks};self.extra={t:{}for t in tasks}
  for task in tasks:self.tasks.setdefault(task,dict(attempts=[],qualified=0))
  self.next={t:s+len(self.tasks[t]['attempts'])for t,s in tasks.items()}
  self.initial={t:copy.deepcopy(self.tasks[t]['attempts'])for t in tasks}
 def take(self,task):
  with self.lock:
   if self.tasks[task]['qualified']>=self.target:return None
   seed=max(self.next[task],self.starts[task]+len(self.tasks[task]['attempts']))
   while seed in self.pending[task]or seed in self.extra[task]:seed+=1
   if seed>=self.starts[task]+self.maximum:return None
   self.next[task]=seed+1;return seed
 def submit(self,task,row):
  with self.lock:
   seed=row['seed'];rows=self.tasks[task]['attempts'];expected=self.starts[task]+len(rows)
   if seed<expected:raise RuntimeError('Duplicate expert result')
   if seed in self.pending[task]or seed in self.extra[task]:raise RuntimeError('Duplicate pending result')
   self.pending[task][seed]=copy.deepcopy(row)
   while self.tasks[task]['qualified']<self.target:
    expected=self.starts[task]+len(rows)
    if expected not in self.pending[task]:break
    value=self.pending[task].pop(expected);value['episode_index']=self.tasks[task]['qualified']
    rows.append(value);self.tasks[task]['qualified']+=int(value['eligible'])
   if self.tasks[task]['qualified']==self.target:
    self.extra[task].update(self.pending[task]);self.pending[task].clear()
   assert rows[:len(self.initial[task])]==self.initial[task]
 def complete(self):return all(r['qualified']==self.target for r in self.tasks.values())

def route_seed(screen,lane,c):
 from concurrent_core import closure_set
 setup=screen.__closure__[screen.__code__.co_freevars.index('setup')].cell_contents
 original=setup.__closure__[setup.__code__.co_freevars.index('seed_all')].cell_contents
 def seed(value):return c.call(lane,original,value)
 closure_set(setup,'seed_all',seed)
 return screen

def main():
 import torch
 from shared6_eval_worker import capture
 from concurrent_core import Coordinator
 from concurrent_rng import Lane
 from model_service import ModelService
 from native_physics import load_native
 from cx_tta.native_entry_storage import NativeEntryStorage
 from task_adapters import eval_resume,evaluate
 from benefit_success_shortcircuit import install as exact_success
 from collection_pool import route_planners,prepare_task
 if torch.cuda.device_count()!=1:raise RuntimeError('One visible GPU required')
 torch.set_num_threads(1)
 recipe=json.loads((ROOT/'protocol.json').read_text());sources,previous_sources=source_maps()
 prior_file=ROOT/'qualification_progress.json';prior=json.loads(prior_file.read_text())if prior_file.exists()else{}
 if prior and prior['protocol_sha256']!=sha(ROOT/'protocol.json'):raise RuntimeError('Protocol changed')
 if prior:
  allowed={str(D/'benefit_growth_screen.py'):sha(D/'benefit_growth_screen.py'),str(D/'benefit_stage_bounds.py'):sha(D/'benefit_stage_bounds.py')}
  if prior['source_sha256']not in(allowed,previous_sources,sources):raise RuntimeError('Unknown qualification ancestry')
 ledger=OrderedEligibility(TASKS,prior.get('tasks',{}),recipe['maximum_screen_attempts_per_task'])
 result=dict(complete=False,policy_episodes=0,qualification_only=True,gpu_uuid=os.environ['CX_TARGET_GPU_UUID'],protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=sources,previous_source_sha256=prior.get('source_sha256'),private_original_planner_pairs=None,qualification_scheduling='Eight genuine expert slots distributed across unfinished tasks at resume; owner-only planning and RNG; original physics/render/control; ordered eligibility; no policy outcomes')
 saved=ROOT/'parallel_screen_completed.json'
 if saved.exists():
  old=json.loads(saved.read_text())
  if old['protocol_sha256']!=result['protocol_sha256']or old['source_sha256']not in(previous_sources,sources):raise RuntimeError('Pending expert provenance changed')
  if 'tasks'in old:
   for task,previous in ledger.tasks.items():
    if old['tasks'][task]['attempts'][:len(previous['attempts'])]!=previous['attempts']:raise RuntimeError('Write-ahead qualification prefix changed')
   ledger=OrderedEligibility(TASKS,old['tasks'],recipe['maximum_screen_attempts_per_task'])
  for task in TASKS:
   ledger.extra[task].update({int(k):v for k,v in old.get('extra_completed_not_members',{}).get(task,{}).items()})
   for row in sorted(old['pending'].get(task,{}).values(),key=lambda r:r['seed']):ledger.submit(task,row)
   # Pending holes are assigned again; retained complete pending rows are skipped below.
 c=Coordinator(cuda=True);c.service=ModelService(c);c.native=load_native();c.rng.install();KEEP.append(c)
 from collection_native_step import load as load_step
 from collection_native_control import load as load_control
 c.collection_step=load_step();c.collection_control=load_control()
 master=[None];callbacks=[];storage_objects=[];planner_owner={};errors=[]
 def persist():
  with ledger.lock:
   result['tasks']=copy.deepcopy(ledger.tasks)
   atomic(saved,dict(protocol_sha256=result['protocol_sha256'],source_sha256=sources,tasks=copy.deepcopy(ledger.tasks),pending={t:{str(k):v for k,v in r.items()}for t,r in ledger.pending.items()},extra_completed_not_members={t:{str(k):v for k,v in r.items()}for t,r in ledger.extra.items()}))
   atomic(prior_file,result)
 allocation=lane_allocation(ledger.tasks)
 result['lane_allocation']=allocation
 for task in TASKS:
  if ledger.tasks[task]['qualified']==20:continue
  for number in range(allocation[task]):
   args=eval_resume.options(['--task',task,'--training-run',str(TRAINING),'--condition','Z_zero','--episodes','20','--seed-start',str(TASKS[task]),'--max-attempts','256'])
   protocol,reference,tokenizer,manifest=eval_resume.preflight(args)
   session=ROOT/task/('parallel_screen_'+uuid.uuid4().hex[:12]);session.mkdir(parents=True)
   for name in('queries','onsets'):(session/name).mkdir()
   args.dataset_run=session;args.initial_completed=0
   class Storage(NativeEntryStorage):
    def prepare_runtime(self,run):
     if hasattr(self,'parallel_runtime'):return self.parallel_runtime
     self.parallel_runtime=super().prepare_runtime(run)
     from cx_tta.recording import instrument_runtime
     instrument_runtime(self.parallel_runtime);return self.parallel_runtime
    def configure_runtime(self,run,runtime,tokenizer):
     if master[0]is None:super().configure_runtime(run,runtime,tokenizer);master[0]=self
     else:
      self.run=Path(run);self.scratch=master[0].scratch
      record=json.loads((master[0].run/'storage.json').read_text());record['shared_process_scratch_owner']=str(master[0].run);atomic(self.run/'storage.json',record)
    def cleanup(self):pass
   storage=Storage(runtime_source=Path(manifest['runtime_source']));runtime=storage.prepare_runtime(session)
   actual={str(f.relative_to(runtime)):sha(f)for f in runtime.rglob('*.py')}
   changed=[n for n,h in actual.items()if manifest['task_runtime_sha256'].get(n)!=h]
   proof=json.loads((D/'reports/target_collection_control_cpu_20261024.json').read_text())
   if changed!=['envs/_base_task.py']or actual['envs/_base_task.py']!=proof['instrumented_base_sha256']:raise RuntimeError('Private runtime changed')
   manifest['original_task_runtime_sha256']=manifest['task_runtime_sha256'];manifest['task_runtime_sha256']=actual
   atomic(session/'qualified_private_runtime_manifest.json',manifest)
   if not callbacks:
    from collection_optional_open3d import install as optional
    optional(runtime)
   lane=Lane('parallel_qualification/'+task+'/'+str(number))
   with c.rng.bind(lane),c.rng.torch_scope(lane):part=capture(evaluate.execute,args,protocol,reference,tokenizer,manifest,storage,session)
   from benefit_stage_bounds import install as bounds
   bounds(part['task'],task)
   original_setup=part['task'].setup_demo
   def zero(*a,_setup=original_setup,_task=part['task'],**kw):
    if a:raise RuntimeError('Original setup positional API changed')
    kw['now_ep_num']=0;_task._cx_tick=0;_task._cx_dense=[]
    return _setup(**kw)
   part['task'].setup_demo=zero
   prepare_task(part['task'],lane,c,c.native)
   setup=part['task'].setup_demo
   def private_planner_setup(*a,_setup=setup,_task=part['task'],_lane=lane,**kw):
    value=_setup(*a,**kw)
    for planner in(_task.robot.left_planner,_task.robot.right_planner):
     old=planner_owner.setdefault(id(planner),_lane.name)
     if old!=_lane.name:raise RuntimeError('Expert planner shared across independent lanes')
    exact_success(_task)
    return value
   part['task'].setup_demo=private_planner_setup
   route_seed(part['screen'],lane,c)
   callbacks.append((task,lane,part,session));storage_objects.append(storage);KEEP.extend([storage,lane,part])
 # No singleton planner pool: real lanes retain/use their own original planners.
 import envs.robot.robot as robot
 from collection_capture_gate import install as capture_gate
 KEEP.append(capture_gate(robot.CuroboPlanner,c));route_planners(robot.CuroboPlanner,c)
 result['private_original_planner_pairs']=len(callbacks);result['exact_lift_success_optimization']='Archived CPU-proven necessary height predicate; same original success at every physical tick; only after original setup'
 persist()
 def worker(task,lane,part,session):
  try:
   with c.rng.bind(lane):
    while not c.cancel.is_set():
     seed=ledger.take(task)
     if seed is None:return
     with ledger.lock:
      if seed in ledger.pending[task]or seed in ledger.extra[task]:continue
     row=part['screen'](seed,0);row['deployment_screen_origin']='live_original_screen_private_parallel_planner'
     ledger.submit(task,row);persist()
     print('PARALLEL_EXPERT_SCREEN',json.dumps(dict(task=task,seed=seed,eligible=row['eligible'],qualified=ledger.tasks[task]['qualified'])),flush=True)
  except BaseException:
   errors.append(traceback.format_exc());c.cancel.set();print('PARALLEL_EXPERT_FAILURE',errors[-1],flush=True)
 threads=[threading.Thread(target=worker,args=entry,daemon=True)for entry in callbacks]
 for thread in threads:thread.start()
 while any(t.is_alive()for t in threads):
  c.service.process_one(.01)
  for thread in threads:thread.join(.001)
 if errors:raise RuntimeError(errors[0])
 if not ledger.complete():raise RuntimeError('Expert qualification window exhausted')
 result['tasks']=copy.deepcopy(ledger.tasks);result['runtime_audits']=[s.verify_source()for s in storage_objects];result['complete']=True
 atomic(ROOT/'qualified.json',result);print('PARALLEL_40_EXPERT_QUALIFIED',json.dumps({t:r['qualified']for t,r in result['tasks'].items()}),flush=True)
 sys.stdout.flush();sys.stderr.flush();os._exit(0)
if __name__=='__main__':main()

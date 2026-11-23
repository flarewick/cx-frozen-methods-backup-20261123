"""New TRAIN/VAL A/A/single-event pilot, preserving the native executor."""
import ast,copy,inspect,json,os,textwrap,time
from pathlib import Path
from benefit_pilot_protocol import D,ROOT,TRAINING,TASKS,BRANCHES,CANDIDATE,sha,atomic,condition,decode,prepare
FILES=('benefit_pilot_protocol.py','benefit_pilot_policy.py','benefit_pilot_trace.py','benefit_pilot_runtime.py','candidate_eval_checkpoint.py','candidate_eval_reference.py','cross_eval_initial_state.py','benefit_pilot_scene_bounds.py')
def install_sources():
 from shared6_eval_resume import install as resume
 resume()
 import shared6_eval_backend as backend
 if getattr(backend,'_pilot_sources',False):return
 before=backend.sources
 backend.sources=lambda:dict(before(),**{name:sha(D/name)for name in FILES})
 backend._pilot_sources=True

def make_plan(task,training_run,uuid,lanes=4,episodes=100):
 if task!='benefit_pilot_12' or lanes!=4:raise ValueError('Only four fixed pilot chains authorized')
 from task_adapters import eval_resume as original
 from cx_tta.evaluation_resume import identity,load_journal
 import shared6_eval_backend as backend
 prepare();qualification=ROOT/'qualified.json';qualified=json.loads(qualification.read_text())
 if not qualified['complete']or qualified['policy_episodes']!=0 or qualified['protocol_sha256']!=sha(ROOT/'protocol.json'):raise RuntimeError('New eligibility evidence incomplete')
 jobs=[]
 for name,start in TASKS.items():
  for branch in BRANCHES:
   external=condition(name,branch);internal='M_first_full'if branch=='B'else'Z_zero'
   args=original.options(['--task',name,'--training-run',str(training_run),'--condition',internal,'--episodes','2','--seed-start',str(start),'--max-attempts','64','--task-config','demo_randomized'])
   protocol,reference,tokenizer,raw=original.preflight(args)
   raw.update(condition=external,checkpoint_sha256=sha(CANDIDATE),checkpoint_caption='shared250_alpha050_frozen',candidate_stage='new_native_benefit_feasibility_only',comparison='New TRAIN/VAL scenes; not formal EVAL100',pilot=dict(protocol_sha256=sha(ROOT/'protocol.json'),qualification_sha256=sha(qualification),branch=branch,split_by_index=['TRAIN','VAL'],alpha=.5,max_interventions=1 if branch=='B'else 0,offline_ground_truth_for_policy=False,discarded_probe_seed=100003 if name=='dump_bin_bigbin'else 8400000,discarded_probe_policy_actions=0))
   raw['source_sha256'].update({str(D/n):v for n,v in backend.sources().items()})
   for f in (ROOT/'protocol.json',qualification,CANDIDATE,CANDIDATE.parent/'result.json',CANDIDATE.parent.parent/'audit.json',CANDIDATE.parent.parent/'protocol.json'):raw['source_sha256'][str(f)]=sha(f)
   from benefit_pilot_scene_bounds import contract as bounds_contract
   raw['pilot']['initial_clutter_bounds']=bounds_contract(name,qualified['tasks'][name]['attempts'])
   manifest=backend.decorate(raw,uuid,lanes);manifest['deployment_shared6']['protocol']='new_train_val_native_pilot_v1';manifest['deployment_shared6']['condition_lanes']=2;manifest['evaluation_identity']=identity(manifest)
   run=ROOT/('execution_'+manifest['evaluation_identity'][:16])/name/branch;run.mkdir(parents=True,exist_ok=True);(run/'sessions').mkdir(exist_ok=True)
   if(run/'manifest.json').exists():
    if json.loads((run/'manifest.json').read_text())!=manifest:raise RuntimeError('Pilot identity collision')
   else:atomic(run/'manifest.json',manifest)
   state=load_journal(run,manifest)
   jobs.append(dict(task=name,branch=branch,condition=external,run=str(run),identity=manifest['evaluation_identity'],completed=len(state['episodes']),complete=len(state['episodes'])==2,previous_sessions=sorted(p.name for p in(run/'sessions').glob('process_*')),manifest=manifest,protocol=protocol,reference=str(reference),tokenizer=str(tokenizer)))
 plan=dict(task=task,training_run=str(training_run),gpu_uuid=uuid,lanes=lanes,episodes=2,jobs=jobs,source_sha256=backend.sources())
 atomic(ROOT/'active_plan.json',plan);return plan

def transform(function,mutate):
 node=ast.parse(textwrap.dedent(inspect.getsource(function))).body[0];mutate(node)
 ns=dict(function.__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(D/'benefit_pilot_runtime.py'),'exec'),ns);return ns[function.__name__]

def install():
 install_sources()
 import shared6_eval_backend as backend,shared6_eval_worker as worker
 if getattr(backend,'_pilot_installed',False):return
 backend.make_plan=make_plan
 from task_adapters import eval_resume as original
 old_options=original.options
 def options(argv=None):
  import sys
  values=list(sys.argv[1:]if argv is None else argv);pos=values.index('--condition')+1;external=values[pos]
  try:_,branch=decode(external)
  except ValueError:return old_options(values)
  values[pos]='M_first_full'if branch=='B'else'Z_zero';value=old_options(values);value.condition=external;return value
 original.options=options
 def cpu_edit(node):
  calls=[n for n in ast.walk(node)if isinstance(n,ast.Call)and ast.unparse(n.func)=='range'and len(n.args)==1 and ast.unparse(n.args[0])=='a.lanes // 2']
  if len(calls)!=1:raise RuntimeError('Original scene allocation changed')
  calls[0].args[0]=ast.Constant(2)
 original_prepare=transform(worker.prepare_cpu,cpu_edit)
 def prepare_cpu(args):
  prepared=original_prepare(args)
  from concurrent_queue import EpisodeQueue
  qualified=json.loads((ROOT/'qualified.json').read_text())
  for job in prepared['plan']['jobs']:
   queue=EpisodeQueue(Path(job['run']),job['manifest']);rows=qualified['tasks'][job['task']]['attempts']
   if not queue.state['attempts']:queue.state['attempts']=copy.deepcopy(rows);queue.validate();queue._persist();queue._commit()
   elif queue.state['attempts']!=rows:raise RuntimeError('Qualification changed on resume')
  return prepared
 worker.prepare_cpu=prepare_cpu
 import shared6_screen_reference as screen
 screen.load_references=lambda plan:dict(expected={},sources=[dict(new_qualification=str(ROOT/'qualified.json'),sha256=sha(ROOT/'qualified.json'),no_policy_results=True)])
 # Keep the original worker body; vary only task-specific arguments and native tap.
 def gpu_edit(node):
  changed=0
  for call in [n for n in ast.walk(node)if isinstance(n,ast.Call)and ast.unparse(n.func)=='original.options']:
   call.args[0]=ast.parse("['--task',job['task'],'--training-run',str(a.training_run),'--condition',job['condition'],'--episodes','2','--seed-start',str(job['manifest']['seed_start']),'--max-attempts','64','--task-config','demo_randomized']",mode='eval').body;changed+=1
  for assignment in [n for n in ast.walk(node)if isinstance(n,ast.Assign)and any(ast.unparse(t)=='c.native'for t in n.targets)]:
   if ast.unparse(assignment.value)!='load_native()':raise RuntimeError('Native load changed')
   assignment.value=ast.parse("__import__('benefit_pilot_trace').NativeTap(load_native(),c)",mode='eval').body;changed+=1
  if changed!=2:raise RuntimeError('Original worker call boundaries changed')
 original_gpu=transform(worker.run_gpu,gpu_edit)
 def run_gpu(*args,**kwargs):
  import task_adapters.policy as policy
  from benefit_pilot_policy import PilotPolicy
  policy.StackingPolicy=PilotPolicy
  return original_gpu(*args,**kwargs)
 worker.run_gpu=run_gpu
 import shared6_eval_core as core
 from concurrent_core import closure_set
 previous_configure=core.configure
 qualified=json.loads((ROOT/'qualified.json').read_text());plans={}
 def configure(callbacks,c,lane):
  value=previous_configure(callbacks,c,lane);rollout=value['rollout'];args=value['args'];name,branch=decode(args.condition);task=value['task'];session=Path(value['run'])
  from benefit_pilot_trace import install as trace_install,chunk_recorder,install_scene_clock
  from benefit_pilot_scene_bounds import install as isolate_bounds
  isolate_bounds(task,name,qualified['tasks'][name]['attempts'])
  lane.pilot_task=task;task._pilot_recording=False
  closure_set(rollout,'step_chunk_official',chunk_recorder(value['step_chunk_official'],session))
  rows=qualified['tasks'][name]['attempts'];expected={r['seed']:r for r in rows if r['eligible']};first=next(iter(expected.values()))
  original_probe=rollout.__closure__[rollout.__code__.co_freevars.index('steady_probe')].cell_contents
  def probe(*args,**kwargs):
   from envs.utils.create_actor import UnStableError
   values=list(args);values[2]=100003 if name=='dump_bin_bigbin'else first['seed']
   # Discarded sampler parity uses a previously verified stable index0
   # dump scene, with no policy actions or data retained for learning.
   # Formal setup below still requires the four NEW exact physical hashes.
   for attempt in range(1,6):
    try:return original_probe(*values,**kwargs)
    except UnStableError as error:
     c.note('pilot_discarded_probe_unstable',seed=values[2],attempt=attempt,error=str(error))
     if attempt==5:raise
  closure_set(rollout,'steady_probe',probe)
  previous_setup=task.setup_demo
  def setup(*args,**kwargs):
   task._pilot_recording=False;task._pilot_scene_seed=kwargs.get('seed')
   if getattr(lane,'phase',None)=='discarded_probe':return previous_setup(*args,**kwargs)
   if args:raise RuntimeError('Unexpected native setup arguments')
   seed=kwargs['seed'];index=kwargs['now_ep_num'];row=expected.get(seed)
   if row is None or row['episode_index']!=index:raise RuntimeError('Outside four fixed pilot scenes')
   from cross_eval_initial_state import replay
   from cx_tta.evaluate_b2 import seed_all
   from cx_tta.attribution_physics import physical_snapshot
   from cx_tta.eval_diagnostics import json_hash
   from envs.utils.create_actor import UnStableError
   result=replay(previous_setup,task.close_env,seed_all,lambda:json_hash(physical_snapshot(task)),row['initial_physical_sha256'],UnStableError,seed=seed,index=index,kwargs={k:v for k,v in kwargs.items()if k not in('seed','now_ep_num')},note=lambda **f:c.note('pilot_fixed_new_initial_state',**f))
   if not getattr(task,'_pilot_trace_installed',False):trace_install(task,lane,c);task._pilot_trace_installed=True
   install_scene_clock(task)
   task._pilot_ticks=0;task._pilot_stem=f'episode{index:03d}_seed{seed}';task._pilot_evidence=[];task._pilot_recording=True;return result
  task.setup_demo=setup
  def ordered(seed,index,row):
   # Waiting happens on a scene thread, never the sampler/GPU owner thread.
   if branch!='A0':
    plan=json.loads((ROOT/'active_plan.json').read_text());predecessor='A0'if branch=='A1'else'A1';job=next(j for j in plan['jobs']if j['task']==name and j['branch']==predecessor);file=Path(job['run'])/'concurrency_queue.json'
    while str(seed)not in json.loads(file.read_text())['completed']:
     c.check_cancel();time.sleep(.05)
   task._pilot_recording=False;result=rollout(seed,index,row);task._pilot_recording=False
   for f in set(task._pilot_evidence):result['evidence_sha256'][str(f)]=sha(f)
   result.update(pilot_branch=branch,pilot_split='TRAIN'if index==0 else'VAL',native_physics_ticks=task._pilot_ticks,passive_trace_complete=True)
   return result
  value['rollout']=ordered;return value
 core.configure=configure
 backend._pilot_installed=True

def main():
 from shared6_eval_prepared import main as prelude
 def edit(node):
  index=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='torch.set_num_threads')
  node.body[index+1:index+1]=ast.parse("__import__('benefit_pilot_runtime').install()").body
 transform(prelude,edit)()
if __name__=='__main__':main()

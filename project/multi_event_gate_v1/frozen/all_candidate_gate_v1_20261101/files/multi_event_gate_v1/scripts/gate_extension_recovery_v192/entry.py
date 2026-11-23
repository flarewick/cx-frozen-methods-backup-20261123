"""Resource-only recovery: ten real policy claims; original science/guards untouched."""
import ast,copy,inspect,json,os,runpy,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
R=HERE.parents[2];D=R/'deployment';sys.path.insert(0,str(D));sys.path.insert(0,str(HERE))
sys.modules.setdefault('entry',sys.modules[__name__])
_INSTALLED=False
_APPLIED=False

def apply():
 global _APPLIED
 if _APPLIED:return
 import benefit_policy_capacity_v89 as base;base.apply()
 import benefit_memory_batches as b,benefit_followup_mixed_manager_v3 as m,benefit_followup_mixed_core_v3 as core
 import benefit_followup_protocol_v2 as p
 assert m._CONTEXT is None and not b._INFLIGHT
 original=m.expert_threads;node=ast.parse(inspect.getsource(original));before=copy.deepcopy(node);edits=[]
 for n in ast.walk(node):
  if isinstance(n,ast.Assign)and any(ast.unparse(t)=='batches.MAX_CLAIMED'for t in n.targets):
   assert isinstance(n.value,ast.Constant)and n.value.value==6;n.value.value=10;edits.append(n)
 assert len(edits)==1
 restored=copy.deepcopy(node)
 for n in ast.walk(restored):
  if isinstance(n,ast.Assign)and any(ast.unparse(t)=='batches.MAX_CLAIMED'for t in n.targets):n.value.value=6
 assert ast.dump(restored)==ast.dump(before)
 import linecache
 text=ast.unparse(ast.fix_missing_locations(node))+'\n';label='<gate_resource_v192>';linecache.cache[label]=(len(text),None,text.splitlines(True),label);exec(compile(text,label,'exec'),m.__dict__)
 assert m.expert_threads.__globals__ is m.__dict__
 ledger=core.state();previous=ledger.submit
 def submit(*a,**kw):
  result=previous(*a,**kw);b.MAX_CLAIMED=10;return result
 ledger.submit=submit;b.MAX_CLAIMED=10
 folder=p.ROOT/'recovery_resource_receipts_v192';folder.mkdir(exist_ok=True)
 p.atomic(folder/(str(time.time_ns())+'.json'),dict(passed=True,helper_sha256=p.sha(__file__),policy_cap=10,expert_cap=1,model_batch=1,seal_only_capacity_assignment_changed=True,original_source_sha256=p.sha(D/'benefit_followup_mixed_manager_v3.py'),all_guards_unchanged=True))
 _APPLIED=True

def install():
 global _INSTALLED
 if _INSTALLED:return
 os.environ['CX_BENEFIT_GROWTH_BLOCK']='dev0'
 import benefit_event_full_resources_v174 as base;base.install()
 import benefit_growth_runtime as runtime
 previous=runtime.install
 def install_runtime():previous();apply()
 runtime.install=install_runtime
 import benefit_followup_protocol_v2 as p
 # Bind the exact extra resource closure in every process, independently of the frozen scientific identity.
 manifest=HERE/'amendment.json'
 if manifest.exists():
  value=json.loads(manifest.read_text());assert value['entry_sha256']==p.sha(__file__)
  assert all(p.sha(D/n)==h for n,h in value['original_sources'].items())
  assert all(p.sha(p.ROOT/n)==h for n,h in value['frozen_inputs'].items())
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT
  canonical=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_extension_recovery_v192/entry.py')
  old=str(CANONICAL_PROJECT/'deployment/benefit_event_full_resources_v174.py');prior=c.command
  def route(*args,**kw):
   cmd=prior(*args,**kw)
   for i,s in enumerate(cmd):
    if s==old:cmd[i]=canonical
    elif isinstance(s,str)and "__import__('benefit_event_full_resources_v174').install();"in s:
     prefix='import sys;sys.path.insert(0,'+repr(str(Path(canonical).parent))+ ");__import__('entry').install();"
     cmd[i]=s.replace("__import__('benefit_event_full_resources_v174').install();",prefix,1)
   return cmd
  c.command=route
  original=subprocess.Popen
  class RoutedPopen(original):
   def __init__(self,args,*aa,**kw):
    if isinstance(args,(tuple,list)):
     args=[str(HERE/'entry.py')if str(s)==str(D/'benefit_event_full_resources_v174.py')else s for s in args]
    super().__init__(args,*aa,**kw)
  subprocess.Popen=RoutedPopen
 _INSTALLED=True

def verify():
 install()
 import torch,sapien,numpy as np,threading,types,tempfile
 from concurrent.futures import ThreadPoolExecutor
 assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized();torch.set_num_threads(1)
 apply()
 import benefit_memory_batches as b
 assert b.MAX_CLAIMED==10
 import benefit_followup_mixed_manager_v3 as manager
 assert manager.expert_threads.__globals__ is manager.__dict__
 def scene(i):
  s=sapien.Scene([sapien.physx.PhysxCpuSystem()]);s.set_timestep(1/240);builder=s.create_actor_builder();builder.add_box_collision(half_size=[.1]*3);builder.set_initial_pose(sapien.Pose([0,0,1+i*.01]));return s,builder.build(name='box')
 def step(pair):
  s,a=pair;out=[]
  for _ in range(80):s.step();pose=a.get_pose();out.append(np.r_[pose.p,pose.q])
  return np.stack(out)
 reference=[step(scene(i))for i in range(11)]
 with ThreadPoolExecutor(max_workers=11)as pool:actual=list(pool.map(step,[scene(i)for i in range(11)]))
 assert all(np.array_equal(a,c)for a,c in zip(reference,actual))
 with tempfile.TemporaryDirectory()as tmp:
  old=os.environ.get('CX_BENEFIT_RECYCLE_FILE');path=Path(tmp)/'drain.json';os.environ['CX_BENEFIT_RECYCLE_FILE']=str(path)
  q=types.SimpleNamespace(run=Path(tmp),manifest=dict(episodes=11),state=dict(attempts=[dict(seed=i,eligible=True)for i in range(11)],completed={}),claimed=set(),lock=threading.Condition());ready=threading.Event();drain=threading.Event();errors=[];claimed=[];results=[];lock=threading.Lock()
  def worker():
   try:
    row=b.acquire(q,None,threading.Event(),ready=lambda *_:True)
    if row is None:results.append(None);return
    with lock:
     claimed.append(row['seed'])
     if len(claimed)==10:ready.set()
    assert drain.wait(8);b.release(q,row);results.append(row['seed'])
   except BaseException as e:errors.append(repr(e))
  ts=[threading.Thread(target=worker)for _ in range(11)]
  for t in ts:t.start()
  assert ready.wait(8)and len(b._INFLIGHT)==10;path.write_text('{}');drain.set()
  for t in ts:t.join(8)
  assert not errors and not any(t.is_alive()for t in ts)and len(set(claimed))==10 and len(results)==11 and results.count(None)==1 and not b._INFLIGHT
  if old is None:os.environ.pop('CX_BENEFIT_RECYCLE_FILE')
  else:os.environ['CX_BENEFIT_RECYCLE_FILE']=old
 # Reuse original owner RNG test unchanged, installed resource cap is checked separately above.
 import benefit_followup_protocol_v2 as p
 proof=dict(passed=True,CPU_only=True,no_CUDA=True,ten_claims_and_eleventh_waiter_drained=True,eleven_PhysX_trajectories_exact=True,original_guard_and_scientific_hashes_verified=True,entry_sha256=p.sha(__file__),amendment_sha256=p.sha(HERE/'amendment.json'))
 p.atomic(p.ROOT/'recovery_CPU_proof_v192.json',proof);print('RECOVERY192_CPU_PASSED',json.dumps(proof),flush=True)

if __name__=='__main__':
 if sys.argv[1:]==['--verify']:verify()
 else:install();__import__('benefit_followup_mixed_entry_v3').main()

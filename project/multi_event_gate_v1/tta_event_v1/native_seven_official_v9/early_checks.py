"""Real original ModelService/RNG: branches may all finish before their full leader."""
import threading,time,random,pickle
import numpy as np,torch
from .configuration import GROUPS
from .work import FixedWorkQueue
from .branch_queue import BranchQueue
from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
from concurrent_rng import Lane

def witness(case):
 return dict(query=0,actor_observation_tick=0,base_infer_calls=1,actual_internal_noise_calls=1,native_environment_unchanged=True,source_policy_file=case+'/M_event_full/policy000.npz',source_policy_sha256='0'*64,only_q0_provenance_no_full_result_or_labels=True,full_result_absent_at_dispatch=True)

def trial(tasks,lanes,parallel,*,per_task=1,prefix=None):
 torch.manual_seed(71);np.random.seed(71);random.seed(71);c=Coordinator();c.rng.install();count=len(tasks)*per_task
 prefix=count if prefix is None else prefix;cached={t:dict(items=[])for t in tasks}
 for n in range(prefix):cached[tasks[n%len(tasks)]]['items'].append({})
 work=FixedWorkQueue(count,tasks,cached,c.cancel);jobs=BranchQueue(work,count,c.cancel,early=True)
 names=['M_event_full','handoff_reference','TTA','zero_update','shadow_update']
 def branches(n):return list(names)if n<len(tasks)else['M_event_full','TTA']
 outputs={};states={};lock=threading.Lock();errors=[];threads=[];finish_order=[]
 def begin(number,item,name):
  physical=Lane('physical'+str(number));actor=Lane(item['case_id']+'_'+name)
  def seed_sim():torch.manual_seed(item['seed']);np.random.seed(item['seed']);random.seed(item['seed'])
  def seed_actor():torch.manual_seed(17);np.random.seed(17);random.seed(17)
  def draw():return np.r_[torch.rand(4).numpy(),np.random.rand(4),random.random()]
  c.call(physical,seed_sim);initial=c.call(physical,draw);c.call(actor,seed_actor)
  return actor,draw,[initial,c.call(actor,draw)]
 def finish(item,name,actor,draw,values):
  values.extend(c.call(actor,draw)for _ in range(3))
  with lock:
   key=(item['case_id'],name);assert key not in outputs;outputs[key]=tuple(values)
   states[key]=pickle.dumps((actor.numpy.get_state(),actor.python.getstate(),actor.torch_cpu.numpy()));finish_order.append(key)
 def done(item,shared,name):
  with shared['lock']:
   assert name not in shared['done'];shared['done'].add(name)
   if shared['done']==set(shared['branches']):jobs.complete(item['case_id'])
 def worker(number):
  try:
   if not parallel:
    for n in range(count):
     item=dict(case_id=str(n),seed=100+n)
     for name in branches(n):
      a,d,v=begin(number,item,name);finish(item,name,a,d,v)
    return
   while True:
    job=jobs.take()
    if job is None:return
    kind,value,shared,name=job
    if kind=='full':
     n=value;work.mark_selected(n);item=dict(case_id=str(n),seed=100+n);work.register_case(item['case_id'],n)
     a,d,v=begin(number,item,'M_event_full');work.mark_case_ready(item['case_id'])
     shared=dict(branches=branches(n),done=set(),lock=threading.Lock(),early_pairing=True)
     jobs.publish_initial(item,shared,witness(item['case_id']))
     # Force full to finish LAST, proving finalization cannot rely on full-first.
     deadline=time.monotonic()+8
     while n==0:
      with shared['lock']:ready=len(shared['done'])==len(shared['branches'])-1
      if ready:break
      assert not c.cancel.is_set()and time.monotonic()<deadline;time.sleep(.001)
     finish(item,'M_event_full',a,d,v);done(item,shared,'M_event_full')
    else:
     item=value
     try:
      a,d,v=begin(number,item,name);finish(item,name,a,d,v);done(item,shared,name)
     finally:jobs.retire(job)
  except BaseException as e:errors.append(repr(e));c.cancel.set()
 try:
  threads=[threading.Thread(target=worker,args=(n,))for n in range(lanes if parallel else 1)]
  for t in threads:t.start()
  deadline=time.monotonic()+15
  while any(t.is_alive()for t in threads):assert time.monotonic()<deadline,errors;c.service.process_one(.005)
  for t in threads:t.join()
  assert not errors,errors
  assert len(outputs)==2*count+3*len(tasks) and not c.service.requests.qsize()
  if parallel:
   assert jobs.completed==jobs.submitted=={str(n)for n in range(count)}and not jobs.pending
   assert jobs.active_references==0 and jobs.max_active_references==1
   assert [name for case,name in finish_order if case=='0'][-1]=='M_event_full'
  return outputs,states
 finally:
  c.cancel.set()
  for t in threads:t.join(1);assert not t.is_alive()
  c.rng.uninstall()

def check():
 for group,cfg in GROUPS.items():
  a=trial(cfg['tasks'],cfg['lanes'],False);b=trial(cfg['tasks'],cfg['lanes'],True)
  assert a[1]==b[1]and a[0].keys()==b[0].keys()
  assert all(x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()for key in a[0]for x,y in zip(a[0][key],b[0][key],strict=True))
 for group,cfg in GROUPS.items():
  tasks=cfg['tasks'];count=6*len(tasks);a=trial(tasks,cfg['lanes'],False,per_task=6)
  for prefix in(0,1,len(tasks),count-1,count):
   b=trial(tasks,cfg['lanes'],True,per_task=6,prefix=prefix)
   assert a[1]==b[1]and a[0].keys()==b[0].keys()
   assert all(x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()for key in a[0]for x,y in zip(a[0][key],b[0][key],strict=True))
 tasks=GROUPS['plain']['tasks'];work=FixedWorkQueue(3,tasks,{},threading.Event());n=work.take();work.mark_selected(n);work.register_case('case',n);work.mark_case_ready('case')
 shared=dict(branches=['M_event_full','TTA'],done=set(),early_pairing=True)
 for field,value in(('query',1),('base_infer_calls',2),('native_environment_unchanged',False),('full_success',True)):
  q=witness('case');q[field]=value;jobs=BranchQueue(work,3,threading.Event(),early=True)
  try:jobs.publish_initial(dict(case_id='case'),shared,q)
  except AssertionError:pass
  else:raise AssertionError('Invalid q0/label publication accepted')
 jobs=BranchQueue(work,3,threading.Event(),early=True);jobs.publish_initial(dict(case_id='case'),shared,witness('case'))
 try:jobs.publish_initial(dict(case_id='case'),shared,witness('case'))
 except AssertionError:pass
 else:raise AssertionError('Duplicate early dispatch')
 return True

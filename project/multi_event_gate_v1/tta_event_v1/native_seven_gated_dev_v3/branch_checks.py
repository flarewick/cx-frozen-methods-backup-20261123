"""Actual original ModelService/RNG under serial and bounded whole-branch scheduling."""
import threading,time,random,pickle
import numpy as np,torch
from .branch_queue import BranchQueue
from .work import FixedWorkQueue
from .common import TASKS
from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
from concurrent_rng import Lane

def trial(parallel,prepared=False):
 torch.manual_seed(71);np.random.seed(71);random.seed(71);c=Coordinator();c.rng.install();cancel=c.cancel
 cached={t:dict(items=[{},{}])for t in TASKS}if prepared else{}
 work=FixedWorkQueue(8,TASKS,cached,cancel);jobs=BranchQueue(work,8,cancel);outputs={};states={};lock=threading.Lock();errors=[];threads=[]
 def branch(number,item,name):
  physical=Lane('physical'+str(number));actor=Lane(item['case_id']+'_'+name)
  def seed_sim():torch.manual_seed(item['seed']);np.random.seed(item['seed']);random.seed(item['seed'])
  def seed_actor():torch.manual_seed(17);np.random.seed(17);random.seed(17)
  def draw():return np.r_[torch.rand(4).numpy(),np.random.rand(4),random.random()]
  c.call(physical,seed_sim);initial=c.call(physical,draw)
  c.call(actor,seed_actor);value=[c.call(actor,draw)for _ in range(4)]
  with lock:
   key=(item['case_id'],name);assert key not in outputs;outputs[key]=(initial,*value)
   states[key]=pickle.dumps((actor.numpy.get_state(),actor.python.getstate(),actor.torch_cpu.numpy()))
 def branches(n):return ['M_event_full','handoff_reference','TTA','baseline','zero_update','shadow_update']if n<4 else['M_event_full','TTA','baseline']
 def worker(number):
  try:
   if not parallel:
    for n in range(8):
     item=dict(case_id=str(n),seed=100+n)
     for name in branches(n):branch(number,item,name)
    return
   while True:
    job=jobs.take()
    if job is None:return
    kind,value,shared,name=job
    if kind=='full':
     n=value;work.mark_selected(n);item=dict(case_id=str(n),seed=100+n);work.register_case(item['case_id'],n)
     branch(number,item,'M_event_full');shared=dict(branches=branches(n),done={'M_event_full'},lock=threading.Lock());work.mark_case_ready(item['case_id']);jobs.submit(item,shared)
    else:
     item=value;branch(number,item,name)
     with shared['lock']:
      assert name not in shared['done'];shared['done'].add(name)
      if shared['done']==set(shared['branches']):jobs.complete(item['case_id'])
  except BaseException as e:errors.append(repr(e));cancel.set()
 try:
  threads=[threading.Thread(target=worker,args=(n,))for n in range(12 if parallel else 1)]
  for t in threads:t.start()
  end=time.monotonic()+15
  while any(t.is_alive()for t in threads):
   assert time.monotonic()<end,errors;c.service.process_one(.005)
  for t in threads:t.join()
  assert not errors,errors
  assert len(outputs)==36 and not c.service.requests.qsize()
  if parallel:
   assert jobs.leaders==8 and jobs.completed==jobs.submitted==set(str(n)for n in range(8))and not jobs.pending
   assert jobs.take()is None
   try:jobs.complete('0')
   except AssertionError:pass
   else:raise AssertionError('Duplicate finalization accepted')
  return outputs,states
 finally:
  cancel.set()
  for t in threads:t.join(1);assert not t.is_alive()
  c.rng.uninstall()

def check():
 a=trial(False)
 for prepared in(False,True):
  b=trial(True,prepared=prepared);assert a[1]==b[1]and a[0].keys()==b[0].keys()
  assert all(x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()for key in a[0]for x,y in zip(a[0][key],b[0][key],strict=True))
 cancel=threading.Event();cancel.set();q=BranchQueue(FixedWorkQueue(8,TASKS,{},cancel),8,cancel)
 try:q.take()
 except InterruptedError:pass
 else:raise AssertionError('Cancelled work dispatched')
 return True

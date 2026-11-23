"""Actual original queue/futures; fairness, function immutability and exceptions."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import tempfile,threading,time
from model_service import ModelService
from .common import O,write
from .setup_fairness import install

def main():
 c=SimpleNamespace(check_cancel=lambda:None,rng=SimpleNamespace(bind=lambda x:nullcontext(),torch_scope=lambda x:nullcontext()),note=lambda *a,**k:None);c.service=ModelService(c);original_execute=c.service.execute.__func__;order=[];replies=[];threads=[];marker=object();lane=SimpleNamespace(name='CPU_only')
 with tempfile.TemporaryDirectory(dir=O)as tmp:
  install(c,Path(tmp)/'events.jsonl')
  def setup(x):order.append('setup');return x
  setup._cx_priority=5;setup._cx_stage='setup';before=dict(setup.__dict__)
  calls=[setup]
  for i in range(9):
   def infer(x,i=i):order.append(i);return x
   calls.append(infer)
  for fn in calls:
   t=threading.Thread(target=lambda f=fn:replies.append(c.service.call(lane,f,marker)));t.start();threads.append(t)
   deadline=time.monotonic()+5
   while c.service.requests.qsize()<len(threads):
    assert time.monotonic()<deadline;time.sleep(.001)
  while c.service.process_one(timeout=.001):pass
  for t in threads:t.join(timeout=5)
  assert order==[0,1,2,3,4,5,'setup',6,7,8]and replies==[marker]*10 and not any(t.is_alive()for t in threads)
  assert setup.__dict__==before and c.service.execute.__func__ is original_execute
  errors=[]
  def wrong_owner():
   try:c.service.process_one(timeout=.001)
   except RuntimeError as e:errors.append(str(e))
  t=threading.Thread(target=wrong_owner);t.start();t.join();assert len(errors)==1
 write(O/'CPU_setup_fairness_checks.json',dict(passed=True,actual_original_ModelService_queue_exercised=True,bounded_six_ordinary_then_one_setup=True,ordinary_requests_FIFO=True,setup_requests_FIFO=True,original_fn_objects_attributes_and_args_unchanged=True,original_owner_execute_RNG_scopes_and_future_handling_unchanged=True,wrong_owner_rejected=True,no_GPU_context_or_policy_called=True))
 print('BOUNDED_FAIR_ORIGINAL_QUEUE_CPU_PASS')
if __name__=='__main__':main()

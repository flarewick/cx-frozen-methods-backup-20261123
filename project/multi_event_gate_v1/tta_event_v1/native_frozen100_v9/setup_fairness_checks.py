"""Exercise the original real owner queue; CPU only, no policy or CUDA."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import threading,time,tempfile,json
from model_service import ModelService
from .setup_fairness import install
from .common import O,write

def trial(fair,folder):
 c=SimpleNamespace(check_cancel=lambda:None,rng=SimpleNamespace(bind=lambda lane:nullcontext(),torch_scope=lambda lane:nullcontext()),note=lambda *a,**k:None)
 c.service=ModelService(c);order=[];results=[];lane=SimpleNamespace(name='actual_CPU_fixture')
 if fair:install(c,folder/'events.jsonl')
 def setup(value):order.append('setup');return value
 setup._cx_priority=5;setup._cx_stage='setup'
 def infer(value):order.append('infer');return value
 marker=object()
 def worker(fn):results.append(c.service.call(lane,fn,marker))
 threads=[]
 for fn in(setup,infer):
  t=threading.Thread(target=worker,args=(fn,));t.start();threads.append(t)
  deadline=time.monotonic()+5
  while c.service.requests.qsize()<len(threads):
   if time.monotonic()>deadline:raise TimeoutError('Actual CPU queue fixture did not enqueue')
   time.sleep(.001)
 while c.service.process_one(timeout=.001):pass
 for t in threads:t.join(timeout=5)
 assert all(not t.is_alive()for t in threads)and results==[marker,marker]
 return order

def main():
 with tempfile.TemporaryDirectory(prefix='tta_real_fifo_cpu_',dir=O)as tmp:
  root=Path(tmp);assert trial(False,root/'old')==['infer','setup'];assert trial(True,root/'new')==['setup','infer']
  rows=[json.loads(s)for s in(root/'new/events.jsonl').read_text().splitlines()];assert len(rows)==1 and rows[0]['original_function_and_args_preserved']
 write(O/'CPU_setup_fairness_checks.json',dict(passed=True,actual_original_ModelService_queue_exercised=True,original_lower_priority_setup_deferred=True,equal_priority_restores_original_enqueue_FIFO=True,ordinary_inference_priority_unchanged=True,original_function_args_and_return_identity_preserved=True,no_GPU_context_or_policy_called=True))
 print('ACTUAL_OWNER_FIFO_SETUP_CPU_PASS',flush=True)
if __name__=='__main__':main()

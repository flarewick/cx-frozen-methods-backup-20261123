"""CPU actual owner/router/gate; original queue selection, RNG and release."""
import threading,time,random,pickle,tempfile
from concurrent.futures import Future
from pathlib import Path
import numpy as np,torch
from concurrent_rng import Lane
from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
from native_expert_request_priority_v1.fairness import install
from native_policy_execution_v1.common import T,write,sha
from .scheduler import NativeOverlap
O=T/'outputs/native_render_overlap_v4'

def trial(overlap,*,error=False,cancel=False,controls=False,unsafe_first=False):
 c=Coordinator();c.rng.install();outputs=[];events=[];threads=[];rendering=threading.Event();owner=threading.get_ident()
 def draw():return np.r_[torch.rand(7).numpy(),np.random.rand(7),[random.random()for _ in range(7)]]
 def encode():
  events.append(('encode',threading.get_ident(),rendering.is_set()));return draw()
 def sample():
  events.append(('sample',threading.get_ident(),rendering.is_set()));return draw()
 def seed():
  events.append(('seed',threading.get_ident(),rendering.is_set()));torch.manual_seed(17);return draw()
 def check():
  events.append(('check',threading.get_ident(),rendering.is_set()));return draw()
 def diagnostic_owner():
  events.append(('diagnostic_owner',threading.get_ident(),rendering.is_set()));return draw()
 def operation():
  events.append(('operation',threading.get_ident(),rendering.is_set()));return draw()
 def other_ordinary():
  events.append(('other_ordinary',threading.get_ident(),rendering.is_set()));return draw()
 class Native:
  def render_step(self,*args):
   rendering.set();time.sleep(.08)
   try:
    if error:raise ValueError('original render failure')
    if cancel:c.cancel.set()
    return 37
   finally:rendering.clear()
  take_pictures=render_step
 try:
  install(c,O/('fair_'+str(time.monotonic_ns())+'.jsonl'))
  functions=(seed,check,diagnostic_owner,operation,encode,sample)if controls else(encode,sample)
  bridge=NativeOverlap(c,torch,Native(),lambda:True,CPU_fixture_codes={fn.__code__ for fn in functions})
  if unsafe_first:functions=(other_ordinary,*functions)
  lanes=[Lane('case'+str(i)+('_sampler'if fn is sample else'_encoder'))for i,fn in enumerate(functions)];outer=Lane('physical')
  for lane in[*lanes,outer]:c.call(lane,lambda:torch.manual_seed(17))
  def submit(lane,fn):
   try:outputs.append((lane.name,c.call(lane,fn)))
   except BaseException as ex:outputs.append((lane.name,type(ex).__name__))
  for lane,fn in zip(lanes,functions):
   thread=threading.Thread(target=submit,args=(lane,fn));threads.append(thread);thread.start()
  deadline=time.monotonic()+3
  while c.service.requests.qsize()!=len(functions):assert time.monotonic()<deadline;time.sleep(.001)
  def observed():
   before=draw()
   with c.scene_gate.read(),bridge.active(overlap):result=bridge.render_step(None)
   after=draw()
   return before,result,after
  original_error=None
  try:value=c.call(outer,observed)
  except BaseException as ex:
   original_error=type(ex).__name__;value=None
   if controls:
    import traceback
    print('CONTROL_FIXTURE_EXCEPTION',traceback.format_exc(),flush=True)
  c.cancel.clear()
  while any(t.is_alive()for t in threads):assert time.monotonic()<deadline;c.service.process_one(.01)
  for thread in threads:thread.join(1);assert not thread.is_alive()
  bridge.close();assert bridge.closed and not c.scene_gate.readers and c.scene_gate.writer is None
  if overlap and not error and not cancel:
   from .audit import validate
   report=bridge.report();assert validate(report)
   bad=dict(report,worker_threads=[owner])
   try:validate(bad)
   except AssertionError:pass
   else:raise AssertionError('Owner also renderer accepted')
  assert all(tid==owner for _,tid,_ in events)
  assert not any(t.name.startswith('original-native-render')for t in threading.enumerate())
  states=[(l.name,l.numpy.get_state(),l.python.getstate(),l.torch_cpu.numpy())for l in[*lanes,outer]]
  return value,sorted(outputs,key=lambda x:x[0]),pickle.dumps(states),events,bridge.calls,bridge.pumped,original_error
 finally:c.rng.uninstall()

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();O.mkdir(parents=True,exist_ok=True)
 # Reset only CPU test initial state between independent software trials.
 def seeded(*args,**kwargs):torch.manual_seed(17);np.random.seed(17);random.seed(17);return trial(*args,**kwargs)
 a=seeded(False);b=seeded(True)
 assert a[0][1]==b[0][1]==37 and np.array_equal(a[0][0],b[0][0])and np.array_equal(a[0][2],b[0][2])
 assert all(x[0]==y[0]and np.array_equal(x[1],y[1])for x,y in zip(a[1],b[1]))and a[2]==b[2]
 assert not any(v[2]for v in a[3])and all(v[2]for v in b[3])and b[4]==1 and b[5]==2
 ca=seeded(False,controls=True);cb=seeded(True,controls=True)
 assert ca[-1]is None and cb[-1]is None
 assert ca[2]==cb[2]and all(x[0]==y[0]and np.array_equal(x[1],y[1])for x,y in zip(ca[1],cb[1]))
 assert np.array_equal(ca[0][0],cb[0][0])and np.array_equal(ca[0][2],cb[0][2])and cb[5]==6
 assert all(v[2]for v in cb[3])and {v[0]for v in cb[3]}=={'seed','check','diagnostic_owner','operation','encode','sample'}
 ua=seeded(False,controls=True,unsafe_first=True);ub=seeded(True,controls=True,unsafe_first=True)
 assert ua[-1]is None and ub[-1]is None and ua[2]==ub[2]
 assert all(x[0]==y[0]and np.array_equal(x[1],y[1])for x,y in zip(ua[1],ub[1]))
 assert np.array_equal(ua[0][0],ub[0][0])and np.array_equal(ua[0][2],ub[0][2])and ub[5]==6
 assert not next(v[2]for v in ub[3]if v[0]=='other_ordinary')and all(v[2]for v in ub[3]if v[0]!='other_ordinary')
 failed=seeded(True,error=True);assert failed[-1]=='ValueError'
 canceled=seeded(True,cancel=True);assert canceled[-1]in(None,'AssertionError')
 fairness=queue_checks()
 from model_service import ModelService
 from concurrent_rng import RngRouter
 from scene_gate import SceneGate
 import inspect
 sources={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()}
 for cls in(ModelService,RngRouter,SceneGate):
  p=Path(inspect.getsourcefile(cls));sources[str(p)]=sha(p)
 p=T/'native_expert_request_priority_v1/fairness.py';sources[str(p)]=sha(p)
 write(O/'CPU_contract_checks_v1.json',dict(passed=True,CPU_only=True,actual_original_ModelService_RngRouter_SceneGate=True,all_lane_arrays_and_final_RNG_byte_exact=True,actual_two_original_owner_requests_overlapped_CPU_native_fixture=True,actual_six_control_model_requests_and_final_RNG_match_original=True,unsafe_ordinary_first_bypassed_only_during_render_without_execution=True,all_seven_lanes_and_outer_RNG_and_arrays_match_original=True,render_failure_propagated_and_threads_released=True,cancellation_released_worker_and_scene_reads=True,actual_original_fairness_and_graph_guard_checks=fairness,actual_report_positive_and_wrong_owner_rejected=True,production_source_not_GPU_qualified=True,no_new_scientific_data=True,source_sha256=sources))
 print('RENDER_OVERLAP_CPU_ROUTER_GATE_RNG_RELEASE_PASS',flush=True)
def queue_checks():
 summaries=[]
 for kind,bound in(('setup',6),('original_expert_plan_batch',16)):
  trace=[];c=Coordinator();c.rng.install()
  def encode():trace.append('encode')
  def sample():trace.append('sample')
  def blocked():trace.append(kind)
  blocked._cx_stage=kind;blocked._cx_priority=5 if kind=='setup'else 8
  try:
   install(c,O/('queue_'+str(time.monotonic_ns())+'.jsonl'))
   bridge=NativeOverlap(c,torch,object(),lambda:False,CPU_fixture_codes={encode.__code__,sample.__code__})
   lane=Lane('query_encoder');other=Lane('scene')
   for number in range(bound+1):c.service.requests.put((0,number,lane,encode,(),{},Future()))
   c.service.requests.put((1 if kind=='setup'else 8,bound+1,other,blocked,(),{},Future()))
   bridge.pumping=True
   for _ in range(bound):assert c.service.process_one(0)
   assert trace==['encode']*bound and not c.service.process_one(0)and c.service.requests.qsize()==2
   bridge.pumping=False;assert c.service.process_one(0)and trace[-1]==kind
   assert c.service.process_one(0)and trace[-1]=='encode'
   # A legal sample closure waits until graph initialization completes outside rendering.
   item=(0,99,Lane('query_sampler'),sample,(),{},Future());bridge.pumping=True
   assert not bridge.allowed(item)
   bridge.graph_ready=lambda:True;assert bridge.allowed(item)
   def impostor():pass
   impostor.__name__='encode';assert not bridge.allowed((0,100,lane,impostor,(),{},Future()))
   bridge.pumping=False;bridge.close();summaries.append(dict(kind=kind,original_burst_bound=bound,unsafe_deferred_without_dequeue_or_counter_change=True))
  finally:c.rng.uninstall()
 return summaries

if __name__=='__main__':main()

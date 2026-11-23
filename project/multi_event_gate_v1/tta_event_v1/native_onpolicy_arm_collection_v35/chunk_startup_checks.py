"""Actual CPU threads: q0 is insufficient, original completed32 releases peers."""
from pathlib import Path
import threading
from types import SimpleNamespace
from .startup import StagedLanes,actor_class

def run():
 cancel=threading.Event();rows=[];gate=StagedLanes(cancel,rows.append,lanes=9)
 entered=[threading.Event()for _ in range(9)];admitted=[threading.Event()for _ in range(9)]
 def worker(n):
  entered[n].set()
  if gate.wait(n):admitted[n].set()
 threads=[threading.Thread(target=worker,args=(n,))for n in range(9)]
 for thread in threads:thread.start()
 try:
  assert all(e.wait(2)for e in entered)and all(e.wait(2)for e in admitted[:4])
  assert not admitted[4].wait(.05)and not any(e.is_set()for e in admitted[4:])
  calls=[];token=object();marked=[]
  class Original:
   def query(self,*args):calls.append(('query',args));return token
   def finish_chunk(self,*args):calls.append(('finish',args));return args[1]
  actor=actor_class(Original,gate,{'case0':0},SimpleNamespace(mark_case_ready=marked.append))()
  folder=Path('/CPU_fixture/case0/M_event_full')
  assert actor.query(None,dict(query=0,global_tick=0),folder)is token and marked==['case0']
  assert not gate.eight.is_set()and len(calls)==1
  partial=dict(query=0,actual_complete_native32=False,commands_returned=31,actor_observation_tick=0)
  assert actor.finish_chunk(None,partial,None,folder)is partial and not gate.eight.is_set()and len(calls)==2
  complete=dict(query=0,actual_complete_native32=True,commands_returned=32,actor_observation_tick=0)
  assert actor.finish_chunk(None,complete,None,folder)is complete and len(calls)==3
  assert all(e.wait(2)for e in admitted)and len(rows)==1 and rows[0]['reason']=='actual_complete_initial_native32'
 finally:
  cancel.set()
  for thread in threads:thread.join(2);assert not thread.is_alive()
 import numpy as np
 fallback=StagedLanes(threading.Event(),lambda row:None,lanes=9)
 initial=dict(native_action_count=0,rgb=[np.zeros((240,320,3),np.uint8)for _ in range(3)],command_state=np.zeros(14))
 assert fallback.release_q0(initial,'real-case','0'*64)is False and not fallback.eight.is_set()
 assert fallback.release('actual_case_terminal',lane=0)and fallback.eight.is_set()
 error=ValueError('original failure')
 class Failing:
  def finish_chunk(self,*args):raise error
 failed=StagedLanes(threading.Event(),lambda row:None,lanes=9)
 actor=actor_class(Failing,failed,{'case0':0},None)()
 try:actor.finish_chunk(None,complete,None,folder)
 except ValueError as actual:assert actual is error
 else:raise AssertionError('Original exception swallowed')
 assert not failed.eight.is_set()
 return dict(passed=True,actual_CPU_threads=9,only_first4_before_actual_complete32=True,
  query_and_passive_q0_cannot_release=True,partial31_cannot_release=True,
  original_query_finish_once_and_return_exception_identity=True,terminal_fallback=True,
  no_GPU_or_native_qualification=True)

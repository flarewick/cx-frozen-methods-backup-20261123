"""CPU-only boundary/order/reset/count contracts; not native policy validation."""
from pathlib import Path
import hashlib,json
from feedback_queue import FeedbackQueue,Ticket
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_phase_feedback_v1'
H='1'*64
def run():
 checks=0
 def reject(fn):
  nonlocal checks
  try:fn()
  except ValueError:checks+=1
  else:raise AssertionError('Invalid future/cross-source/update admitted')
 q=FeedbackQueue();q.arrive(100,0.,H);t=q.register(0)
 reject(lambda:q.window(t,row=0,start_tick=100,end_tick=125,kind='origin',command_start_tick=100,command_end_tick=125))
 q.arrive(125,.1,H);q.window(t,row=0,start_tick=100,end_tick=125,kind='origin',command_start_tick=100,command_end_tick=125)
 reject(lambda:q.at_query(query=1,tick=126,query_proof_sha256=H))
 reject(lambda:q.at_query(query=1,tick=125,query_proof_sha256=H))
 q.arrive(150,.2,H);q.window(t,row=1,start_tick=125,end_tick=150,kind='origin',command_start_tick=100,command_end_tick=150)
 q.close_origin(t,commands_returned=32,complete_native32=True,receipt_tick=150,phase95_tick=145,phase95_elapsed_s=.18)
 reject(lambda:q.at_query(query=2,tick=150,query_proof_sha256=H))
 assert q.at_query(query=1,tick=150,query_proof_sha256=H)is None;checks+=1
 # Before minlag and start-before-phase cannot be rescued merely by waiting.
 assert q.eligible(t)==();checks+=1
 q.arrive(225,.5,H)
 reject(lambda:q.window(t,row=2,start_tick=125,end_tick=225,kind='origin',command_start_tick=100,command_end_tick=225))
 reject(lambda:q.window(t,row=2,start_tick=125,end_tick=225,kind='following',command_start_tick=150,command_end_tick=225))
 reject(lambda:q.window(t,row=2,start_tick=150,end_tick=225,kind='following',command_start_tick=150,command_end_tick=226))
 q.window(t,row=2,start_tick=150,end_tick=225,kind='following',command_start_tick=150,command_end_tick=225)
 assert q.eligible(t)==(2,);checks+=1
 out=q.at_query(query=2,tick=225,query_proof_sha256=H);assert out.rows==(2,)and out.source_query==0 and out.query==2;checks+=1
 reject(lambda:q.at_query(query=2,tick=225,query_proof_sha256=H))
 q.arrive(250,.6,H);assert not q.window(t,row=3,start_tick=225,end_tick=250,kind='following',command_start_tick=150,command_end_tick=250);checks+=1
 assert out.rows==(2,)and q.released==1;checks+=1
 assert q.at_query(query=3,tick=250,query_proof_sha256=H)is None;checks+=1
 reject(lambda:q.arrive(225,.5,H))
 reject(lambda:q.arrive(250,.61,H))
 q.finish();reject(lambda:q.register(4));q.reset();q.arrive(100,0.,H);reject(lambda:q.event(t))
 # Source budgets include already-consumed tickets; no fourth source at episode end.
 for query in range(3):
  tt=q.register(query);q.arrive(150+query*150,(150+query*150-100)*.004,H)
  q.close_origin(tt,commands_returned=32,complete_native32=True,receipt_tick=q.now,phase95_tick=q.now-25,phase95_elapsed_s=q.elapsed-.1)
  start=q.now;q.arrive(start+100,q.elapsed+.4,H)
  q.window(tt,row=query,start_tick=start,end_tick=start+100,kind='following',command_start_tick=start,command_end_tick=start+100)
  packet=q.at_query(query=query+1,tick=q.now,query_proof_sha256=H);assert packet is not None;checks+=1
 reject(lambda:q.register(3));assert q.released==3;checks+=1
 # Censor cannot release, even with later RGB meeting the elapsed lag.
 q.reset();q.arrive(100,0.,H);t=q.register(0);q.arrive(150,.2,H)
 reject(lambda:q.close_origin(t,commands_returned=31,complete_native32=True,receipt_tick=150,phase95_tick=145,phase95_elapsed_s=.18))
 reject(lambda:q.close_origin(t,commands_returned=32,complete_native32=True,receipt_tick=150,phase95_tick=151,phase95_elapsed_s=.18))
 q.close_origin(t,commands_returned=22,complete_native32=False,receipt_tick=150,phase95_tick=None,phase95_elapsed_s=None)
 q.arrive(250,.6,H);q.window(t,row=0,start_tick=150,end_tick=250,kind='following',command_start_tick=150,command_end_tick=250)
 reject(lambda:q.at_query(query=1,tick=250,query_proof_sha256=H))
 value=dict(passed=True,checks=checks,CPU_fixture_only=True,real_GPU_or_third_query_NOT_verified=True,
  no_early_query_future_commands_or_duplicate_update=True,episode_reset_and_max3_verified=True,
  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in(C/'checks.py',C/'feedback_queue.py')},
  online_updates=0,TTA_utility_proven=False)
 O.mkdir(parents=True,exist_ok=True);p=O/'queue_checks_v2.json';s=json.dumps(value,indent=2)+'\n'
 if p.exists():assert p.read_text()==s
 else:p.write_text(s)
 print('PHASE_QUERY_BOUNDARY_CPU_CONTRACTS_PASS',checks,flush=True)
if __name__=='__main__':run()

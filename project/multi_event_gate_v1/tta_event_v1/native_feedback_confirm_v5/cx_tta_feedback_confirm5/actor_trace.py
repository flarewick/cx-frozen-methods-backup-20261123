"""Actual model query arguments before mutable transforms; no extra inference."""
from pathlib import Path
import hashlib,numpy as np
from .paths import read,write
from .contract import load_packet,BRANCHES
from cx_tta.eval_diagnostics import array_hash
from native_nextchunk_v1.sampler_fork import base_hash
CAMERAS=('cam_high','cam_right_wrist','cam_left_wrist')
class ActorTrace:
 def __init__(self,base,router):
  self.base,self.router=base,router;self.original=base.infer;self.rows={}
  def infer(obs,*args,**kwargs):
   lane=router.current()
   if lane is None or lane.name not in self.rows:raise ValueError('Model query outside active native case trace')
   row=dict(actor_images={str(i):array_hash(obs['images'][name].transpose(1,2,0))for i,name in enumerate(CAMERAS)},
     command_state_sha256=array_hash(obs['state']),
     instruction_sha256=hashlib.sha256(obs['prompt'].encode()).hexdigest())
   result=self.original(obs,*args,**kwargs)
   row['base_sha256']=base_hash(np.asarray(result['actions'],np.float32))
   self.rows[lane.name]['rows'].append(row)
   return result
  base.infer=infer
 def begin(self,lane,case_id):
  if lane.name in self.rows:raise ValueError('Previous case trace not closed')
  self.rows[lane.name]=dict(case_id=case_id,rows=[])
 def finish(self,lane,folder):
  value=self.rows.pop(lane.name);value.update(actual_infer_arguments=True,extra_inference_calls=0)
  write(Path(folder)/'actor_model_trace.json',value)
 def uninstall(self):self.base.infer=self.original
def verify_trace(folder,instruction):
 folder=Path(folder);case=read(folder/'case.json');trace=read(folder/'actor_model_trace.json')
 assert trace['case_id']==case['case_id']and trace['actual_infer_arguments']and trace['extra_inference_calls']==0
 if not(folder/'observed_history.npz').exists():
  assert read(folder/'unavailable.json')['reason']=='native_episode_terminated_before_event'and trace['rows']==[]
  return dict(passed=True,actual_model_queries=0,native_terminal_before_any_query=True)
 h=load_packet(folder/'observed_history.npz');meta=read(folder/'actor_history.json')
 text=hashlib.sha256(instruction.encode()).hexdigest()
 expected=[]
 def history():
  for q,row in enumerate(meta['queries']):
   expected.append(dict(actor_images=row['actor_images'],command_state_sha256=array_hash(h['states'][q]),
     instruction_sha256=text,base_sha256=base_hash(h['base_chunks'][q])))
 def following(where,branch):
  p=where/(branch+'_following_query.json')
  if p.exists():
   qm=read(p);origin=load_packet(where/(branch+'_origin.npz'))
   expected.append(dict(actor_images=qm['actor_images'],command_state_sha256=array_hash(origin['command_state'][-1]),
      instruction_sha256=text,base_sha256=qm['base_sha256']))
 def delayed(where,branch):
  p=where/(branch+'_delayed_query.json')
  if p.exists():
   qm=read(p);previous=load_packet(where/(branch+'_following.npz'))
   expected.append(dict(actor_images=qm['actor_images'],command_state_sha256=array_hash(previous['command_state'][-1]),instruction_sha256=text,base_sha256=qm['base_sha256']))
 history()
 if not(folder/'unavailable.json').exists():
  for branch in BRANCHES:
   following(folder,branch);delayed(folder,branch)
  if case.get('requires_sequential_reference'):
   for branch in('Z_reference','C_original'):
    history();following(folder/'sequential_reference',branch);delayed(folder/'sequential_reference',branch)
 assert trace['rows']==expected,'Actual model images/state/prompt/base or query count differ from actor metadata'
 return dict(passed=True,actual_model_queries=len(expected),same_actor_arguments_as_saved_current_observation=True,
   no_extra_base_draw=True,trace_scope='Raw arguments captured before mutable transforms; case identifiers only audit routing.')

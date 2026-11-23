"""Only actual arrived RGB/command/language/prior legal history enter policy sampling."""
import random
import numpy as np
import torch
from concurrent_rng import Lane
from native_nextchunk_v1.sampler_fork import audited_draw,full_hash
from native_nextchunk_v1.query_ledger import QueryLedger
from native_nextchunk_v1.query_current import query_current
from cx_tta_feedback_confirm5.views import view_inputs,batched_encode
from cx_tta.eval_diagnostics import array_hash
from .common import write,sha
class ActualQuery:
 def __init__(self,c,base,head,robots,projection,cache,*,name,instruction,native_diagnostic=None):
  self.native_diagnostic=native_diagnostic
  self.c,self.base,self.head,self.robots,self.projection,self.cache=c,base,head,robots,projection,cache
  self.encoder=Lane(name+'_encoder');self.sampler=Lane(name+'_sampler');self.previous=None;self.instruction=instruction;self.rows=[]
  def seed():random.seed(17);np.random.seed(17);torch.manual_seed(17)
  c.call(self.encoder,seed);c.call(self.sampler,seed)
 def query(self,observed,record,folder):
  q=record['query'];tick=record['global_tick'];assert q==len(self.rows)
  before=self.native_diagnostic()if self.native_diagnostic is not None else None
  images=observed['rgb'];state=observed['command_state']
  def encode():
   before=full_hash(self.c.rng.capture(self.encoder))
   value=self.cache.encode(lambda views,prompt:batched_encode(self.base,views,prompt),view_inputs(images,state,images,state),self.instruction)
   assert full_hash(self.c.rng.capture(self.encoder))==before
   return value
  joint,encoded=self.c.call(self.encoder,encode)
  current=dict(joint=joint,command_state=state.copy(),rgb=images,images={str(i):array_hash(im)for i,im in enumerate(images)},observation_tick=tick)
  def sample():
   captured={}
   def hook(name):
    def store(module,args,value):captured[name]=value.detach().cpu().numpy().copy()
    return store
   hooks=[self.head.action_head[1].register_forward_hook(hook('hidden')),self.head.action_head[2].register_forward_hook(hook('logits'))]
   ledger=QueryLedger('unexecuted_expert_state_comparator_'+str(q))
   try:
    result,after,witness=audited_draw(self.c.rng,self.sampler,self.base._model,ledger,
      lambda:query_current(self.base,self.head,self.robots,self.projection,current,self.instruction,self.previous,decision_tick=tick),query=0,tick=tick)
    assert ledger.awaiting_execution and len(ledger.records)==1
   finally:
    for h in hooks:h.remove()
   return result,witness,captured
  result,witness,captured=self.c.call(self.sampler,sample);self.previous=result['previous']
  assert result['actor_images']==current['images']and result['actor_observation_tick']==tick
  values=dict(base=result['base'],raw_delta=result['raw_delta'],hidden=captured['hidden'][0],logits=captured['logits'][0],joint=joint,
    **{'legal_'+k:v for k,v in result['legal'].items()})
  assert values['hidden'].shape==(32,64)and values['logits'].shape==(32,14)and all(np.isfinite(v).all()for v in values.values())
  file=folder/('policy%03d.npz'%q)
  with file.open('xb')as f:np.savez_compressed(f,**values)
  row=dict(query=q,actor_observation_tick=tick,observation_file=record['file'],observation_sha256=record['sha256'],
    actor_images=current['images'],command_state_sha256=array_hash(state),noise_witness=witness,encoder=encoded,
    policy_file=str(file),policy_sha256=sha(file),base_infer_calls=1,actual_internal_noise_calls=1,
    sampled_base_executed=False,prior_query_history_only=True,actual_online_actor_query_claimed=False)
  if self.native_diagnostic is not None:
   after=self.native_diagnostic();assert before==after,'Passive query changed native physics or RNG'
   write(folder/('query%03d_environment_audit.json'%q),dict(passed=True,**before,policy_call_changed_physics_or_RNG=False))
  write(folder/('policy%03d.json'%q),row);self.rows.append(row)
  return result,row,values

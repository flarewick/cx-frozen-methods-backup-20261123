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
class ActorQuery:
 def __init__(self,c,base,head,robots,projection,cache,*,name,instruction):
  self.c,self.base,self.head,self.robots,self.projection,self.cache=c,base,head,robots,projection,cache
  self.encoder=Lane(name+'_encoder');self.sampler=Lane(name+'_sampler');self.previous=None;self.instruction=instruction;self.rows=[];self.ledger=QueryLedger('native_actor_'+name);self.poisoned=False
  def seed():random.seed(17);np.random.seed(17);torch.manual_seed(17)
  c.call(self.encoder,seed);c.call(self.sampler,seed)
 def query(self,observed,record,folder):
  q=record['query'];tick=record['global_tick'];assert q==len(self.rows)
  if self.poisoned:raise ValueError('Failed actor query cannot resume')
  self.ledger.permit_query(q,tick)
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
   try:
    result,after,witness=audited_draw(self.c.rng,self.sampler,self.base._model,self.ledger,
      lambda:query_current(self.base,self.head,self.robots,self.projection,current,self.instruction,self.previous,decision_tick=tick),query=q,tick=tick)
    assert self.ledger.awaiting_execution and len(self.ledger.records)==q+1
   finally:
    for h in hooks:h.remove()
   return result,witness,captured
  try:result,witness,captured=self.c.call(self.sampler,sample)
  except BaseException:self.poisoned=True;raise
  self.previous=result['previous']
  assert result['actor_images']==current['images']and result['actor_observation_tick']==tick
  values=dict(base=result['base'],raw_delta=result['raw_delta'],hidden=captured['hidden'][0],logits=captured['logits'][0],joint=joint,
    **{'legal_'+k:v for k,v in result['legal'].items()})
  assert values['hidden'].shape==(32,64)and values['logits'].shape==(32,14)and all(np.isfinite(v).all()for v in values.values())
  file=folder/('policy%03d.npz'%q)
  with file.open('xb')as f:np.savez_compressed(f,**values)
  row=dict(query=q,actor_observation_tick=tick,observation_file=record['file'],observation_sha256=record['sha256'],
    actor_images=current['images'],command_state_sha256=array_hash(state),noise_witness=witness,encoder=encoded,
    policy_file=str(file),policy_sha256=sha(file),base_infer_calls=1,actual_internal_noise_calls=1,
    actual_policy_reference_query=True,prior_query_history_only=True,actual_actor_execution_record_required=True)
  write(folder/('policy%03d.json'%q),row);self.rows.append(row)
  return result,row,values

 def finish_chunk(self,actions,metadata,trace_file,folder):
  if self.poisoned or not self.ledger.awaiting_execution:raise ValueError('Exactly one pending actor draw required')
  q=len(self.rows)-1;start=self.rows[-1]['actor_observation_tick']
  actions=np.asarray(actions)
  if actions.shape!=(32,14)or actions.dtype!=np.float32 or not np.isfinite(actions).all():raise ValueError('Actual finite32 command chunk required')
  with np.load(trace_file,allow_pickle=False)as z:
   if not np.array_equal(z['source_commands'],actions)or len(z['dt_s'])!=metadata['committed_ticks']:raise ValueError('Execution trace differs from composed actor actions')
  receipts=metadata['receipts']
  if [r['offset']for r in receipts]!=list(range(len(receipts)))or not 1<=len(receipts)<=32:raise ValueError('Actual native command receipts required')
  returned=sum(r['status']=='returned'for r in receipts)
  complete=bool(len(receipts)==32 and returned==32 and metadata['reason']=='source_chunk_returned')
  if bool(metadata['complete_source_chunk'])!=complete:raise ValueError('Contradictory native32 completion')
  end=start+metadata['committed_ticks'];self.ledger.finish_chunk(returned,end_tick=end)
  from cx_tta.eval_diagnostics import array_hash
  proof=dict(query=q,actor_observation_tick=start,end_tick=end,commands_returned=returned,actual_complete_native32=complete,execution_reason=metadata['reason'],executed_actions_sha256=array_hash(actions),actual_trace_file=str(trace_file),actual_trace_sha256=sha(trace_file),once_base_witness=self.rows[-1]['noise_witness'],no_new_query_after_censor=self.ledger.censored)
  write(folder/('query%03d_execution.json'%q),proof)
  return proof

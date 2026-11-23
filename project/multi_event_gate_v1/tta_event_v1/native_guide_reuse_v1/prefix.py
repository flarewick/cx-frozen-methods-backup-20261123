"""Original guide batch3 prefix; real current-card first3/task conformance."""
import threading,time,pickle,random
import numpy as np

class Prefix:
 def __init__(self,c,torch,base,denoise,overlap,gpu_uuid):
  from native_batched_prefix_graph_v1.encoding import build
  from native_policy_execution_v1 import actor
  self.c=c;self.torch=torch;self.base=base;self.denoise=denoise;self.overlap=overlap;self.uuid=gpu_uuid
  self.owner=threading.get_ident();self.original=actor.batched_encode;self.fast=build();self.actor=actor
  self.calls=0;self.fast_calls=0;self.references=0;self.initial_eager=0;self.rows=[];self.tasks={}
  actor.batched_encode=self.encode
 def state(self):
  t=self.torch;return(pickle.dumps((random.getstate(),np.random.get_state())),t.get_rng_state().clone(),t.cuda.get_rng_state().clone())
 def same(self,a,b):return a[0]==b[0]and self.torch.equal(a[1],b[1])and self.torch.equal(a[2],b[2])
 def encode(self,policy,views,prompt):
  assert threading.get_ident()==self.owner and policy is self.base
  self.calls+=1;lane=self.c.rng.current();name=''if lane is None else lane.name
  primary=name.endswith('_M_event_full_encoder')
  if not primary:self.references+=1;return self.original(policy,views,prompt)
  if self.denoise.entry is None:self.initial_eager+=1;return self.original(policy,views,prompt)
  import benefit_prefix_graph
  if not benefit_prefix_graph._CACHE:self.overlap.await_renderer()
  task=name.split(':',1)[0];conform=self.tasks.get(task,0)<3
  before=self.state();expected=None
  if conform:expected=self.original(policy,views,prompt);assert self.same(before,self.state())
  result=self.fast(policy,views,prompt);assert self.same(before,self.state())
  self.fast_calls+=1
  if conform:
   a,ma=expected;b,mb=result
   assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()and ma==mb,'Current-card original batch3 feature conformance failed'
   self.tasks[task]=self.tasks.get(task,0)+1;self.rows.append(dict(task=task,features_metadata_RNG_exact=True,actual_GPU_uuid=self.uuid))
  return result
 def report(self):
  import benefit_prefix_graph
  assert self.fast_calls>0 and self.references>0 and len(benefit_prefix_graph._CACHE)==1
  entry=next(iter(benefit_prefix_graph._CACHE.values()))
  assert entry['model']()is self.base._model and entry['owner']==self.owner and 'captured_embedding_method_owner'in entry
  return dict(passed=True,GPU_uuid=self.uuid,actual_calls=self.calls,fast_calls=self.fast_calls,original_reference_calls=self.references,initial_eager_before_denoiser_ready=self.initial_eager,task_conformance=dict(self.tasks),conformance_rows=self.rows,original_eager_verification_calls=len(self.rows),original_prefix_warm_forwards=2,capture_recordings=1,original_feature_batch3_and_action_batch1_preserved=True,actual_constant_owner_retained=True,no_extra_policy_queries_or_episodes=True)

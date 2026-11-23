"""Persistent SAM2 object identity; physical-time metadata stays separate from query order."""
from pathlib import Path
import importlib.util,hashlib,sys
import numpy as np,torch
T=Path(__file__).resolve().parents[1]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
video=module('query_object_frozen_SAM2_adapter',T/'video_instances_v1/adapter.py')
clock=module('query_object_original_query_clock',T/'query_identity_v1/stream.py')

def snapshot(model):
 cuda=next(model.parameters()).device
 return torch.random.get_rng_state().clone(),torch.cuda.get_rng_state(cuda).clone() if cuda.type=='cuda' else None

def preserved(model,before):
 cpu,cuda=before;assert torch.equal(cpu,torch.random.get_rng_state())
 if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state(next(model.parameters()).device))

class MaskCore:
 def __init__(self,model):self.model=model;self.reset()
 def reset(self):self.stream=None;self.frames=[]
 def initialize(self,rgb,masks,*,tick,decision_tick):
  if tick!=0 or decision_tick!=tick or self.frames:raise ValueError('First actually observed image only')
  before=snapshot(self.model);self.stream=video.Stream(self.model,rgb,masks);self.frames=self.stream.frames;logits=self.stream.initial_logits;score=np.array([float(self.stream.state['output_dict_per_obj'][obj]['cond_frame_outputs'][0]['object_score_logits'].sigmoid().cpu().item()) for obj in range(2)]);preserved(self.model,before)
  return dict(tick=0,decision_tick=0,masks=(logits[:,0]>0),presence=score,raw_logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest())
 def append(self,rgb,*,tick,decision_tick):
  before=snapshot(self.model);logits,score=self.stream.append(rgb,tick=tick,decision_tick=decision_tick);preserved(self.model,before)
  return dict(tick=tick,decision_tick=decision_tick,masks=logits>0,presence=score,raw_logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest())

class QueryMaskStream(clock.QueryPointStream):
 def __init__(self,model):
  self.core=MaskCore(model);self.query_readouts=[];self.native_readouts=[];self.last_physics=None;self.next_query=0;self.physics_started=False

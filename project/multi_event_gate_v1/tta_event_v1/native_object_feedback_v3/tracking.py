"""Actual one/two-ID SAM2 stream; absent compatibility slot never enters SAM2."""
import hashlib
import numpy as np
from query_object_v1.stream import video,snapshot,preserved
class RealObjectStream:
 def __init__(self,model,rgb,masks):
  import sam2.sam2_video_predictor as official
  self.count=len(masks)
  if self.count not in(1,2)or masks.dtype!=bool or masks.shape[1:]!=rgb.shape[:2]or not masks.reshape(self.count,-1).any(1).all():raise ValueError('One/two nonempty actually observed object masks')
  official.load_video_frames=video.arrived_loader
  self.model=model;self.frames=video.ArrivedFrames(rgb,model.image_size);self.state=model.init_state(self.frames,offload_video_to_cpu=True,offload_state_to_cpu=False);self.index=0
  for obj,mask in enumerate(masks):model.add_new_mask(self.state,frame_idx=0,obj_id=obj,mask=mask.copy())
  rows=list(model.propagate_in_video(self.state,start_frame_idx=0,max_frame_num_to_track=0));assert len(rows)==1 and rows[0][0]==0 and rows[0][1]==list(range(self.count));self.initial_logits=rows[0][2].detach().cpu().numpy().copy()
 def append(self,rgb,*,tick,decision_tick):
  self.frames.append(rgb,tick=tick,decision_tick=decision_tick);self.index+=1;self.state['num_frames']=len(self.frames)
  rows=list(self.model.propagate_in_video(self.state,start_frame_idx=self.index,max_frame_num_to_track=0,reverse=False));assert len(rows)==1 and rows[0][0]==self.index and rows[0][1]==list(range(self.count))
  logits=rows[0][2].detach().cpu().numpy()[:,0].copy();scores=[float(self.state['output_dict_per_obj'][obj]['non_cond_frame_outputs'][self.index]['object_score_logits'].sigmoid().cpu().item())for obj in range(self.count)]
  return logits,np.asarray(scores)

class MaskCore:
 def __init__(self,model):self.model=model;self.stream=None;self.frames=[];self.count=None
 def pack(self,logits,presence):
  if logits.shape[0]!=self.count or np.shape(presence)!=(self.count,):raise ValueError('Actual SAM2 object count changed')
  masks=np.zeros((2,*logits.shape[-2:]),bool);scores=np.zeros(2);masks[:self.count]=logits>0;scores[:self.count]=presence
  return dict(masks=masks,presence=scores,raw_logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest())
 def initialize(self,rgb,masks,*,tick,decision_tick):
  if tick!=0 or decision_tick!=tick or self.stream is not None:raise ValueError('Actual earliest RGB only')
  if masks.dtype!=bool or masks.shape!=(2,*rgb.shape[:2])or not masks[0].any():raise ValueError('Real first object and explicit absent-or-real second slot')
  self.count=2 if masks[1].any()else 1;before=snapshot(self.model);self.stream=RealObjectStream(self.model,rgb,masks[:self.count]);self.frames=self.stream.frames
  score=np.asarray([float(self.stream.state['output_dict_per_obj'][obj]['cond_frame_outputs'][0]['object_score_logits'].sigmoid().cpu().item())for obj in range(self.count)]);preserved(self.model,before)
  return dict(tick=0,decision_tick=0,**self.pack(self.stream.initial_logits[:,0],score))
 def append(self,rgb,*,tick,decision_tick):
  before=snapshot(self.model);logits,scores=self.stream.append(rgb,tick=tick,decision_tick=decision_tick);preserved(self.model,before)
  return dict(tick=tick,decision_tick=decision_tick,**self.pack(logits,scores))

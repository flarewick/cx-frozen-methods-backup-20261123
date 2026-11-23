"""Frozen SAM2 core with real physical-time provenance; only one arrived RGB enters."""
import hashlib
import numpy as np
class NativeMaskStream:
 def __init__(self,model=None,*,core=None):
  if core is None:
   from query_object_v1.stream import MaskCore
   core=MaskCore(model)
  self.core=core;self.rows=[];self.origin_index=None
 @staticmethod
 def image(rgb):
  if not isinstance(rgb,np.ndarray)or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('Actual uint8 camera RGB required')
  return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
 def _save(self,result,rgb,tick,elapsed_s,kind):
  masks=np.asarray(result['masks']);presence=np.asarray(result['presence'])
  if masks.dtype!=bool or masks.shape!=(2,*rgb.shape[:2])or presence.shape!=(2,)or not np.isfinite(presence).all():raise ValueError('Malformed fixed two-object readout')
  value=dict(global_tick=tick,elapsed_episode_s=float(elapsed_s),kind=kind,
    RGB_sha256=self.image(rgb),raw_logits_sha256=result['raw_logits_sha256'],
    masks=masks.copy(),presence=presence.copy(),semantic_GT_certified=False)
  for k in('masks','presence'):value[k].setflags(write=False)
  self.rows.append(value);return value
 def initialize(self,rgb,masks,*,global_tick=0,elapsed_s=0.,decision_tick=0):
  if self.rows or type(global_tick)is not int or global_tick!=0 or type(decision_tick)is not int or decision_tick!=0 or type(elapsed_s)is bool or elapsed_s!=0:raise ValueError('Earliest actually observed query0 only')
  self.image(rgb)
  if np.asarray(masks).dtype!=bool or np.asarray(masks).shape!=(2,*rgb.shape[:2]):raise ValueError('Exact earliest two-object seed masks required')
  result=self.core.initialize(rgb,masks.copy(),tick=0,decision_tick=0)
  return self._save(result,rgb,0,0.,'query')
 def append(self,rgb,*,global_tick,elapsed_s,decision_tick,kind):
  if not self.rows:raise ValueError('Actual earliest initialization required')
  if(kind not in('query','origin','following')or type(global_tick)is not int or type(decision_tick)is not int
     or decision_tick!=global_tick or global_tick<=self.rows[-1]['global_tick']
     or type(elapsed_s)is bool or not np.isfinite(elapsed_s)or elapsed_s<=self.rows[-1]['elapsed_episode_s']):raise ValueError('Strictly newer arrived physical-time observation required')
  self.image(rgb)
  if rgb.shape[:2]!=self.rows[0]['masks'].shape[1:]:raise ValueError('Camera shape changed')
  result=self.core.append(rgb,tick=global_tick,decision_tick=decision_tick)
  return self._save(result,rgb,global_tick,elapsed_s,kind)
 def mark_selected_origin(self,global_tick):
  if type(global_tick)is not int or self.origin_index is not None or not self.rows or global_tick!=self.rows[-1]['global_tick']or self.rows[-1]['kind']!='query':raise ValueError('Selected origin must be latest actual query, marked once')
  self.origin_index=len(self.rows)-1
  return self.rows[self.origin_index]

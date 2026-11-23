"""Episode-shared queue, source-local mask histories and immediate owner retirement."""
import gc,hashlib,weakref
import numpy as np
from native_phase_feedback_v1.feedback_queue import FeedbackQueue

def rgb_sha(rgb):return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()

class SourcePool:
 def __init__(self,source_factory,*,queue=None,retire=None):
  self.queue=FeedbackQueue()if queue is None else queue;self.factory=source_factory;self.retire=retire
  self.history=[];self.live={};self.finished=[];self.closed=False
 def arrive_query(self,current):
  if self.closed or current['query']!=len(self.history):raise ValueError('One contiguous actual episode query required')
  self.queue.arrive(current['tick'],current['episode_elapsed_s'],rgb_sha(current['rgb']))
  for source in self.live.values():source.boundary(current)
  self.history.append({k:current[k].copy()if isinstance(current[k],np.ndarray)else current[k]for k in ('query','tick','episode_elapsed_s','rgb','command_state')})
 def admit(self):
  if self.closed or not self.history:raise ValueError('Actual current query required before registration')
  q=self.history[-1]['query']
  if self.queue.last_query!=q:raise ValueError('Current query feedback must be consumed before new source admission')
  if q in self.live:raise ValueError('Duplicate source')
  ticket=self.queue.register(q);source=self.factory(self.history,self.queue,ticket)
  if source.queue is not self.queue or source.ticket!=ticket or source.source_query!=q:raise ValueError('Source escaped current shared queue')
  self.live[q]=source;return ticket
 def rows(self):return {q:{i:r for i,r in enumerate(s.rows)}for q,s in self.live.items()}
 def capture_plan(self,query,start_tick):
  if query!=len(self.history)-1:raise ValueError('Actual issued current query only')
  active=[s for q,s in self.live.items()if query in(q,q+1)]
  if len(active)>1:raise ValueError('Frozen cooldown must keep primary source windows disjoint')
  if not active:return dict(collect_dense=False,origin_source_query=None,on_arrival=None)
  s=active[0];kind='origin'if query==s.source_query else'following'
  s.capture_callback(kind,start_tick,query);source_ref=weakref.ref(s)
  def arrived(*args,**kwargs):
   owner=source_ref()
   if owner is None or owner.closed:raise ValueError('Retired source callback cannot consume new frames')
   return owner.arrived(*args,**kwargs)
  return dict(collect_dense=True,origin_source_query=s.source_query if kind=='origin'else None,on_arrival=arrived)
 def close_origin(self,query,trace,metadata):
  if query in self.live:return self.live[query].close_origin(trace,metadata)
  return None
 def retire_after_query(self,query,*,force=False):
  for key in list(self.live):
   source=self.live[key];event=self.queue.event(source.ticket)
   if not(force or event['used']or query>=key+2):continue
   summary=source.finish();summary.update(retired_at_query=query,feedback_consumed=event['used'],forced_episode_end=force,rows=source.rows,mask_readouts=[]if source.owner.memory is None else source.owner.memory.rows)
   self.finished.append(summary);box=[source];refs=[weakref.ref(source),weakref.ref(source.owner)]
   del self.live[key];del source
   if self.retire is not None:self.retire(box,refs)
   else:
    box.clear();gc.collect()
    if any(r()is not None for r in refs):raise ValueError('Completed source ownership remains live')
 def finish(self):
  if self.closed:raise ValueError('Episode pool already closed')
  self.retire_after_query(len(self.history)-1,force=True);self.closed=True
  return dict(completed_sources=self.finished,remaining_live_sources=len(self.live),source_mask_history='query-only prefix per event, dense origin/following until first feedback release or q2 limit')

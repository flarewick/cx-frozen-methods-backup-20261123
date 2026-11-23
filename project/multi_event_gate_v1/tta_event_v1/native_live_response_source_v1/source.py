"""One source-local arrived-mask state and physical local-response queue wiring."""
import numpy as np
from native_delayed_query_v1.mask_owner import ArrivedMaskOwner
from native_delayed_query_v1.local_feedback import LocalFeedbackStream
from native_phase_feedback_v1.feedback_queue import FeedbackQueue

class SourceStream:
 def __init__(self,model,seed_masks,history,instruction,*,robots,cfg,camera,call,queue=None,ticket=None,owner_factory=None):
  if not history or [h['query']for h in history]!=list(range(len(history))):raise ValueError('Actual contiguous query0..source history required')
  if history[0]['tick']!=0:raise ValueError('Actual earliest query must have tick0')
  self.source_query=history[-1]['query'];self.origin_tick=history[-1]['tick'];self.call=call
  self.owner=ArrivedMaskOwner(model,seed_masks)if owner_factory is None else owner_factory(model,seed_masks)
  self.rows=[];self.missing=[];self.boundaries=[];self.closed=False;self.segment=None;self.last_s=0.;self.previous_tick=self.origin_tick
  for h in history:
   current=self.call(self.owner.at_query,h['rgb'],query=h['query'],tick=h['tick'],elapsed_s=h['episode_elapsed_s'])
  self.owner.mark_origin();self.latest=current
  self.owns_queue=queue is None;self.queue=FeedbackQueue()if queue is None else queue
  if self.owns_queue:
   if ticket is not None:raise ValueError('Ticket without its owning queue')
   self.queue.arrive(self.origin_tick,0.,current['RGB_sha256']);self.ticket=self.queue.register(self.source_query)
  else:
   if ticket is None:raise ValueError('Shared queue requires actual registered source ticket')
   self.ticket=ticket
  event=self.queue.event(self.ticket)
  if event['source_query']!=self.source_query or event['start_tick']!=self.origin_tick:raise ValueError('Source/ticket did not start at actual current query')
  self.queue_start_s=event['start_s'];self.origin_episode_s=history[-1]['episode_elapsed_s']
  self.local=None
  if seed_masks is not None:
   self.local=LocalFeedbackStream(history[0]['rgb'],seed_masks,instruction,robots=robots,cfg=cfg,camera=camera)
   self.local.start(history[-1]['rgb'],current['masks'],current['presence'],history[-1]['command_state'],tick=self.origin_tick,elapsed_s=0.)
 def capture_callback(self,kind,start_tick,query):
  if self.closed or kind not in ('origin','following')or start_tick!=self.previous_tick:raise ValueError('Contiguous actual native segment required')
  if query!=self.source_query+(kind=='following'):raise ValueError('Borrowed issuing query segment')
  self.segment=dict(kind=kind,start_tick=start_tick,query=query,start_s=self.last_s)
  return self.arrived
 def arrived(self,tick,elapsed_s,observed,recorder):
  if self.closed or self.segment is None:raise ValueError('No current actual source segment')
  segment=self.segment;global_tick=segment['start_tick']+tick;relative_s=segment['start_s']+elapsed_s
  rgb=observed['rgb'][0]
  current=self.call(self.owner.after_native,rgb,tick=global_tick,elapsed_s=self.origin_episode_s+relative_s,kind=segment['kind'])
  self.queue.arrive(global_tick,self.queue_start_s+relative_s,current['RGB_sha256'])
  if self.local is not None:
   first=self.previous_tick-segment['start_tick'];last=tick
   window={k:np.stack([r[k]for r in recorder.rows[first:last]])for k in ('arm_position_target','arm_velocity_target','gripper_logical_command')}
   window['dt_s']=np.asarray(recorder.dt[first:last],np.float64)
   new,missing=self.local.arrive(rgb,current['masks'],current['presence'],window,observed['command_state'],tick=global_tick,elapsed_s=relative_s,segment=segment['kind'],segment_start_tick=segment['start_tick'])
   for row in new:
    row['command_query']=segment['query'];i=len(self.rows);self.rows.append(row)
    self.queue.window(self.ticket,row=i,start_tick=row['start_tick'],end_tick=row['end_tick'],kind=segment['kind'],command_start_tick=segment['start_tick'],command_end_tick=global_tick)
   self.missing.extend(missing)
  self.previous_tick=global_tick;self.last_s=relative_s;self.latest=current
 def close_origin(self,trace,metadata):
  if self.segment is None or self.segment['kind']!='origin':raise ValueError('Actual origin only')
  if metadata['committed_ticks']==0:
   return dict(complete=False,reason='no_committed_native_steps_retained')
  completion=metadata['completion'];phase=completion[0].get('requested95_tick')if completion else None
  returned=sum(r['status']=='returned'for r in metadata['receipts'])
  phase_s=None if phase is None else self.queue_start_s+float(np.sum(trace['dt_s'][:phase]))
  self.queue.close_origin(self.ticket,commands_returned=returned,complete_native32=metadata['complete_source_chunk'],receipt_tick=self.previous_tick,phase95_tick=None if phase is None else self.origin_tick+phase,phase95_elapsed_s=phase_s)
  return dict(complete=metadata['complete_source_chunk'],requested95_tick=phase,commands_returned=returned)
 def boundary(self,current):
  if self.closed or current['query']not in(self.source_query+1,self.source_query+2)or current['tick']!=self.previous_tick:raise ValueError('Actual source q1/q2 arrived boundary required')
  before=self.owner.actual_tracking_calls
  row=self.call(self.owner.at_query,current['rgb'],query=current['query'],tick=current['tick'],elapsed_s=self.origin_episode_s+self.last_s)
  if self.owner.actual_tracking_calls!=before:raise ValueError('Native endpoint must be reused at actual query')
  self.queue.arrive(current['tick'],self.queue_start_s+self.last_s,row['RGB_sha256']);self.latest=row
  value=dict(query=current['query'],tick=current['tick'],eligible_rows=list(self.queue.eligible(self.ticket)))
  self.boundaries.append(value);return value
 def finish(self):
  if self.closed:raise ValueError('Source already closed')
  self.closed=True;local=None if self.local is None else self.local.finish();owner=self.owner.finish()
  return dict(source_query=self.source_query,local_rows=len(self.rows),missing=self.missing,local=local,owner=owner,boundaries=self.boundaries,queue=self.queue.finish()if self.owns_queue else None,shared_queue_ownership_preserved=not self.owns_queue)

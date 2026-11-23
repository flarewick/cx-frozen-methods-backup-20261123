"""Joint gripper phase is the latest actual milestone of all requested sides."""
import copy
from stage_feedback_v3.completion import CompletionQueue as Previous,KINDS
class CompletionQueue(Previous):
 def register(self,request_id,origin,kind,*args,**kwargs):
  self.joint=None
  if isinstance(kind,str):return super().register(request_id,origin,kind,*args,**kwargs)
  if not isinstance(kind,(tuple,list))or len(kind)!=2 or len(set(kind))!=2 or not set(kind)<=set(KINDS):raise ValueError('Explicit two actual gripper kinds required')
  if len({KINDS[k][0]for k in kind})!=2:raise ValueError('Contradictory same-side kinds are not joint feedback')
  self.joint=dict(request_id=request_id,origin=origin,kinds=list(kind))
  for i,k in enumerate(kind):super().register(request_id+':'+str(i),origin,k,*args,**kwargs)
 def observe_rgb(self,*args,**kwargs):
  values=super().observe_rgb(*args,**kwargs)
  if self.joint is None:return values
  components=copy.deepcopy(self.closed);assert len(components)==2
  ready=all(r['status']=='feedback_ready'and r['requested95_tick']is not None for r in components)
  value=dict(self.joint,status='feedback_ready'if ready else'censored',reason='all_actual_requested95_milestones'if ready else'one_or_more_actual_sides_not_completed',requested95_tick=max(r['requested95_tick']for r in components)if ready else None,commands_returned=32 if ready else min(r.get('commands_returned',32)for r in components),components=components,all_sides_required=True,physical_contact_certified=False)
  self.closed=[value];return [value]if ready else[]

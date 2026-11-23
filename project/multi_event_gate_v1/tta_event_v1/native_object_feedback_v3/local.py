"""Local visible points on current real masks, including achromatic surfaces."""
import numpy as np
from native_best100_projection_guard_v1.local import LocalFeedbackStream as Previous,PROJECTION_ERROR
from native_delayed_query_v1.primitives import old
FOCUS_RADIUS_PIXELS=48.
class LocalFeedbackStream(Previous):
 def __init__(self,*args,real_object_slots,**kwargs):
  slots=np.asarray(real_object_slots)
  if slots.dtype!=bool or slots.shape!=(2,)or not slots[0]:raise ValueError('Real observed seed-object slots required')
  self.real_object_slots=slots.copy()
  super().__init__(*args,**kwargs)
 def start(self,rgb,masks,presence,command_state,**kwargs):
  self._frame_command=np.asarray(command_state,float).copy()
  return super().start(rgb,masks,presence,command_state,**kwargs)
 def arrive(self,rgb,masks,presence,trace_window,command_state,**kwargs):
  self._frame_command=np.asarray(command_state,float).copy()
  return super().arrive(rgb,masks,presence,trace_window,command_state,**kwargs)
 def frame(self,rgb,masks,presence,*,tick,elapsed_s):
  if type(tick)is not int or tick<0 or not np.isfinite(elapsed_s)or rgb.dtype!=np.uint8 or rgb.shape!=(240,320,3):raise ValueError('Actual physical RGB/tick required')
  if np.asarray(masks).dtype!=bool or np.shape(masks)!=(2,240,320)or np.shape(presence)!=(2,)or not np.isfinite(presence).all():raise ValueError('Two compatibility slots with explicit observed presence')
  refined=np.asarray(masks).copy();area=refined.reshape(2,-1).sum(1);valid=(np.asarray(presence)>=.7)&(area>=32)
  if not self.real_object_slots[1]:refined[1]=False;valid[1]=False
  inter=int((refined[0]&refined[1]).sum());union=int((refined[0]|refined[1]).sum())
  if inter/max(union,1)>.2 or inter/max(int(area.min()),1)>.35:valid[:]=False
  focus=np.zeros_like(refined);reason=None;command=getattr(self,'_frame_command',None)
  if command is None or command.shape!=(14,)or not np.isfinite(command).all():raise ValueError('Current own commanded state required for local point support')
  try:tcp=old.tcp(command[None],self.robots,self.cfg,self.camera)[0]
  except ValueError as exc:
   if str(exc)!=PROJECTION_ERROR:raise
   tcp=None;reason='commanded_TCP_projection_unsupported'
  if tcp is not None:
   yy,xx=np.mgrid[:240,:320];distance=np.min((xx[None]-tcp[:,0,None,None])**2+(yy[None]-tcp[:,1,None,None])**2,axis=0)
   for obj in range(2):focus[obj]=refined[obj]&(distance<=FOCUS_RADIUS_PIXELS**2)
  valid&=focus.reshape(2,-1).sum(1)>=32
  points,good,colors=old.flow.corners(rgb,focus,valid)
  value=dict(rgb=rgb.copy(),mask=focus,valid=valid,p=points,good=good,colors=colors,tick=tick,elapsed_s=float(elapsed_s),point_support_interface='native_object_feedback_v3',real_object_slots=self.real_object_slots.copy(),focus_radius_pixels=FOCUS_RADIUS_PIXELS,focus_missing_reason=reason)
  if getattr(self,'_capture_arrival',False):self._arrived_frame=value
  return value

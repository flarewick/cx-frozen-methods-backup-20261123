"""Coalesce only the existing Torch graph-entry GC in the first real probe.

Reuse the already verified original ConstructorGC; collect before/after the
probe and retain every original non-graph collection, call, error and RNG phase.
"""
import gc,threading
_INSTALLED=False
TARGET='original_zero_L0_steady_probe'

def execute(c,original,lane,fn,args,kwargs):
 if getattr(fn,'_cx_stage',None)!=TARGET or getattr(c,'_sampler_startup_gc_done_v11',False):return original(lane,fn,args,kwargs)
 from collection_gc_schedule import ConstructorGC
 schedule=getattr(gc.collect,'__self__',None)
 if not isinstance(schedule,ConstructorGC)or schedule.depth or threading.get_ident()!=schedule.owner:raise RuntimeError('Expected original idle owner GC scheduler')
 note=schedule.note
 def sampler_note(stage,**fields):
  if stage!='planner_gc_schedule':raise RuntimeError('Unexpected GC context note')
  c.note('sampler_startup_gc_coalesced_v11',**fields)
 schedule.note=sampler_note
 try:
  with schedule.constructor():result=original(lane,fn,args,kwargs)
 finally:schedule.note=note
 c._sampler_startup_gc_done_v11=True
 return result

def install():
 global _INSTALLED
 if _INSTALLED:return
 from model_service import ModelService
 original=ModelService.execute
 def scheduled(self,lane,fn,args,kwargs):return execute(self.coordinator,lambda l,f,a,k:original(self,l,f,a,k),lane,fn,args,kwargs)
 ModelService.execute=scheduled;_INSTALLED=True

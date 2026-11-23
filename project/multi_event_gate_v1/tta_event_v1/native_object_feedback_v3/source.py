"""Preserve original source/queue/phase ownership; specialize only real-mask support."""
import numpy as np
from native_live_response_source_v1.source import SourceStream as Original
from native_delayed_query_v1.mask_owner import ArrivedMaskOwner
from .tracking import MaskCore
from .local import LocalFeedbackStream
class SourceStream(Original):
 def __init__(self,model,seed_masks,history,instruction,*,robots,cfg,camera,call,queue=None,ticket=None):
  super().__init__(model,seed_masks,history,instruction,robots=robots,cfg=cfg,camera=camera,call=call,queue=queue,ticket=ticket,owner_factory=lambda m,masks:ArrivedMaskOwner(m,masks,core=MaskCore(m)))
  if self.local is not None:
   # Rebuild the pure-CPU initial frame before any native command window. Reusing
   # the previous class's already-started frame would retain its color filter.
   self.local=LocalFeedbackStream(history[0]['rgb'],seed_masks,instruction,robots=robots,cfg=cfg,camera=camera,real_object_slots=np.asarray([True,bool(seed_masks[1].any())]))
   self.local.start(history[-1]['rgb'],self.latest['masks'],self.latest['presence'],history[-1]['command_state'],tick=self.origin_tick,elapsed_s=0.)

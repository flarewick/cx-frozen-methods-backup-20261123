from .common import W,C
import sys
sys.path.insert(0,str(C/'scripts'));sys.path.insert(0,str(W/'scripts'))
from compose import FrozenV1Composer,ARM
from cx_tta.multi_event import EventConfig
class EventComposer:
 def __init__(self,config):
  assert vars(config)==vars(EventConfig())
  self.inner=FrozenV1Composer('GT',W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4/fixed_ensemble_TRAIN472_v118/diagnostic_gate.json')
 def step(self,base,reference,legal,*,adapted_ARM=None):return self.inner.step(base,reference,legal,adapted_ARM=adapted_ARM)

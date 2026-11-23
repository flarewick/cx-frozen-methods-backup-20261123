"""Keep frozen event selection and replace only the selected ARM direction."""
import numpy as np
from cx_tta.multi_event import EventDetector,LocalController,apply_arm_candidate
from cx_tta.empirical_onset_gate import candidate_packet
ARM=np.array([0,1,2,3,4,5,7,8,9,10,11,12],np.int64)

class EventComposer:
 def __init__(self,config):
  self.detector=EventDetector(config);self.controller=LocalController(config);self.query=0
 def reset(self):
  self.detector.reset();self.controller.reset();self.query=0
 def step(self,base,reference,legal,*,adapted_ARM=None):
  if any(x.shape!=(32,14)or x.dtype!=np.float32 or not np.isfinite(x).all()for x in (base,reference)):raise ValueError('Actual once-base and frozen residual arrays required')
  if np.any(reference[:,[6,13]])or np.max(np.abs(reference[:,ARM]))>.100001:raise ValueError('Frozen ARM/gripper residual bounds')
  if adapted_ARM is not None:
   if adapted_ARM.shape!=(32,12)or adapted_ARM.dtype!=np.float32 or not np.isfinite(adapted_ARM).all()or np.max(np.abs(adapted_ARM))>.100001:raise ValueError('Finite bounded adapted ARM direction required')
  packet=candidate_packet(legal,reference)
  event,evidence=self.detector.step(self.query,packet)
  def score(p,context):return dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.)
  decision=self.controller.step(self.query,event,packet,score,enabled=True)
  direction=reference if adapted_ARM is None else reference.copy()
  if adapted_ARM is not None:direction[:,ARM]=adapted_ARM
  actions,delta=apply_arm_candidate(base,direction,decision['active'])
  assert np.array_equal(actions[:,[6,13]],base[:,[6,13]])
  metadata=dict(query=self.query,event=event,evidence=evidence,decision=decision,adapted_direction_supplied=adapted_ARM is not None,adaptation_applied=bool(decision['active']and not np.array_equal(direction,reference)))
  self.query+=1
  return actions,delta,metadata

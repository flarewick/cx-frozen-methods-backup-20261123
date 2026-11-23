"""Original passive readout grid; only initial own ARM path gets an ARM milestone."""
from native_feedback_application_v1.readouts import CurrentReadouts as Original
from native_stage_v1.clock import COMMON_TICKS
from .completion import ArmPathCompletion
KIND='initial_own_ARM_path'
class CurrentReadouts(Original):
 def __init__(self,recorder,reader,*,kind=None,initial=None,base=None,on_arrival=None):
  self.arm_path=kind==KIND
  super().__init__(recorder,reader,kind=None if self.arm_path else kind,initial=initial,base=base,on_arrival=on_arrival)
  if self.arm_path:self.completion=ArmPathCompletion(initial['command_state'],base)
 def command_begin(self,index):
  if self.arm_path:self.completion.command_begin(index)
 def after(self,before):
  if not self.arm_path:return super().after(before)
  if self.recorder.failed or len(self.recorder.rows)!=before+1:
   self.failed=True;raise ValueError('Exactly one actual native step required')
  self.flush();tick=len(self.recorder.rows);was=self.completion.pending['event']['requested95_tick']
  self.completion.after_arm_actuation(self.recorder.rows[-1]['arm_position_target'],committed_tick=tick,dt_s=self.recorder.dt[-1])
  new95=was!=self.completion.pending['event']['requested95_tick']
  if tick in COMMON_TICKS or new95:self.read(tick,common=tick in COMMON_TICKS,requested95=new95)

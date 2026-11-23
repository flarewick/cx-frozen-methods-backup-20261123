from native_arm_phase_feedback_v1.episode import execute as plain_original
from gated_arm_context_v1.episode import execute as gated_original
from native_q0_policy_overlap_v1.episode import make_execute
from native_seven_task_gate_expansion_v1.composite_head import CompositeHead
_plain=make_execute(plain_original);_gated=make_execute(gated_original)
CONDITIONS=('M_event_full','TTA','zero_update','shadow_update')
def execute(task,actor,observe,**kwargs):
 return(_gated if type(actor.head)is CompositeHead else _plain)(task,actor,observe,**kwargs)

def original_execute(task,actor,observe,**kwargs):
 return(gated_original if type(actor.head)is CompositeHead else plain_original)(task,actor,observe,**kwargs)
execute._q0_original=original_execute

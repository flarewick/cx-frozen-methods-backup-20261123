"""Dispatch only the two declared frozen methods, preserving original owner CodeTypes."""
from pathlib import Path
from native_seven_task_gate_expansion_v1.composite_head import CompositeHead
from native_seven_task_gate_expansion_v1.paired_actor import AuditedActor as PairedActor
from native_trained_runtime_v1.actor import AuditedActor as PlainActor

class AuditedActor(PairedActor):
    def query(self,*args,**kwargs):
        if type(self.head)is CompositeHead:return super().query(*args,**kwargs)
        return PlainActor.query(self,*args,**kwargs)

def build_head(training):
    from native_seven_task_gate_expansion_v1.contract import GATE_TRAINING,plan
    if Path(training).resolve()==GATE_TRAINING.resolve():
        from native_seven_task_gate_expansion_v1.gated_preload import build_head as gated
        return gated(training)
    assert Path(training).resolve()==Path(plan()['plain_training']).resolve()
    from cx_tta_feedback_confirm5.query import build_head as plain
    return plain(training)

def compare(folder,spec):
    if spec['group']=='gated':
        from gated_handoff_compare_v1.control import compare as implementation
    else:
        assert spec['group']=='plain'
        from native_q0_task_handoff_v3.control import compare as implementation
    return implementation(folder,spec)

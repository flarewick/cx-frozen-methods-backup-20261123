"""Dispatch a pure-base control; full/TTA use the explicit derived application version."""
from native_feedback_application_v1.episode import execute as derived_execute
from .baseline import execute as baseline_execute
CONDITIONS=('baseline','M_event_full','zero_update','shadow_update','TTA')
def execute(*args,condition,**kwargs):
 if condition=='baseline':return baseline_execute(*args,condition=condition,**kwargs)
 return derived_execute(*args,condition=condition,**kwargs)

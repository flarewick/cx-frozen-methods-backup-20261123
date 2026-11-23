"""Dispatch a pure-base control; full/TTA still call the frozen implementation."""
from native_best100_endpoint_v1.episode import execute as frozen_execute
from .baseline import execute as baseline_execute
CONDITIONS=('baseline','M_event_full','zero_update','shadow_update','TTA')
def execute(*args,condition,**kwargs):
 if condition=='baseline':return baseline_execute(*args,condition=condition,**kwargs)
 return frozen_execute(*args,condition=condition,**kwargs)

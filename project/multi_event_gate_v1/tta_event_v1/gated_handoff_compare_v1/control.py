"""Original whole-reference control with the explicit dual-readout audit binding."""
import types
from native_q0_task_handoff_v3.control import compare as original
from gated_native_audit_v1.replay import branch,same_query

compare=types.FunctionType(original.__code__,dict(original.__globals__,branch=branch,same_query=same_query),original.__name__,original.__defaults__,original.__closure__)

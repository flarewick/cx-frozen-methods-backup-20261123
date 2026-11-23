"""Same audit predicates; resolve recorded container paths on the host."""
from pathlib import Path
import types
from native_semantic_budget_pool_v1.audit import audit as original
from native_onpolicy_recovery_v3.common import mapped
from native_policy_execution_v1.common import write

def compute(root):
 captured=[]
 def capture(path,value):captured.append((path,value))
 namespace=dict(original.__globals__,Path=mapped,write=capture)
 function=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
 value=function(root);assert len(captured)==1 and captured[0][1]is value
 assert Path(captured[0][0]).resolve()==(Path(root)/'CPU_semantic_thread_budget_audit.json').resolve()
 assert all(mapped(name).resolve().is_relative_to((Path(root)/'semantic').resolve())for name in value['input_sha256'])
 return value

def audit(root):
 value=compute(root);write(Path(root)/'CPU_semantic_thread_budget_audit.json',value);return value

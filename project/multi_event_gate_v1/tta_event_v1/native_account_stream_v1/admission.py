"""Original admission code, changing only its fresh inventory dependency."""
import types
from native_resource_policy_v4 import admission as original
from .transport import inventory
require_admission=original.require_admission
scope=dict(original.check_admission.__globals__);scope['inventory']=inventory
check_admission=types.FunctionType(original.check_admission.__code__,scope,name=original.check_admission.__name__,argdefs=original.check_admission.__defaults__,closure=original.check_admission.__closure__)
assert check_admission.__code__ is original.check_admission.__code__

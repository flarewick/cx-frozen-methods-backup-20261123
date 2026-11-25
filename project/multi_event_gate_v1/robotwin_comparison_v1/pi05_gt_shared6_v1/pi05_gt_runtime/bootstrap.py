"""Redirect OUTPUT writers only, before importing original actor/feedback modules."""
from .common import O,write
def install():
 import native_policy_execution_v1.common as policy_common
 import native_cross_task100_v19.common as cross_common
 policy_common.write=write;cross_common.O=O

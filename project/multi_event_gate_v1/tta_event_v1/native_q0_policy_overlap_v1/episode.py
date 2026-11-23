"""Move the semantic wait across one unchanged q0 actor query, never an action."""
import inspect
import hashlib
import numpy as np

def check_initial(actor,initial,expected,instruction):
 if actor.instruction!=instruction:raise ValueError('Actor instruction mismatch before q0 overlap')
 if initial['native_action_count']!=0 or expected['native_action_count']!=0:raise ValueError('q0 overlap requires original zero-action state')
 if len(initial['rgb'])!=3 or any(not np.array_equal(a,b)for a,b in zip(initial['rgb'],expected['rgb'],strict=True))or not np.array_equal(initial['command_state'],expected['command_state']):raise ValueError('Initial overlap actor RGB or command mismatch')

def check_resolved(actor,initial,expected,available,instruction):
 if not isinstance(available,bool)or actor.instruction!=instruction or expected['instruction_sha256']!=hashlib.sha256(instruction.encode()).hexdigest()or expected['available']!=available:raise ValueError('Resolved semantic instruction or availability mismatch')
 if len(initial['rgb'])!=3 or any(not np.array_equal(a,b)for a,b in zip(initial['rgb'],expected['rgb'],strict=True))or not np.array_equal(initial['command_state'],expected['command_state']):raise ValueError('Resolved semantic RGB or command mismatch')

def source(original):
 before=inspect.getsource(original)
 signature='check_cancel=lambda:None,cpu_fixture=False):'
 typed=" if not isinstance(seed_available,bool)or int(task.step_lim)<=0 or int(task.take_action_cnt)!=0:raise ValueError('Actual semantic availability and original native budget required')"
 initial="""  if actor.instruction!=instruction or expected_initial['instruction_sha256']!=hashlib.sha256(instruction.encode()).hexdigest()or expected_initial['available']!=seed_available:raise ValueError('Semantic cache/actor instruction or availability mismatch')
  initial=observe()
  if len(initial['rgb'])!=3 or any(not np.array_equal(a,b)for a,b in zip(initial['rgb'],expected_initial['rgb'],strict=True))or not np.array_equal(initial['command_state'],expected_initial['command_state']):raise ValueError('Initial semantic-cache/current actor RGB or command mismatch')"""
 query='   proposal,policy,values=actor.query(current,observed,folder)'
 end="  if stop_reason is None:stop_reason='native_success_terminal'if bool(task.eval_success)else'native_step_limit'"
 new_typed=""" if semantic_bootstrap is None:
 """+typed+"""
 else:
  if condition not in _q0_allowed or seed_available is not None or expected_initial is not None or bootstrap_initial is None or int(task.step_lim)<=0 or int(task.take_action_cnt)!=0:raise ValueError('Explicit deferred q0 semantic bootstrap required')"""
 new_initial='  if semantic_bootstrap is None:\n'+'\n'.join(' '+line for line in initial.splitlines())+"""
  else:
   initial=observe()
   _q0_check_initial(actor,initial,bootstrap_initial,instruction)"""
 new_query=query+"""
   if q==0 and semantic_bootstrap is not None:
    expected_initial,seed_available=semantic_bootstrap(observed,policy)
    _q0_check_resolved(actor,initial,expected_initial,seed_available,instruction)
    semantic_bootstrap=None"""
 new_end="""  if semantic_bootstrap is not None:
   expected_initial,seed_available=semantic_bootstrap(None,None)
   _q0_check_resolved(actor,initial,expected_initial,seed_available,instruction)
   semantic_bootstrap=None
"""+end
 edits=[(signature,'check_cancel=lambda:None,cpu_fixture=False,semantic_bootstrap=None,bootstrap_initial=None):'),(typed,new_typed),(initial,new_initial),(query,new_query),(end,new_end)]
 changed=before
 for old,new in edits:
  assert changed.count(old)==1,('Unexpected original episode source',old[:60])
  changed=changed.replace(old,new)
 restored=changed
 for old,new in reversed(edits):
  assert restored.count(new)==1
  restored=restored.replace(new,old)
 assert restored==before,'Only enumerated semantic ordering and guards may change'
 return changed

def make_execute(original,*,allowed_deferred=('M_event_full',)):
 code=source(original);namespace=dict(original.__globals__,_q0_check_initial=check_initial,_q0_check_resolved=check_resolved,_q0_allowed=tuple(allowed_deferred))
 exec(compile(code,__file__,'exec'),namespace)
 fn=namespace[original.__name__];fn._q0_original=original;fn._q0_transformed_source=code
 return fn

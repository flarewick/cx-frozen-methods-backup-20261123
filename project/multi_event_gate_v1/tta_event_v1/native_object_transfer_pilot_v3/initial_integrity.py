"""Elide only a discarded setup when the complete original q0 boundary agrees."""
from .common import write

def prepare_initial(task,item,cfg,raw,routed_seed,*,expected_diagnostic,expected_key,instruction,replay,key_fn,unstable,output):
 if expected_diagnostic is not None:assert set(expected_diagnostic)=={'physical_sha256','rng_sha256'}
 attempts=0;initial=None;first_failed=False
 try:
  attempts+=1;initial=replay(task,item,[],cfg,raw,routed_seed)
 except unstable:first_failed=True
 def same(value):
  return value is not None and key_fn(value['rgb'],value['command_state'],instruction)==expected_key and expected_diagnostic is not None and all(value[k]==v for k,v in expected_diagnostic.items())
 fast=same(initial)
 if not fast:
  task.close_env(clear_cache=False)
  attempts+=1;initial=replay(task,item,[],cfg,raw,routed_seed)
 assert initial['native_action_count']==0
 assert key_fn(initial['rgb'],initial['command_state'],instruction)==expected_key,'Original q0 legal inputs changed'
 if expected_diagnostic is not None:assert same(initial),'Original q0 physical/RNG boundary changed'
 write(output,dict(passed=True,actual_native_setup_attempts=attempts,discarded_setup_elided=fast,original_double_initialization_retained=not fast,first_UnStableError=first_failed,expected_legal_RGB_command_language_key_matched=True,expected_physical_and_RNG_boundary_matched=True if expected_diagnostic is not None else None,missing_diagnostic_keeps_original_two_setups=expected_diagnostic is None,no_policy_actions_before_boundary=True,diagnostic_hashes_not_forward_inputs=True,original_native_setup_and_render_functions_unchanged=True))
 return initial

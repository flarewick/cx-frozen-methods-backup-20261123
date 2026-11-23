"""CPU-only original-boundary equivalence, fallback and fail-closed checks."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
from .initial_integrity import prepare_initial
from .common import O,read,write

def main():
 class Unstable(Exception):pass
 good=dict(rgb='same',command_state='command',physical_sha256='physical',rng_sha256='rng',native_action_count=0)
 expected={k:good[k]for k in('physical_sha256','rng_sha256')}
 checked=[]
 with tempfile.TemporaryDirectory(prefix='tta_initial_boundary_cpu_',dir=O)as temp:
  root=Path(temp)
  def trial(name,frames,diagnostic=expected,fail=False):
   calls=[];closes=[];task=SimpleNamespace(close_env=lambda **k:closes.append(k))
   def replay(*args):
    assert args[0]is task and args[2]==[]
    row=frames[len(calls)];calls.append(row)
    if isinstance(row,Exception):raise row
    return row
   try:value=prepare_initial(task,dict(case_id='opaque'),{},None,None,expected_diagnostic=diagnostic,expected_key='same',instruction='legal-language',replay=replay,key_fn=lambda rgb,command,instruction:rgb,unstable=Unstable,output=root/(name+'.json'))
   except AssertionError:
    assert fail;checked.append(name);return
   assert not fail and value is good
   proof=read(root/(name+'.json'));assert len(calls)==proof['actual_native_setup_attempts']and len(closes)==len(calls)-1 and all(c==dict(clear_cache=False)for c in closes);checked.append(name)
   return proof
  assert trial('exact_boundary',[good])['discarded_setup_elided']
  assert trial('RGB_mismatch_retains_original_warmup',[dict(good,rgb='different'),good])['original_double_initialization_retained']
  assert trial('RNG_mismatch_retains_original_warmup',[dict(good,rng_sha256='different'),good])['original_double_initialization_retained']
  assert trial('physical_mismatch_retains_original_warmup',[dict(good,physical_sha256='different'),good])['original_double_initialization_retained']
  assert trial('missing_diagnostic_original_path',[good,good],None)['missing_diagnostic_keeps_original_two_setups']
  assert trial('original_UnStable_warmup_fallback',[Unstable(),good])['first_UnStableError']
  trial('second_initialization_wrong_RGB_rejected',[dict(good,rgb='bad'),dict(good,rgb='bad')],fail=True)
  trial('second_initialization_wrong_RNG_rejected',[dict(good,rng_sha256='bad'),dict(good,rng_sha256='bad')],fail=True)
 write(O/'CPU_initial_integrity_checks.json',dict(passed=True,CPU_only=True,no_GPU_or_native_scene_created=True,checks=checked,only_exact_original_boundary_can_skip_discarded_setup=True,original_fallback_and_failure_semantics_preserved=True,no_outcomes_or_model_features_used=True))
 print('EXACT_ORIGINAL_Q0_BOUNDARY_CPU_PASS',flush=True)
if __name__=='__main__':main()

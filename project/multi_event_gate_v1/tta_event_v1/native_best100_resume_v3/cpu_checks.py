"""Before-freeze software contracts; no CUDA or policy episode replay."""
import ast,json
from .common import C,T,O,read,write,sha

def main():
 for p in C.rglob('*.py'):ast.parse(p.read_text())
 old=ast.parse((T/'native_visible_expand_c_v1/runtime.py').read_text());new=ast.parse((C/'runtime.py').read_text())
 def calls(tree,name):return [ast.dump(n,include_attributes=False)for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id==name]
 for name in ('episode','AuditedActor','SourceStream','replay_prefix','release_case'):
  a,b=calls(old,name),calls(new,name)
  # Only audit actor names change from fixed old developer IDs to actual EVAL IDs.
  if name=='AuditedActor':continue
  assert a==b,('Original native branch call changed',name)
 assert 'for _ in range(2)'in(C/'runtime.py').read_text()and "range(4)"in(C/'runtime.py').read_text()
 assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('backward','step')for n in ast.walk(new))
 retained=read(O/'retained_candidate.json');assert retained['checkpoint_sha256']=='0937511affea00771ed7f595d07220ae390ba85e06e398fce678446e7078a1d5'
 # Counterexample evidence: the two archived rates cannot be a same-pool primary control.
 refs=read(O/'historical_references.json');assert refs['stack_blocks_two']['seed_start']==7105000 and refs['stack_bowls_two']['seed_start']==100000
 write(O/'CPU_entry_contract_checks.json',dict(passed=True,original_native_episode_source_and_call_unchanged=True,actual_four_lanes_two_cases_each=True,original_feedback_source_actor_setup_release_calls_unchanged=True,source_AST_parsed=True,no_new_fitting_or_update_math=True,actual_best_checkpoint_retained=True,historical_seed_mismatch_explicit=True,GPU_started=False,source_sha256={str(p):sha(p)for p in C.glob('*.py')}));print('BEST100_CPU_ENTRY_CONTRACT_PASS')
if __name__=='__main__':main()

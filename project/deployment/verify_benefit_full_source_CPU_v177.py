"""Original v5 CPU verifier with one strictly proven source-table extension."""
import ast,copy,inspect,json,textwrap
from pathlib import Path
import benefit_followup_protocol_v2 as p

def verify_sources(live):
 original=json.loads((p.HOME/'learning_v6/preregistration.json').read_text())['collection_source_sha256'];impl=json.loads((p.ROOT.parent/'implementation_v174.json').read_text());expected=dict(original,**impl['source_sha256']);assert len(original)==83
 assert live==expected,'Original83 or explicit full-event overlay table differs'
 assert all(p.sha(p.D/n)==h for n,h in expected.items()),'An original or overlay source changed'
 return True

def main():
 import verify_benefit_followup_mixed_cpu_v5 as old
 function=old.main;original=ast.parse(textwrap.dedent(inspect.getsource(function)));tree=copy.deepcopy(original);hits=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.BoolOp):
   for i,value in enumerate(n.values):
    if ast.unparse(value)=='len(live) == 83':hits.append((n,i));n.values[i]=ast.parse("__import__('verify_benefit_full_source_CPU_v177').verify_sources(live)",mode='eval').body
 assert len(hits)==1
 restored=copy.deepcopy(tree)
 for n in ast.walk(restored):
  if isinstance(n,ast.BoolOp):
   for i,value in enumerate(n.values):
    if ast.unparse(value)=="__import__('verify_benefit_full_source_CPU_v177').verify_sources(live)":n.values[i]=ast.parse('len(live)==83',mode='eval').body
 assert ast.dump(original,include_attributes=False)==ast.dump(restored,include_attributes=False),'Any original runtime/planner/RNG/scope/identity checks changed'
 namespace=dict(function.__globals__);exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace);namespace['main']()
 file=p.ROOT/'mixed_runtime_cpu_proof.json';data=json.loads(file.read_text());assert data['passed']and verify_sources(data['source_sha256']);base=p.ROOT/'mixed_runtime_CPU_base_result_v177.json';assert not base.exists();base.write_bytes(file.read_bytes())
 extension=dict(passed=True,base_checker_file_sha256=p.sha(p.D/'verify_benefit_followup_mixed_cpu_v5.py'),executed_extension_sha256=p.sha(__file__),inverse_AST_all_original_checks_except_count_exact=True,source_count_check_replaced_by_exact_original83_plus_registered_overlay_hashes=True,actual_runtime_source_count=len(data['source_sha256']),original_runtime_count=83,base_CPU_result_sha256=p.sha(base),checker_sha256_field_retained_as_original_base_version_for_old_contract_compatibility=True,not_claiming_unmodified_original_main_executed=True)
 data['explicit_source_table_extension_v177']=extension;p.atomic(file,data);p.atomic(p.ROOT/'source_extension_CPU_proof_v177.json',dict(extension,mixed_runtime_CPU_proof_sha256=p.sha(file)));print('FULL_SOURCE_EXTENSION_CPU_PASSED',len(data['source_sha256']),flush=True)
if __name__=='__main__':main()

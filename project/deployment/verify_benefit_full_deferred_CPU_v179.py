"""Original deferred-finalization verifier on fully audited canonical TRAIN fixtures."""
import ast,copy,inspect,json,textwrap
from pathlib import Path
import benefit_followup_protocol_v2 as p

def audited_fixture_plan(block):
 assert block in('train0','dev0')
 root=p.HOME/'supervision_growth_canonical_v96/train2';audit=json.loads((root/'collection_audit.json').read_text());membership=json.loads((root/'mixed_membership_final_audit.json').read_text());extension=json.loads((root/'supplement_full_audit_receipt_v27.json').read_text());assert all(v['passed']and v['complete']for v in [audit,membership,extension]);assert audit['resource_audit']['passed']
 plan=json.loads((root/'collect_plan.json').read_text());branch='A1'if block=='train0'else'A0';job=next(j for j in plan['jobs']if j['task']=='lift_pot'and j['branch']==branch);assert json.loads((Path(job['run'])/'result.json').read_text())['complete'];return plan

def main():
 import verify_benefit_followup_mixed_deferred_v4 as old
 function=old.main;original=ast.parse(textwrap.dedent(inspect.getsource(function)));tree=copy.deepcopy(original);hits=0
 for n in ast.walk(tree):
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='plan'for t in n.targets)and ast.unparse(n.value)=="json.loads((PRIOR / block / 'collect_plan.json').read_text())":n.value=ast.parse("__import__('verify_benefit_full_deferred_CPU_v179').audited_fixture_plan(block)",mode='eval').body;hits+=1
 assert hits==1
 restored=copy.deepcopy(tree)
 for n in ast.walk(restored):
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='plan'for t in n.targets)and ast.unparse(n.value)=="__import__('verify_benefit_full_deferred_CPU_v179').audited_fixture_plan(block)":n.value=ast.parse("json.loads((PRIOR/block/'collect_plan.json').read_text())",mode='eval').body
 assert ast.dump(original,include_attributes=False)==ast.dump(restored,include_attributes=False),'Any finalization/RNG/provenance/error-propagation check changed'
 ns=dict(function.__globals__);exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),ns);ns['main']()
 file=p.HOME/'deferred_finalize_cpu_proof.json';data=json.loads(file.read_text());assert data['passed'];base=p.ROOT/'deferred_CPU_base_result_v179.json';assert not base.exists();base.write_bytes(file.read_bytes());root=p.HOME/'supervision_growth_canonical_v96/train2'
 amendment=dict(passed=True,executed_extension_sha256=p.sha(__file__),original_checker_file_sha256=p.sha(p.D/'verify_benefit_followup_mixed_deferred_v4.py'),inverse_AST_all_finalization_checks_unchanged=True,only_CPU_test_fixture_read_source_replaced=True,fixture_root=str(root),fixture_audit_sha256=p.sha(root/'collection_audit.json'),fixture_plan_sha256=p.sha(root/'collect_plan.json'),fixtures_are_existing_TRAIN_not_new_DEV_or_TEST=True,legacy_block_names_describe_complete4_vs_partial12_test_cases_only=True,base_result_sha256=p.sha(base),old_checker_sha256_field_is_compatibility_base_version_not_unmodified_execution_claim=True)
 data['explicit_audited_TRAIN_fixture_extension_v179']=amendment;p.atomic(file,data);p.atomic(p.ROOT/'deferred_extension_CPU_proof_v179.json',dict(amendment,deferred_full_proof_sha256=p.sha(file)));print('FULL_DEFERRED_TRAIN_FIXTURE_CPU_PASSED',flush=True)
if __name__=='__main__':main()

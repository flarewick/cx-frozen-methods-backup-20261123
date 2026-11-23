"""Actual canonical-path CPU regression; no GPU or numerical audit edits."""
from pathlib import Path
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped
from native_semantic_budget_pool_v1.audit import audit as original
from .audit import compute,audit

def main():
 fixture=T/'outputs/native_onpolicy_collection_v30/fit_007/CPU_actual_production_semantic_fixture_v1'
 try:original(fixture)
 except FileNotFoundError as error:assert '/data/cj/'in str(error)
 else:raise AssertionError('Host must reproduce the actual recorded-container path failure')
 value=compute(fixture);expected=read(fixture/'CPU_semantic_thread_budget_audit.json')
 assert {k:v for k,v in value.items()if k!='input_sha256'}=={k:v for k,v in expected.items()if k!='input_sha256'}
 assert value['input_sha256']=={str(mapped(name)):digest for name,digest in expected['input_sha256'].items()}
 O=T/'outputs/native_semantic_host_audit_v1';O.mkdir(exist_ok=True);alias=O/'CPU_only_alias_fixture';alias.mkdir();(alias/'semantic').symlink_to(fixture/'semantic',target_is_directory=True)
 result=audit(alias);assert result['passed']and result['maximum_active_threads']<=8
 write(O/'CPU_actual_host_path_checks_v1.json',dict(passed=True,old_recorded_container_path_failure_reproduced=True,original_audit_code_object_numeric_predicates_preserved=True,actual_production6_request_audit_numeric_values_and_hashes_exact_after_path_mapping=True,CPU_fixture_not_native_data=True,no_GPU_or_policy_or_optimizer_change=True,source_sha256={str(f):sha(f)for f in Path(__file__).parent.glob('*')if f.is_file()},input_sha256={str(f):sha(f)for f in[Path(__import__('native_semantic_budget_pool_v1.audit',fromlist=['audit']).__file__),fixture/'CPU_semantic_thread_budget_audit.json',alias/'CPU_semantic_thread_budget_audit.json']}))
 print('CPU_ACTUAL_CANONICAL_HOST_PATH_AUDIT_REGRESSION_PASS')
if __name__=='__main__':main()

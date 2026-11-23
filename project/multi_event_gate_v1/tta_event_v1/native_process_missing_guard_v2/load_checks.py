"""Actual original host guard import/compile with derived helpers; no NVML/CUDA calls."""
from pathlib import Path
import ast
from .guard import _implementation,MIN_GPU_UTILIZATION,MAX_GPU_MEMORY_MIB
from .empty import admissible_missing,zero_previous
from .terminal import transform_text,terminal_ast_exact
from native_policy_execution_v1.common import T,read,write,sha

def main():
 original=read(T/'outputs/native_process_missing_guard_v2/CPU_actual_C20_idle_lower_bound_checks.json');assert original['passed']
 for source,digest in original['source_sha256'].items():assert sha(source)==digest
 actual=read(T/'outputs/native_process_missing_guard_v2/CPU_actual_host_exit_v3.json');assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
 module=_implementation();assert module.admissible_missing is admissible_missing and module.zero_previous is zero_previous
 from native_account_stream_v1.transport import inventory
 from .rewrite import transform
 tree=transform((T/'native_process_sampler_v13/strict_guard.py').read_text());imports=[n for n in ast.walk(tree)if isinstance(n,ast.ImportFrom)and n.module=='native_account_stream_v1.transport'];assert len(imports)==1
 namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=imports,type_ignores=[])),__file__,'exec'),namespace)
 assert namespace['inventory']is inventory and MIN_GPU_UTILIZATION==34 and MAX_GPU_MEMORY_MIB==24563
 text=(T/'native_onpolicy_collection_v20/host.py').read_text();a=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='terminal');b=next(n for n in ast.parse(transform_text(text)).body if isinstance(n,ast.FunctionDef)and n.name=='terminal');assert terminal_ast_exact(b,a)
 write(T/'outputs/native_process_missing_guard_v2/CPU_actual_host_guard_load_checks_v2.json',dict(passed=True,actual_original_guard_imported_and_compiled=True,actual_original_account_inventory_identity=True,derived_helpers_bound_to_actual_runtime=True,only_terminal_missing_idle_predicate_changed=True,original_average_thresholds_34_and_cap24563=True,no_NVML_requests_or_GPU_work=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()}));print('ACTUAL_HOST_GUARD_LOAD_AND_STRICT_TERMINAL_EDIT_PASS',flush=True)
if __name__=='__main__':main()

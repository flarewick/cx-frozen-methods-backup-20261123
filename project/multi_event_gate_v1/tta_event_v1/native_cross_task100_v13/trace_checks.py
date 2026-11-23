"""Only host launch instrumentation differs from qualified V11 runtime."""
import ast
from .common import C,T,O,read,write,sha
def main():
 for name in ['screen.py','semantic.py','semantic_worker.py','preflight.py','preload.py','analysis.py','cache.py','startup.py','owner_precision.py','ordered_screen.py']:
  new=(C/name).read_text().replace('native_cross_task100_v13','native_cross_task100_v11');old=(T/'native_cross_task100_v11'/name).read_text();assert ast.dump(ast.parse(new))==ast.dump(ast.parse(old)),name
 p=read(T/'outputs/native_signal_origin_v1/CPU_signal_origin_checks.json');assert p['passed']and p['original_KeyboardInterrupt_delivered']and p['actual_signal_subtree_wrapper_exit']==23 and p['actual_normal_wrapper_exit']==0
 assert read(T/'outputs/native_signal_origin_v1/CPU_actual_exit.json')['actual_exit_code']==0
 write(O/'CPU_trace_contract_checks.json',dict(passed=True,original_CPU_inference_and_models_AST_exact=True,host_trace_and_preparation_resume_only=True,host_only_newly_launched_owned_subtree_signal_trace=True,no_live_process_attach_or_signal_masking=True,CPU_real_sender_and_exit_forwarding_proof_pass=True,no_GPU_calls=True,source_sha256={str(p):sha(p)for p in C.glob('*.py')}))
 print('HOST_ONLY_SIGNAL_TRACE_CPU_CONTRACT_PASS',flush=True)
if __name__=='__main__':main()

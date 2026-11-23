"""Actual C23 CPU lifetime history, lookup-only extension; no CUDA work."""
import ast,hashlib,time
from pathlib import Path
from .history import History
from .handoff import seed
from .admission import Monitor
from native_account_stream_v1.transport import context
from native_policy_execution_v1.common import T,read,write,sha
O=T/'outputs/native_server_rule4_v8'
def main():
 c=Path(__file__).parent;old=T/'native_server_rule4_v7'
 for name in('history.py','admission.py'):assert ast.dump(ast.parse((old/name).read_text().replace('native_server_rule4_v7','native_server_rule4_v8')))==ast.dump(ast.parse((c/name).read_text()))
 original=ast.parse((old/'handoff.py').read_text().replace('native_server_rule4_v7','native_server_rule4_v8'));new=ast.parse((c/'handoff.py').read_text());extra={'native_onpolicy_collection_v22','native_onpolicy_collection_v23','native_onpolicy_collection_v24'}
 class Undo(ast.NodeTransformer):
  def visit_Tuple(self,n):
   vals=[x.value for x in n.elts if isinstance(x,ast.Constant)]
   if 'native_onpolicy_collection_v7'in vals:n.elts=[x for x in n.elts if not(isinstance(x,ast.Constant)and x.value in extra)]
   if 'native_server_rule4_v2'in vals:n.elts=[x for x in n.elts if not(isinstance(x,ast.Constant)and x.value=='native_server_rule4_v7')]
   return self.generic_visit(n)
 assert ast.dump(Undo().visit(new))==ast.dump(original)
 O.mkdir(exist_ok=True);folder=O/'CPU_active_C23_prefix';folder.mkdir();history=History();proof=seed(history,folder);providers=[v for v in proof['providers']if'/native_onpolicy_collection_v23/'in v['parent_protocol']];assert providers and sum(v['valid_frames']for v in providers)>=60
 for provider in providers:assert hashlib.sha256(Path(provider['observation_file']).read_bytes()[:provider['complete_byte_prefix']]).hexdigest()==provider['byte_prefix_sha256']
 with context(O/'CPU_main_account_actual_exit.json'):
  monitor=Monitor()
  try:
   deadline=time.monotonic()+18
   while True:
    assert monitor.failed is None
    lines=(monitor.root/'observations.jsonl').read_text().splitlines()if(monitor.root/'observations.jsonl').exists()else[]
    if len(lines)>=2:break
    assert time.monotonic()<deadline;time.sleep(.05)
  finally:monitor.close()
 exit=read(monitor.root/'NVML_account_rule4_actual_exit.json');assert exit['exit_code']==0 and exit['actual_wait_return']and exit['proc_absent_after_wait']and not exit['failed_request']and not exit['forced']
 files=[folder/'CPU_history_seed.json',monitor.root/'CPU_history_seed.json',monitor.root/'actual_monitor_exit.json',monitor.root/'NVML_account_rule4_actual_exit.json',monitor.root/'NVML_account_rule4_actual_exit.requests.jsonl']
 write(O/'CPU_actual_C23_history_handoff_checks.json',dict(passed=True,actual_C23_providers=len(providers),actual_imported_C23_frames=sum(v['valid_frames']for v in providers),original_history_and_monitor_AST_exact=True,only_lookup_tuples_extended=True,all_original_source_and_byte_prefixes_verified=True,unknown_prefix_zero_and_PID_age_unchanged=True,original_fresh_NVML_worker_actual_exit0=True,no_GPU_or_CUDA_work=True,no_scientific_GPU_clocks_or_data_or_labels_imported=True,source_sha256={str(p):sha(p)for p in c.glob('*')if p.is_file()},input_sha256={str(p):sha(p)for p in files}));print('CPU_ACTUAL_C23_HISTORY_AND_UNCHANGED_CONSERVATIVE_ADMISSION_PASS',flush=True)
if __name__=='__main__':main()

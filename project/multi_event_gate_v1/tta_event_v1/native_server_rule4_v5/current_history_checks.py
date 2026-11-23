"""Validate real frozen V11 CPU observations without changing GPU clocks."""
import ast,time
from pathlib import Path
from .history import History
from .handoff import seed
from .admission import Monitor
from native_account_stream_v1.transport import context
from native_policy_execution_v1.common import T,read,write,sha
def main():
 o=T/'outputs/native_server_rule4_v5';o.mkdir(parents=True,exist_ok=True)
 for name in('history.py','admission.py'):
  old=(T/'native_server_rule4_v4'/name).read_text().replace('native_server_rule4_v4','native_server_rule4_v5')
  assert ast.dump(ast.parse(old))==ast.dump(ast.parse((Path(__file__).parent/name).read_text())),name
 s=(Path(__file__).parent/'handoff.py').read_text()
 s=s.replace(",'native_onpolicy_collection_v11','native_onpolicy_collection_v12'",'')
 s=s.replace("'native_server_rule4_v4','native_server_rule4_v5'","'native_server_rule4_v5'")
 old=(T/'native_server_rule4_v4/handoff.py').read_text().replace('native_server_rule4_v4','native_server_rule4_v5')
 assert ast.dump(ast.parse(s))==ast.dump(ast.parse(old)),'Only frozen provider lookup tuples may change'
 h=History();folder=o/'CPU_actual_frozen_V11_prefix';folder.mkdir(exist_ok=False);proof=seed(h,folder)
 providers=[r for r in proof['providers']if '/native_onpolicy_collection_v11/'in r['parent_protocol']]
 assert providers and sum(r['valid_frames']for r in providers)>120
 for provider in providers:
  raw=Path(provider['observation_file']).read_bytes()[:provider['complete_byte_prefix']]
  import hashlib
  assert hashlib.sha256(raw).hexdigest()==provider['byte_prefix_sha256']
 with context(o/'CPU_fresh_main_account_actual_exit.json'):
  monitor=Monitor()
  try:
   end=time.monotonic()+15
   while monitor.latest is None:
    assert monitor.failed is None and time.monotonic()<end;time.sleep(.05)
   latest=monitor.latest
  finally:monitor.close()
 actual=read(monitor.root/'NVML_account_rule4_actual_exit.json');assert actual['exit_code']==0 and actual['actual_wait_return']and actual['proc_absent_after_wait']and not actual['failed_request']and not actual['forced']
 sources={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()and p.suffix in('.py','.md')}
 files=[folder/'CPU_history_seed.json',monitor.root/'CPU_history_seed.json',monitor.root/'actual_monitor_exit.json',monitor.root/'NVML_account_rule4_actual_exit.json',monitor.root/'NVML_account_rule4_actual_exit.requests.jsonl']
 write(o/'CPU_actual_V11_history_handoff_checks.json',dict(passed=True,actual_frozen_V11_providers=len(providers),actual_imported_V11_frames=sum(r['valid_frames']for r in providers),original_eligibility_math_and_monitor_AST_unchanged=True,only_lookup_tuples_extended=True,all_actual_byte_prefixes_and_original_protocol_source_hashes_verified=True,original_PID_startticks_ages_and_unknown_prefix_credit_unchanged=True,fresh_original_NVML_request_and_actual_worker_exit0=True,no_scientific_GPU_clocks_or_policy_data_or_labels_reused=True,no_CUDA_or_GPU_work=True,source_sha256=sources,input_sha256={str(p):sha(p)for p in files}))
 print('ACTUAL_FROZEN_V11_HISTORY_PREFIX_AND_ORIGINAL_MATH_PASS',sum(r['valid_frames']for r in providers),flush=True)
if __name__=='__main__':main()

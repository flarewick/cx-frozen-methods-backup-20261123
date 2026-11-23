"""Actual resource history prefixes only; no GPU work or new eligibility math."""
from pathlib import Path
import ast,json
from native_policy_execution_v1.common import T,write,sha,read
from .history import History
from .handoff import seed

def main():
 C=Path(__file__).parent;old=T/'native_server_rule4_v9'
 assert(C/'history.py').read_text()==(old/'history.py').read_text()
 assert(C/'admission.py').read_text()==(old/'admission.py').read_text().replace('native_server_rule4_v9','native_server_rule4_v10')
 O=T/'outputs/native_server_rule4_v10';O.mkdir(exist_ok=True);folder=O/'CPU_actual_C27_and_observer_prefix';assert not folder.exists();folder.mkdir()
 history=History();proof=seed(history,folder);assert proof['imported_frames']>0
 current=[v for v in proof['providers']if'/native_onpolicy_collection_v27/'in v['parent_protocol']];assert len(current)>=2 and all(v['valid_frames']>0 for v in current)
 observer=[v for v in proof['providers']if v.get('standalone_CPU_only_observer')];assert len(observer)==1 and observer[0]['valid_frames']>0
 for row in proof['providers']:
  data=Path(row['observation_file']).read_bytes()[:row['complete_byte_prefix']];assert __import__('hashlib').sha256(data).hexdigest()==row['byte_prefix_sha256']
 inputs=[folder/'CPU_history_seed.json',T/'outputs/native_onpolicy_full_ops_v2/CPU_only_rule4_observer_v1/protocol.json']
 write(O/'CPU_actual_C27_observer_history_checks_v1.json',dict(passed=True,history_and_admission_math_byte_exact_v9=True,threshold55_observed120_unknown_prefix_zero_and_gaps8_unchanged=True,actual_C27_source_bound_providers=len(current),actual_CPU_only_observer_prefix=True,scientific_GPU_metrics_or_clocks_imported=False,no_GPU_optimizer_or_scientific_qualification=True,source_sha256={str(f):sha(f)for f in C.glob('*')if f.is_file()},input_sha256={str(f):sha(f)for f in inputs}))
 print('RULE4V10_ACTUAL_C27_OBSERVER_PREFIX_AND_UNCHANGED_MATH_PASS',flush=True)
if __name__=='__main__':main()

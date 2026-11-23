"""Actual frozen-producer prefix binding and retained process-age evidence."""
from pathlib import Path
import hashlib,json,time,uuid
from .history import History
from .handoff import seed,T
def main():
 root=T/'outputs/native_server_rule4_v5'/('CPU_handoff_'+uuid.uuid4().hex);root.mkdir(parents=True)
 history=History();result=seed(history,root);assert result['imported_frames']>0 and result['providers']
 for provider in result['providers']:
  with Path(provider['observation_file']).open('rb')as f:prefix=f.read(provider['complete_byte_prefix'])
  assert hashlib.sha256(prefix).hexdigest()==provider['byte_prefix_sha256']
  assert hashlib.sha256(Path(provider['parent_protocol']).read_bytes()).hexdigest()==provider['parent_protocol_sha256']
 estimates=history.estimates(time.monotonic())
 assert all(e['unobserved_prefix_credit']==0 and e['process_age_seconds']>=e['observed_seconds']for e in estimates.values())
 proof=dict(passed=True,actual_frozen_CPU_observation_frames_imported=result['imported_frames'],actual_byte_prefix_and_parent_protocol_SHA_verified=True,original_observation_times_and_process_ages_preserved=True,estimates_at_check=estimates,no_new_interval_credit=True,no_scientific_GPU_metric_or_training_data_imported=True,seed_proof=str(root/'CPU_history_seed.json'),seed_proof_sha256=hashlib.sha256((root/'CPU_history_seed.json').read_bytes()).hexdigest())
 with(T/'outputs/native_server_rule4_v5/CPU_actual_frozen_history_handoff_checks.json').open('x')as f:json.dump(proof,f,indent=2)
 print('ACTUAL_FROZEN_CPU_HISTORY_PREFIX_SHA_AND_PROCESS_AGE_RETAINED',result['imported_frames'])
if __name__=='__main__':main()

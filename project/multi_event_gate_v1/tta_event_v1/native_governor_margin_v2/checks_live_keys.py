"""Real runtime integer keys and serialized keys must produce identical guards."""
import json,copy
from pathlib import Path
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped
from .margin import Monitor
from .checks import run

def main():
 value=run();count=0
 for version in [17,18]:
  root=mapped(read(T/f'outputs/native_onpolicy_arm_collection_v{version}/val_000/execution_state.json')['root']);a=Monitor();b=Monitor()
  for line in(root/'attempt/resources.jsonl').read_text().splitlines():
   row=json.loads(line)
   if 'seconds'not in row:continue
   live=copy.deepcopy(row)
   for name in ['process_utilization','process_mean_utilization','missing_process_zero_lower_bound']:
    if name in live:live[name]={int(k):v for k,v in live[name].items()}
   assert a.observe(row)==b.observe(live);count+=1
 value.update(actual_runtime_integer_and_serialized_PID_key_results_exact=count)
 value['source_sha256']={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()}
 write(T/'outputs/native_governor_margin_v2/CPU_live_key_checks_v1.json',value);print('LIVE_INTEGER_AND_SERIALIZED_PID_KEYS_EXACT',count)
if __name__=='__main__':main()

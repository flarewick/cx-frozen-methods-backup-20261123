"""Check exact old60 windows and forwarding to original guarded command."""
import json,math,tempfile
from pathlib import Path
from unittest.mock import patch
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped
from . import margin
from .checks import run

def main():
 value=run();count=0
 for version in [17,18]:
  root=mapped(read(T/f'outputs/native_onpolicy_arm_collection_v{version}/val_000/execution_state.json')['root']);monitor=margin.Monitor()
  for text in(root/'attempt/resources.jsonl').read_text().splitlines():
   row=json.loads(text)
   if 'seconds'not in row:continue
   _,extra=monitor.observe(row)
   for pid in row['gpu_pids']:
    assert math.isclose(extra['process_rolling60'][str(pid)],row['process_recent_60s_utilization'][str(pid)],rel_tol=1e-11,abs_tol=1e-9);count+=1
 calls=[];callbacks=[]
 def original(*args,**kwargs):
  calls.append((args,{k:v for k,v in kwargs.items()if k!='on_sample'}))
  row=dict(seconds=1,gpu_pids=[],process_utilization={},process_mean_utilization={},mean_utilization=0,time_weighted_mean_utilization=0,rolling_300s_utilization=0,recent_60s_utilization=0)
  assert kwargs['on_sample'](row)=='original_callback_reason';return 17
 folder=T/'outputs/native_governor_margin_v2/CPU_callback_fixture_v3';folder.mkdir(parents=True)
 with patch.object(margin.original,'guarded_command',original):
  result=margin.guarded_command(['original','command'],{'uuid':'unchanged'},folder/'resources.jsonl',gpu=5,deadline=3391.053,min_cumulative_utilization=34,max_memory_mib=24563,on_sample=lambda row:callbacks.append(row)or'original_callback_reason')
 assert result==17 and len(calls)==len(callbacks)==1
 assert calls[0][1]==dict(gpu=5,deadline=3391.053,min_cumulative_utilization=34,max_memory_mib=24563)
 value.update(exact_original60_window_values=count,original_call_arguments_and_result_exact=True,original_callback_called_once_and_reason_preserved=True)
 value['source_sha256']={str(f):sha(f)for f in Path(__file__).parent.glob('*')if f.is_file()}
 write(T/'outputs/native_governor_margin_v2/CPU_checks_v3.json',value);print('EXACT_GOVERNOR_WINDOW_AND_ORIGINAL_CALLBACK_PASS',count)
if __name__=='__main__':main()

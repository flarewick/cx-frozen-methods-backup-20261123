"""Actual host/child/resource completion for this source-bound developer attempt."""
from pathlib import Path
import argparse,json
from .paths import O,read,sha,write,verify

def audit():
 runs=list(O.glob('host_run_*'));assert len(runs)==1;hostfile=runs[0]/'exit.json';host=read(hostfile)
 assert host['actual_Popen_wait_return']and host['exit_code']==0
 roots=list((O/'collections').iterdir());assert len(roots)==1;root=roots[0];spec=read(O/'protocol.json');verify(spec)
 result=read(root/'result.json');assert result['complete']and result['actual_developer_native_episodes']==12
 assert result['protocol_identity']==spec['identity']and result['CPU_audit']['passed']
 attempts=list((root/'host_attempts').iterdir());assert len(attempts)==1;attempt=attempts[0]
 childfile=attempt/'barrier/normal_child_exit.json';child=read(childfile)
 assert child['exit_code']==0 and child['normal_exit']and child['proc_absent_after_wait']and not child['cpu_only_fixture']
 assert child['protocol_identity']==spec['identity']and child['gpu_uuid']==spec['required_gpu_uuid']
 resourcesfile=attempt/'resources.jsonl';rows=[json.loads(line)for line in resourcesfile.read_text().splitlines()]
 assert not any('stopped'in r for r in rows)
 releases=[r for r in rows if'own_gpu_released'in r];assert len(releases)==1;release=releases[0]
 assert release['own_gpu_released']and not release['remaining_own_gpu_pids']
 for pid in {pid for r in rows for pid in r.get('gpu_pids',[])}:assert not Path('/proc/%d'%pid).exists(),'Former CUDA process still exists'
 steady=[r for r in rows if r.get('seconds',0)>=280 and r.get('gpu_pids')]
 assert steady,'Full developer workload must actually qualify beyond resource threshold'
 for r in steady:
  assert min(r['mean_utilization'],r['time_weighted_mean_utilization'],r['rolling_300s_utilization'])>=34
  assert 4096<=r['memory_mib']<=24563 and r['account_gpu_count']<=3
  for pid in r['gpu_pids']:
   key=str(pid);assert r['process_utilization'][key]is not None
   assert min(r['process_mean_utilization'][key],r['process_rolling_300s_utilization'][key])>=34
   assert r['process_memory_mib'][key]>=4096 and 40<=r['cpu'][key]<=1600 and r['rss_bytes'][key]<=128*2**30
 files=[O/'protocol.json',hostfile,childfile,resourcesfile,root/'result.json',root/'CPU_episode_audit.json']
 value=dict(passed=True,actual_host_exit_code=0,actual_child_exit_code=0,own_GPU_released=True,steady_resource_samples=len(steady),requested_cases=4,requested_branches=12,comparison_total_branches_including_reused_controls=20,actual_developer_integration_only=True,new_independent_scenes=0,root=str(root),input_sha256={str(p):sha(p)for p in files},TTA_utility_proven=False)
 dest=O/'terminal_audit.json'
 if dest.exists():assert read(dest)==value
 else:write(dest,value)
 return value
if __name__=='__main__':print(json.dumps(audit()),flush=True)

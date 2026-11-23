"""One changed preparation configuration after an audited cold resource stop, never unchanged retry."""
import json
from .common import read,sha,mapped,verify
ALLOWED={'Rule4: rolling180s GPU utilization/VRAM','Rule4: rolling180s GPU utilization/VRAM; Unverified180s process window','Unverified180s process window'}
def evidence(base):
 actual=base/'actual_host_wait_exit.json';a=read(actual)
 assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 root=mapped(read(base/'execution_state.json')['root']);spec=read(root/'protocol.json');verify(spec)
 assert spec['cached_preparation']['cached_valid_cases']==0,'No repeated prepared configuration retries'
 assert 0<=spec['inherited_stage_GPU_seconds']<7200
 file=root/'attempt/resources.jsonl';rows=[json.loads(s)for s in file.read_text().splitlines()]
 stop=[x['stopped']for x in rows if 'stopped'in x];assert len(stop)==1 and stop[0]in ALLOWED
 release=rows[-1];assert release['own_gpu_released']and not release['remaining_own_gpu_pids']
 elapsed=release['cleanup_seconds_after_start'];assert 0<elapsed<360
 assert not(root/'terminal_audit.json').exists()and not(base/'pilot_complete.json').exists()
 assert len(list((root/'initial_prefetch').glob('*.json')))>=8
 return elapsed+spec['inherited_stage_GPU_seconds'],{str(p):sha(p)for p in(actual,base/'execution_state.json',file,root/'failure.json',root/'protocol.json')},spec

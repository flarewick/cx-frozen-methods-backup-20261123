"""One changed all-prepared/grouped queue after actual serial-baseline Rule4 tail."""
import json
from .common import T,read,mapped,sha,verify

def evidence():
 f=T/'outputs/native_calibrated_five100_v31/official_019_p0';actual=f/'actual_host_wait_exit.json';a=read(actual)
 assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 r=mapped(read(f/'execution_state.json')['root']);spec=read(r/'protocol.json');verify(spec)
 assert spec['stage']=='official_019_p0'and spec['current_server_policy']=='governor180_v15'
 assert spec['inherited_stage_GPU_seconds']==643.303 and spec['cached_preparation']['cached_valid_cases']==9
 rows=[json.loads(s)for s in(r/'attempt/resources.jsonl').read_text().splitlines()]
 assert [v['stopped']for v in rows if 'stopped'in v]==['Rule4: rolling180s GPU utilization/VRAM']
 release=rows[-1];assert release['own_gpu_released']and not release['remaining_own_gpu_pids']and release['cleanup_seconds_after_start']==2057.758
 assert all(len(list(r.glob('cases/*/'+condition+'/result.json')))==15 for condition in('M_event_full','TTA'))
 assert len(list(r.glob('cases/*/baseline/result.json')))==14
 assert len(list(r.glob('cases/*/handoff_reference/result.json')))==5
 assert len(list((r/'initial_prefetch').glob('*.json')))==15
 assert not(r/'terminal_audit.json').exists()and not(f/'pilot_complete.json').exists()
 assert 'OIDN Error:'not in(r/'GPU_host.log').read_text(errors='replace')
 g=[json.loads(s)for s in(r/'attempt/governor180.jsonl').read_text().splitlines()]
 assert g[-1]['server_verdict']['reasons']==['Rule4: rolling180s GPU utilization/VRAM']
 total=round(spec['inherited_stage_GPU_seconds']+release['cleanup_seconds_after_start'],3);assert total==2701.061<7200
 files=[actual,f/'execution_state.json',r/'protocol.json',r/'attempt/resources.jsonl',r/'attempt/governor180.jsonl',r/'failure.json']
 return total,{str(p):sha(p)for p in files},spec

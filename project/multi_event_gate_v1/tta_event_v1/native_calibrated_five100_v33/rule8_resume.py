"""Known external-account Rule8 stop, actual release and full elapsed carry; no policy recovery."""
import json
from .common import T,read,mapped,sha,verify

def evidence():
 f=T/'outputs/native_calibrated_five100_v32/official_019_p1';actual=f/'actual_host_wait_exit.json';e=read(actual)
 assert e['exit_code']==1 and e['actual_Popen_wait_return']and e['proc_absent_after_wait']
 r=mapped(read(f/'execution_state.json')['root']);spec=read(r/'protocol.json');verify(spec)
 assert spec['current_server_policy']=='governor180_v15'and spec['stage']=='official_019_p1'
 assert spec['cached_preparation']['cached_valid_cases']==0 and spec['inherited_stage_GPU_seconds']==4497.429
 rows=[json.loads(s)for s in(r/'attempt/resources.jsonl').read_text().splitlines()]
 reason='Rule8: more than one user GPU fails paired180s thresholds'
 assert [v['stopped']for v in rows if 'stopped'in v]==[reason]
 release=rows[-1];assert release['own_gpu_released']and not release['remaining_own_gpu_pids']and release['cleanup_seconds_after_start']==671.721
 assert not(r/'terminal_audit.json').exists()and not(f/'pilot_complete.json').exists()
 assert 'OIDN Error:'not in(r/'GPU_host.log').read_text(errors='replace')
 governance=r/'attempt/governor180.jsonl';g=[json.loads(s)for s in governance.read_text().splitlines()];assert g[-1]['server_verdict']['reasons']==[reason]
 assert len([c for c in g[-1]['fleet']['cards']if c['own_pids']])==2
 assert len(list((r/'initial_prefetch').glob('*.json')))==9
 total=round(spec['inherited_stage_GPU_seconds']+release['cleanup_seconds_after_start'],3);assert total<7200
 files=[actual,f/'execution_state.json',r/'protocol.json',r/'attempt/resources.jsonl',governance,r/'failure.json']
 return total,{str(p):sha(p)for p in files},spec

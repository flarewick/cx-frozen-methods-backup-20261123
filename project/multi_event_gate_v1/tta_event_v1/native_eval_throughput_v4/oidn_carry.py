"""Exact released failed OIDN stage time; zero data reuse from its root."""
import json
from .common import T,read,sha,mapped,verify
def carry():
 base=T/'outputs/native_calibrated_five100_v24/official_007';a=read(base/'actual_host_wait_exit.json')
 assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 root=mapped(read(base/'execution_state.json')['root']);spec=read(root/'protocol.json');verify(spec)
 assert root.name=='716b13c877ff4f65b9d53929e4508e7d'
 log=root/'GPU_host.log';assert 'OIDN Error: out of memory'in log.read_text(errors='replace')
 assert 'Initial overlap actor RGB or command mismatch'in read(root/'failure.json')['traceback']or 'SystemExit: 1'in read(root/'failure.json')['traceback']
 file=root/'attempt/resources.jsonl';rows=[json.loads(s)for s in file.read_text().splitlines()]
 assert not any('stopped'in r for r in rows)
 last=rows[-1];assert last['own_gpu_released']and not last['remaining_own_gpu_pids']and last['cleanup_seconds_after_start']==2606.319
 assert not(root/'terminal_audit.json').exists()and not(base/'pilot_complete.json').exists()
 return 2606.319,{str(p):sha(p)for p in(base/'actual_host_wait_exit.json',base/'execution_state.json',root/'protocol.json',log,file,root/'failure.json')}

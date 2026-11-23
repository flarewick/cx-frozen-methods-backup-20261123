from .common import T,read,mapped,sha,verify

def prior():
 b=T/'outputs/native_seven_gated_dev_v2/dev_gated';a=read(b/'actual_host_wait_exit.json');assert a['exit_code']==0 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 r=mapped(read(b/'execution_state.json')['root']);verify(read(r/'protocol.json'));assert read(r/'terminal_audit.json')['passed']and read(r/'CPU_episode_audit.json')['passed']
 import json
 end=json.loads((r/'attempt/resources.jsonl').read_text().splitlines()[-1]);assert end['own_gpu_released']and not end['remaining_own_gpu_pids']and end['cleanup_seconds_after_start']==1662.266
 return {str(p):sha(p)for p in(b/'actual_host_wait_exit.json',b/'pilot_complete.json',r/'terminal_audit.json',r/'CPU_episode_audit.json',r/'attempt/resources.jsonl')}

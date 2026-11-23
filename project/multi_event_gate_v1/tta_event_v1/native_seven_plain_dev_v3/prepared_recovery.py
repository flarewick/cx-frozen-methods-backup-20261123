"""Read only complete expert/q0 preparation from the specific failed first batch."""
import json
from .common import T,read,mapped,verify

def cache():
    from native_seven_plain_dev_v1 import preparation as original
    b=T/'outputs/native_seven_plain_dev_v1/dev_plain';actual=read(b/'actual_host_wait_exit.json')
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    root=mapped(read(b/'execution_state.json')['root']);old=read(root/'protocol.json');verify(old)
    assert 'Rule4: rolling180s GPU utilization/VRAM'in read(root/'failure.json')['traceback']
    rows=[json.loads(x)for x in(root/'attempt/resources.jsonl').read_text().splitlines()]
    assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']and rows[-1]['cleanup_seconds_after_start']==887.467
    value=original.collect(old,0,old['initial_screen_state'])
    assert value and value['cached_valid_cases']==6 and len(value['task_cache'])==3
    assert not value['primary_policy_results_imported']and not value['old_masks_or_teacher_outputs_imported']
    accepted={p.stem for p in(root/'accepted').glob('*.json')}
    expected={i['case_id']for t in value['task_cache'].values()for i in t['items']};assert expected==accepted and len(expected)==6
    assert all('/cases/'not in path and '/answers/'not in path for path in value['input_sha256'])
    later=T/'outputs/native_seven_plain_dev_v2/dev_plain';stop=read(later/'actual_host_wait_exit.json')
    assert stop['exit_code']==-2 and stop['actual_Popen_wait_return']and stop['proc_absent_after_wait']
    r=mapped(read(later/'execution_state.json')['root']);resource=r/'attempt/resources.jsonl'
    release=json.loads(resource.read_text().splitlines()[-1]);assert release['own_gpu_released']and not release['remaining_own_gpu_pids']and release['cleanup_seconds_after_start']==573.99
    intent=T/'outputs/native_seven_prevent_known_audit_failure_v1/intent.json';v=read(intent)
    assert v['operator_stop']and v['not_server_governor_signal']and v['host_pid']==stop['pid']
    from .common import sha
    for file in(later/'actual_host_wait_exit.json',resource,intent):value['input_sha256'][str(file)]=sha(file)
    return value

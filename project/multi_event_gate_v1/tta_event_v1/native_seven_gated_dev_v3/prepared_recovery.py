"""Read only complete expert/q0 preparation from the specific failed first batch."""
import json
from .common import T,read,mapped,verify

def cache():
    from native_seven_gated_dev_v1 import preparation as original
    b=T/'outputs/native_seven_gated_dev_v1/dev_gated';actual=read(b/'actual_host_wait_exit.json')
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    root=mapped(read(b/'execution_state.json')['root']);old=read(root/'protocol.json');verify(old)
    assert 'native_q0_task_handoff_v3/control.py'in(root/'GPU_host.log').read_text()and 'AssertionError'in(root/'GPU_host.log').read_text()
    rows=[json.loads(x)for x in(root/'attempt/resources.jsonl').read_text().splitlines()]
    assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']and rows[-1]['cleanup_seconds_after_start']==918.294
    value=original.collect(old,0,old['initial_screen_state'])
    assert value and value['cached_valid_cases']==8 and len(value['task_cache'])==4
    assert not value['primary_policy_results_imported']and not value['old_masks_or_teacher_outputs_imported']
    accepted={p.stem for p in(root/'accepted').glob('*.json')}
    expected={i['case_id']for t in value['task_cache'].values()for i in t['items']};assert expected==accepted and len(expected)==8
    assert all('/cases/'not in path and '/answers/'not in path for path in value['input_sha256'])
    return value

"""Actual cleanup accounting; a corrected full batch needs a finite budget."""
import json
from pathlib import Path
from native_policy_execution_v1.common import T, read, sha
from native_onpolicy_recovery_v3.common import mapped, verify
from native_written_server_policy_v1.audit import validate
AUTH = T / 'outputs/native_FIT0_postexit_carry_v1/user_FIT0_14400_authorization.json'


def apply(spec):
    if spec['stage'] != 'fit_000':
        return spec
    base = T / 'outputs/native_onpolicy_arm_collection_v72/fit_000'
    actual = read(base / 'actual_host_wait_exit.json')
    assert actual['exit_code'] == 1 and actual['actual_Popen_wait_return'] and actual['proc_absent_after_wait']
    root = mapped(read(base / 'execution_state.json')['root'])
    protocol = read(root / 'protocol.json')
    verify(protocol)
    assert spec['inherited_stage_GPU_seconds'] == protocol['inherited_stage_GPU_seconds'] == 7205.676
    assert spec['stage_total_GPU_budget_seconds'] == protocol['stage_total_GPU_budget_seconds'] == 10800
    failure = read(base / 'campaign_failure.json')['traceback']
    assert 'native_storage_capture_v1/audit.py' in failure and "spec['simulation_lanes']==10" in failure
    parent = read(root / 'actual_GPU_parent_exit.json')
    assert parent['exit_code'] == 0 and parent['actual_Popen_wait_return']
    terminal = read(root / 'terminal_audit.json')
    verify(terminal)
    assert terminal['passed'] and terminal['actual_host_exit_code'] == terminal['actual_child_exit_code'] == 0
    resource = root / 'attempt/resources.jsonl'
    rows = [json.loads(s) for s in resource.read_text().splitlines()]
    assert validate(rows) > 0
    used = round(protocol['inherited_stage_GPU_seconds'] + rows[-1]['cleanup_seconds_after_start'], 3)
    assert used == 9679.983
    cap = 10800
    files = [base / 'actual_host_wait_exit.json', base / 'campaign_failure.json', root / 'protocol.json',
             root / 'actual_GPU_parent_exit.json', root / 'terminal_audit.json', resource]
    if AUTH.exists():
        auth = read(AUTH)
        verify(auth)
        assert auth['explicit_user_authorization'] is True
        assert auth['previous_total_seconds'] == 10800 and auth['approved_total_seconds'] == 14400
        assert auth['all_prior_GPU_seconds_retained'] == used
        assert auth['scope'] == 'cumulative_additional_FIT0_GPU_diagnostic_and_collection_only'
        assert auth['VAL_and_server_and_scientific_gates_unchanged'] is True
        cap = 14400
        files.append(AUTH)
    spec.update(inherited_stage_GPU_seconds=used, stage_total_GPU_budget_seconds=cap,
                prior_CPU_postexit_failed_batch_excluded=True,
                actual_ARM72_cleanup_seconds_preserved=True)
    spec['input_sha256'].update({str(p): sha(p) for p in files})
    spec['source_sha256'].update({str(p): sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()})
    assert cap - used >= 3600, ('Full fixed25+25 batch needs explicit budget amendment before freeze/GPU',
                               used, cap, cap - used)
    return spec

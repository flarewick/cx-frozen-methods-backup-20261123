"""Explicit normal C5 prefix; no copied results and no failed policy promotion."""
from .common import T, C, read, sha, verify, mapped

IMPORTED_PLAIN_SESSIONS = 1

def folder(group, number):
    namespace = 'native_seven_official_v5' if group == 'plain' and number < IMPORTED_PLAIN_SESSIONS else C.name
    return T / 'outputs' / namespace / (group + '_' + str(number).zfill(3))

def check():
    from .qualification import qualified_stage
    assert IMPORTED_PLAIN_SESSIONS in (1, 2)
    evidence = {}
    for number in range(IMPORTED_PLAIN_SESSIONS):
        base = folder('plain', number)
        root, spec = qualified_stage(base)
        assert spec['version'] == 'native_seven_official_v5' and spec['group'] == 'plain'
        assert spec['stage_index'] == number and spec['episodes_per_task'] == 6 * (number + 1)
        assert spec['official_seed'] == 0 and spec['actual_seed_start'] == 100000
        assert spec['simulation_lanes'] == 8 and spec['early_pairing']
        for p in (base / 'actual_host_wait_exit.json', root / 'protocol.json', root / 'terminal_audit.json', root / 'CPU_episode_audit.json', root / 'attempt/resources.jsonl'):
            evidence[str(p)] = sha(p)
    old = T / 'native_seven_official_v5'
    packaging = {'adoption.py', 'budget.py', 'carry.py', 'checks.py', 'collection_plan.py', 'qualification.py', 'supervision.py', 'preparation.py', 'AGENTS.md'}
    for p in old.rglob('*'):
        if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.md') and p.name not in packaging:
            candidate = C / p.relative_to(old)
            assert candidate.read_text().replace(C.name, old.name).replace('native_governor180_v5','native_governor180_v4').replace('from native_gpu_drained_guard_v4 import guard','from native_gpu_drained_guard_v2 import guard') == p.read_text(), ('Non-packaging source changed', p)
            evidence[str(p)] = sha(p)
            evidence[str(candidate)] = sha(candidate)
    return evidence

def failed_second():
    """Only the bounded-time failure can supply preparation and elapsed time."""
    import json
    assert IMPORTED_PLAIN_SESSIONS == 1
    base = T / 'outputs/native_seven_official_v5/plain_001'
    actual = read(base / 'actual_host_wait_exit.json')
    assert actual['exit_code'] == 1 and actual['actual_Popen_wait_return'] and actual['proc_absent_after_wait']
    root = mapped(read(base / 'execution_state.json')['root'])
    spec = read(root / 'protocol.json'); verify(spec)
    assert spec['version'] == 'native_seven_official_v5' and spec['stage_index'] == 1
    assert read(root / 'failure.json')['traceback'].endswith('RuntimeError: Guarded evaluation reached its bounded time limit\n')
    rows = [json.loads(line) for line in (root / 'attempt/resources.jsonl').read_text().splitlines()]
    assert [r['stopped'] for r in rows if 'stopped' in r] == ['Guarded evaluation reached its bounded time limit']
    release = rows[-1]
    assert release['own_gpu_released'] and not release['remaining_own_gpu_pids']
    assert not (root / 'terminal_audit.json').exists()
    deps = {str(p): sha(p) for p in (base / 'actual_host_wait_exit.json', root / 'protocol.json', root / 'failure.json', root / 'attempt/resources.jsonl')}
    return root, spec, release, deps


def failed_current():
    """Exactly the released C6 startup-Rule8 failure; no partial promotion."""
    import json
    base=T/'outputs/native_seven_official_v6/plain_001'
    actual=read(base/'actual_host_wait_exit.json')
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return'] and actual['proc_absent_after_wait']
    root=mapped(read(base/'execution_state.json')['root']);spec=read(root/'protocol.json');verify(spec)
    assert spec['version']=='native_seven_official_v6' and spec['stage_index']==1
    reason='Rule8: more than one user GPU fails paired180s thresholds'
    assert read(root/'failure.json')['traceback'].endswith('RuntimeError: '+reason+'\n')
    rows=[json.loads(s)for s in(root/'attempt/resources.jsonl').read_text().splitlines()]
    assert [v['stopped'] for v in rows if 'stopped'in v]==[reason]
    release=rows[-1];assert release['own_gpu_released'] and not release['remaining_own_gpu_pids']
    assert not(root/'terminal_audit.json').exists() and not(root/'GPU_episodes_complete.json').exists()
    proof=T/'outputs/native_rule8_startup_v1/CPU_checks_v1.json';p=read(proof);verify(p)
    assert p['passed'] and p['old_failed_actual1_preserved']
    deps={str(p):sha(p)for p in(base/'actual_host_wait_exit.json',root/'protocol.json',root/'failure.json',root/'attempt/resources.jsonl',root/'attempt/governor180.jsonl',proof)}
    return root,spec,release,deps

def failed_coordination():
    import json
    base=T/'outputs/native_seven_official_v7/plain_001'
    actual=read(base/'actual_host_wait_exit.json')
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    root=mapped(read(base/'execution_state.json')['root']);spec=read(root/'protocol.json');verify(spec)
    assert spec['version']=='native_seven_official_v7'and spec['stage_index']==1
    reason='Rule8: more than one user GPU fails paired180s thresholds'
    assert read(root/'failure.json')['traceback'].endswith('RuntimeError: '+reason+'\n')
    rows=[json.loads(s)for s in(root/'attempt/resources.jsonl').read_text().splitlines()]
    assert [r['stopped']for r in rows if 'stopped'in r]==[reason]
    release=rows[-1];assert release['own_gpu_released']and not release['remaining_own_gpu_pids']and release['cleanup_seconds_after_start']==963.156
    assert not(root/'terminal_audit.json').exists()
    deps={str(p):sha(p)for p in(base/'actual_host_wait_exit.json',root/'protocol.json',root/'failure.json',root/'attempt/resources.jsonl',root/'attempt/governor180.jsonl')}
    return root,spec,release,deps

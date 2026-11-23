"""Resource-only overlay; all frozen v14 numerical sources and entry points stay intact."""
import importlib.util
import json
import sys
from pathlib import Path

from policy import L, verify_rules, require_admission
from observer import admission_snapshot
from guard import guarded_command
sys.path.insert(0, str(L / 'official_baseline_v14'))
from common import read, write, sha


def operational_sources():
    return {str(p.relative_to(L)):sha(p) for p in (L/'resource_guard_v15').glob('*.py')}


def main():
    verify_rules()
    proof = read(L/'outputs/resource_guard_v15_CPU_acceptance.json')
    assert proof['passed'] and proof['source_sha256'] == operational_sources()
    for name, digest in read(L/'outputs/official_baseline_CPU_preflight_v14.json')['source_sha256'].items():
        assert sha(L/name) == digest, name
    spec = importlib.util.spec_from_file_location('frozen_resource_host_v14', L/'official_baseline_v14/host.py')
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)

    def check(info):
        verify_rules(); require_admission(admission_snapshot(), info['uuid'])

    def resource_write(path, value):
        # Only resource telemetry field labels change. Episode/RNG writes happen
        # in the unmodified CPU/GPU clients, outside this host module.
        if Path(path).name == 'phase.json' and value.get('phase') == 'running':
            value = dict(value); r = dict(value['resource'])
            r['device_rolling180'] = r.pop('device_cumulative')
            r['process_rolling180'] = r.pop('process_cumulative')
            r['server_window_seconds'] = 180; value['resource'] = r
        write(path, value)
        if Path(path).name == 'launch.json':
            write(Path(path).parent/'operational_guard_v15.json', dict(
                source_sha256=operational_sources(), CPU_proof_sha256=sha(L/'outputs/resource_guard_v15_CPU_acceptance.json'),
                rules_sha256=proof['rules_sha256'], frozen_numerical_runtime='official_baseline_v14',
                retired_cumulative33_server_check=True, recent40_is_target_only=True))
    h.verify_server_rules = verify_rules
    h.inventory = admission_snapshot
    h.require_admission = require_admission
    h.check_admission = check
    h.guarded_command = guarded_command
    h.write = resource_write
    h.main()


if __name__ == '__main__':
    main()

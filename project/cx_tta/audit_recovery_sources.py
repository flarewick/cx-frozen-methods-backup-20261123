"""Read-back audit of real saved source inputs/actions/RNG and causal grouping."""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit import digest
from .eval_diagnostics import array_hash
from .paths import write_json
from .recovery_source_protocol import validate


def observation_signature(path):
    with np.load(path, allow_pickle=False) as z:
        names = ('head', 'right', 'left')
        if any(z[n + '_rgb'].dtype != np.uint8 or z[n + '_rgb'].shape != (240, 320, 3) for n in names):
            raise ValueError('Camera shape or dtype changed')
        if (z['command_state'].dtype != np.float32 or z['command_state'].shape != (14,)
                or not np.isfinite(z['command_state']).all()
                or not np.array_equal(z['command_state'], z['command_state_raw'].astype(np.float32))):
            raise ValueError('Native command-state convention changed')
        return dict(images={n + '_camera': array_hash(z[n + '_rgb']) for n in names},
                    command_state=array_hash(z['command_state_raw']))


def audit(run):
    from .probe_action_replay import load_source, source_runtime
    from .audit_recovery_cache import restore_json_rng
    from .recovery_contract import rng_fingerprint
    if (run / 'failure.json').exists():
        raise ValueError('Incomplete or failed run cannot pass a source audit')
    m = json.loads((run / 'manifest.json').read_text())
    p = validate(json.loads((run / 'protocol.json').read_text()))
    if digest(run / 'protocol.json') != m['protocol_sha256']:
        raise ValueError('Run protocol differs from frozen original')
    source_runtime(run)
    result = json.loads((run / 'result.json').read_text())
    episodes = json.loads((run / 'episodes.json').read_text())
    expected = p['scenes'][m['task']][m['role']]
    if m['seeds'] != expected or [e['seed'] for e in episodes] != expected:
        raise ValueError('Scene selection/order differs from preregistration')
    reports = []
    fingerprints = {}
    for e in episodes:
        folder = run / 'observations' / f"seed_{e['seed']}"
        trace_file = run / 'traces' / f"r0_seed{e['seed']}.jsonl"
        if e['role'] != m['role']:
            raise ValueError('Role mismatch')
        if not e['eligible']:
            if e['saved_query_observations'] or trace_file.stat().st_size:
                raise ValueError('Ineligible scene contains a partial policy trajectory')
            reports.append(dict(seed=e['seed'], eligible=False, reason=e['reason']))
            continue
        _, _, rows = load_source(run, e['seed'])
        entries = json.loads((folder / 'index.json').read_text())
        if len(entries) != len(rows) or len(rows) != e['saved_query_observations']:
            raise ValueError('Missing saved query')
        count = 0
        for i, (entry, row) in enumerate(zip(entries, rows)):
            if (entry['chunk'] != i or row['chunk'] != i or entry['native_count'] != count
                    or entry['previous_chunk'] != (i-1 if i else None)
                    or entry['executed'] != row['executed']):
                raise ValueError('Noncausal history or misaligned action boundary')
            for rel, sha in entry['files'].items():
                path = (run / rel).resolve()
                if not path.is_relative_to(folder.resolve()) or digest(path) != sha:
                    raise ValueError('Changed or escaped saved source file')
                fingerprints[str(path)] = sha
            prefix = folder / f'chunk_{i:03d}'
            if observation_signature(prefix.with_suffix('.npz')) != row['observation_components']:
                raise ValueError('Saved input is not the actual query observation')
            with np.load(prefix.with_suffix('.base.npz'), allow_pickle=False) as z:
                base = z['base_action']
            n = row['executed']
            if (base.shape != (32, 14) or base.dtype != np.float32 or not np.isfinite(base).all()
                    or array_hash(base) != row['base_chunk_sha256']
                    or array_hash(base) != entry['base_sha256']
                    or not np.array_equal(base[:n], np.array(row['base_executed'], np.float32))
                    or row['base_executed'] != row['actions_executed']
                    or np.count_nonzero(row['delta_executed'])):
                raise ValueError('Not an unchanged L0 native action sequence')
            rng = json.loads(prefix.with_suffix('.rng.json').read_text())
            if any(rng_fingerprint(restore_json_rng(rng[k])) != rng[k]['sha256'] for k in ('before', 'after')):
                raise ValueError('RNG serialization did not round-trip')
            if rng['before']['sha256'] == rng['after']['sha256']:
                raise ValueError('Internal policy RNG did not advance')
            count += n
        if count != e['executed_actions']:
            raise ValueError('Native count mismatch')
        reports.append(dict(seed=e['seed'], eligible=True, success=e['success'],
                            chunks=len(entries), native_actions=count,
                            candidate_fork_counts=[x['native_count'] for x in entries if not e['success']
                                and x['native_count'] in p['fork_candidates']['native_counts']]))
    eligible = [e for e in episodes if e['eligible']]
    if (result['status'] != 'complete' or result['attempted_seeds'] != len(episodes)
            or result['eligible_episodes'] != len(eligible)
            or result['successes'] != sum(e['success'] for e in eligible)
            or result['failures'] != sum(not e['success'] for e in eligible)
            or result['saved_observations'] != sum(e['saved_query_observations'] for e in episodes)
            or result['recovery_labels'] != 0):
        raise ValueError('Incorrect outcome accounting')
    for path in [run / name for name in ('manifest.json', 'protocol.json', 'episodes.json', 'result.json',
                                        'resolved_config.json', 'runtime_hashes.json', 'provenance.json')]:
        fingerprints[str(path)] = digest(path)
    for path in sorted((run / 'traces').glob('*.jsonl')):
        fingerprints[str(path)] = digest(path)
    return dict(passed=True, protocol_sha256=m['protocol_sha256'], scenes=reports,
                checked_query_observations=result['saved_observations'], evidence_sha256=fingerprints,
                scope='Serialization/alignment audit only; physical replay and teacher semantics separate')


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('run', type=Path)
    run = p.parse_args().run.resolve()
    report = audit(run)
    write_json(run / 'source_audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'evidence_sha256'}, indent=2))


if __name__ == '__main__':
    main()

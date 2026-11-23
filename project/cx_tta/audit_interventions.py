"""Validate schema-2 branch observations/actions before building training data."""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit import digest
from .deploy import CAMERA_KEYS
from .eval_diagnostics import array_hash
from .intervention_contract import SHARED_BRANCHES, shared_candidates, validate_transition
from .paths import output_path
from .probe_action_replay import compare_physical


def audit_run(run):
    run = output_path(Path(run))
    manifest = json.loads((run / 'manifest.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    if ((run / 'failure.json').exists() or result['status'] != 'complete'
            or result.get('schema_version') != 2 or manifest.get('schema_version') != 2
            or not result['valid_same_state_intervention']
            or set(result['branches']) != set(SHARED_BRANCHES)):
        raise ValueError('Incomplete or unsupported intervention run')
    source = Path(manifest['source'])
    seed, chunk = manifest['source_seed'], manifest['chunk']
    trace = source / 'traces' / f'r0_seed{seed}.jsonl'
    if (digest(trace) != manifest['source_trace_sha256']
            or digest(source / 'manifest.json') != manifest['source_manifest_sha256']):
        raise ValueError('Source provenance changed')
    reference = [json.loads(line) for line in trace.read_text().splitlines()][chunk]
    with np.load(run / 'checkpoint.npz', allow_pickle=False) as data:
        checkpoint = dict(data)
    base = checkpoint['base_actions']
    if not np.array_equal(base, np.asarray(reference['base_executed'], np.float32)):
        raise ValueError('Saved base actions differ from source')
    specification = json.loads((run / 'proposal_spec.json').read_text())
    proposals, expected_spec = shared_candidates(base, specification['caps'], scene_seed=seed,
                                               chunk=chunk, strength=manifest['probe_strength'])
    if expected_spec != specification:
        raise ValueError('Shared direction specification changed')
    proposals.update(base=base, plus=base + checkpoint['delta'], minus=base - checkpoint['delta'])
    summaries = {}
    first = result['branches']['base']
    for name in SHARED_BRANCHES:
        branch = result['branches'][name]
        if (not compare_physical(reference['before'], branch['before'], manifest['tolerance'])['matched']
                or not compare_physical(first['before'], branch['before'], manifest['tolerance'])['matched']
                or branch['observation_signature'] != first['observation_signature']):
            raise ValueError('Branch start mismatch: ' + name)
        with np.load(run / f'transition_{name}.npz', allow_pickle=False) as data:
            arrays = dict(data)
        summary = validate_transition(arrays)
        if summary != branch['transition'] or summary['executed'] != branch['executed']:
            raise ValueError('Transition summary mismatch: ' + name)
        if not np.allclose(arrays['proposed_actions'], proposals[name], atol=1e-7, rtol=0):
            raise ValueError('Proposal mismatch: ' + name)
        if not np.array_equal(arrays['residual_actions'], arrays['proposed_actions'] - base):
            raise ValueError('Residual mismatch: ' + name)
        if not np.array_equal(arrays['proposed_actions'][arrays['executed_mask']],
                              np.asarray(branch['executed_actions'], np.float32).reshape(-1, 14)):
            raise ValueError('Executed actions mismatch: ' + name)
        for short, camera in zip(('head', 'right', 'left'), CAMERA_KEYS):
            if (array_hash(arrays[short + '_rgb_after'])
                    != branch['after_observation_signature']['images'][camera]):
                raise ValueError('Endpoint image mismatch: ' + name)
        summaries[name] = summary
    if (first['executed'] != 32
            or not compare_physical(reference['after'], first['after'], manifest['tolerance'])['matched']):
        raise ValueError('Base endpoint not reproduced')
    return {'schema_version': 2, 'status': 'passed', 'run': str(run), 'branches': summaries,
            'source_observation_exact_match': result['source_observation_exact_match'],
            'scope': 'data contract only; not evidence of recovery or M1 training readiness'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run), indent=2))


if __name__ == '__main__':
    main()

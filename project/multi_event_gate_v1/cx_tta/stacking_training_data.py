"""Readback requirements for expert initialization, never benefit-gate labels."""
import json
from pathlib import Path

from .audit import digest
from .paths import output_path


def qualify_capture(path, task, seed, split, protocol_sha):
    import h5py
    import numpy as np
    from .capture import validate_capture
    report = validate_capture(path)
    with h5py.File(path, 'r') as h:
        if h.attrs['task'] != task or int(h.attrs['seed']) != seed:
            raise ValueError('Capture scene identity differs')
        if int(h.attrs['schema_version']) != 3:
            raise ValueError('Physics timeline schema3 required')
        if h.attrs.get('stacking_split') != split or h.attrs.get('stacking_protocol_sha256') != protocol_sha:
            raise ValueError('Capture split/protocol not sealed')
        if h.attrs.get('instruction_split') != 'seen':
            raise ValueError('Expert language split differs')
        if not h['cx_effect/transition_valid'][:].any():
            raise ValueError('No positive-time transitions')
        if not np.isfinite(h['cx_dense/command'][:]).all():
            raise ValueError('Nonfinite dense expert commands')
    return dict(report, source_sha256=digest(Path(path)), expert_initialization_source_valid=True,
                residual_cache_ready=False, gate_label_ready=False,
                split=split, protocol_sha256=protocol_sha)


def audit_collection(run, protocol_path):
    from .stacking_training_design import load_design
    run = output_path(Path(run))
    p = load_design(protocol_path)
    m = json.loads((run/'manifest.json').read_text())
    result = json.loads((run/'result.json').read_text())
    sha = digest(Path(protocol_path))
    if m['protocol_sha256'] != sha or result['status'] != 'collection_complete':
        raise ValueError('Complete collection and matching frozen protocol required')
    task, split = m['task'], m['split']
    if task not in p['tasks'] or split not in ('smoke', 'adapter_fit', 'adapter_validation'):
        raise ValueError('Wrong collection role')
    spec = p['splits'][split]
    records = json.loads((run/'attempts.json').read_text())
    accepted = [r for r in records if r.get('accepted')]
    if len(accepted) != spec['episodes'] or len({r['seed'] for r in accepted}) != len(accepted):
        raise ValueError('Wrong scene count or duplicate scene')
    for r in accepted:
        if r['split'] != split or not spec['start'] <= r['seed'] < spec['stop']:
            raise ValueError('Scene outside registered split')
        path = output_path(Path(r['hdf5']))
        if not path.is_relative_to(run) or digest(path) != r['capture']['source_sha256']:
            raise ValueError('Capture was changed or escaped its collection')
        qualify_capture(path, task, r['seed'], split, sha)
    return dict(passed=True, task=task, split=split, scenes=len(accepted),
                adapter_raw_data_valid=True, residual_training_ready=False, gate_training_ready=False,
                next='same-observation VLA cache and action/time alignment audit before training')

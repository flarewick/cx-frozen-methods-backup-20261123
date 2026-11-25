"""Save compiler artifacts from a real numerically matched q0 and reuse on the same device.""" 
import hashlib
import json
import os
from pathlib import Path
import time

from .common import read, sha, write


def identity(root):
    spec = read(root/'protocol.json')
    compiler = read(root/'official_compiled_sampler_restore.json')
    return {
        'uuid': os.environ['CUDA_VISIBLE_DEVICES'],
        'runtime': compiler['actual_runtime'],
        'original_artifact': compiler['artifact_sha256'],
        'scientific_protocol': spec['scientific_protocol_sha256'],
        'source_sha256': spec['source_sha256'],
    }


def restore(root):
    import torch
    key = identity(root)
    for attempt in sorted(root.parent.glob('a*'), reverse=True):
        if attempt == root:
            continue
        receipt = attempt/'matched_query_compiler_cache_v151.json'
        if not receipt.is_file():
            continue
        data = read(receipt)
        then=data['identity']
        source_equal=all(key['source_sha256'].get(path)==digest for path,digest in then['source_sha256'].items())
        if not data['numeric_q0_verified'] or not source_equal or any(
                key[k]!=then[k] for k in ('uuid','runtime','original_artifact','scientific_protocol')):
            continue
        source = attempt/'matched_query_compiler_cache_v151.bin'
        q0 = attempt / data['match_path']
        if not source.is_file() or not q0.is_file():
            continue
        if sha(source) != data['cache_sha256'] or sha(q0) != data['match_sha256']:
            raise ValueError('Changed genuine query compiler-cache artifact')
        before = torch.get_rng_state().clone()
        device_before = torch.cuda.get_rng_state().clone()
        info = torch.compiler.load_cache_artifacts(source.read_bytes())
        assert info is not None
        assert torch.equal(before, torch.get_rng_state())
        assert torch.equal(device_before, torch.cuda.get_rng_state())
        write(root/'matched_query_cache_restore_v151.json',
              dict(restored=True,attempt=attempt.name,sha256=data['cache_sha256'],
                   actual_same_device=True,numerical_q0_check_still_required=True,wall_time=time.time()))
        return True
    return False


def publish(root, match):
    import torch
    dest = root/'matched_query_compiler_cache_v151.json'
    if dest.exists() or not match.is_file():
        return
    row = read(match)
    assert row['passed'] and row['base_byte_exact'] and row['head_byte_exact']
    assert row['all_legal_inputs_byte_exact'] and row['RNG_restored_after_reference']
    before = torch.get_rng_state().clone()
    device_before = torch.cuda.get_rng_state().clone()
    artifact = torch.compiler.save_cache_artifacts()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(device_before, torch.cuda.get_rng_state())
    if artifact is None:
        return
    raw, _ = artifact
    blob = root/'matched_query_compiler_cache_v151.bin'
    temporary = blob.with_suffix('.tmp')
    temporary.write_bytes(raw)
    temporary.replace(blob)
    write(dest,dict(identity=identity(root),numeric_q0_verified=True,
                    match_path=str(match.relative_to(root)),match_sha256=sha(match),
                    cache_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),wall_time=time.time()))

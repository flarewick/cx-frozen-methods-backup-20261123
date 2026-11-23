"""Validated local compiler artifacts; writable caches remain per process."""
import dataclasses
import hashlib
import json
from pathlib import Path

SCHEMA = 1
MAX_BYTES = 1024**3


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def read_bundle(directory, expected=None):
    directory = Path(directory)
    paths = [directory/'bundle.json', directory/'artifacts.bin']
    if directory.is_symlink() or any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('Compiler bundle must contain regular local files')
    meta = json.loads(paths[0].read_text())
    if meta.get('schema') != SCHEMA:
        raise ValueError('Unknown compiler bundle schema')
    if digest(canonical(meta['fingerprint'])) != meta['fingerprint_sha256']:
        raise ValueError('Compiler fingerprint digest mismatch')
    if expected is not None and meta['fingerprint'] != expected:
        raise ValueError('Compiler bundle environment/source identity mismatch')
    size = paths[1].stat().st_size
    if not 0 < size <= MAX_BYTES or size != meta['artifact_bytes']:
        raise ValueError('Invalid compiler artifact size')
    data = paths[1].read_bytes()
    if digest(data) != meta['artifact_sha256']:
        raise ValueError('Compiler artifact digest mismatch')
    return meta, data


def write_bundle(directory, fingerprint, data, info, probe):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if not 0 < len(data) <= MAX_BYTES:
        raise ValueError('Empty or oversized compiler artifacts')
    meta = dict(schema=SCHEMA, fingerprint=fingerprint,
                fingerprint_sha256=digest(canonical(fingerprint)),
                artifact_bytes=len(data), artifact_sha256=digest(data),
                cache_info=dataclasses.asdict(info), probe=probe,
                export_phase='original_steady_probe_passed')
    (directory/'artifacts.bin').write_bytes(data)
    (directory/'bundle.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta

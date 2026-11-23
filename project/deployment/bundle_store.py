"""Publish only this account's validated session exports to the local data disk."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from compiler_bundle import read_bundle
import user_container as container

STORE = container.WORK/'compiler_bundles'


def require_complete(meta):
    counts = meta.get('cache_info', {})
    if not counts.get('inductor_artifacts') or not counts.get('autotune_artifacts'):
        raise ValueError('Use the original complete cold export: this diagnostic bundle lacks graph or tuning records')


def publish(source):
    source = Path(source)
    root = container.PROJECT/'multi_event_gate_v1/outputs'
    if source.is_symlink() or not source.resolve().is_relative_to(root.resolve()):
        raise ValueError('Only a retained evaluation session export may be published')
    if source.name != 'deployment_compiler_bundle' or not source.parent.name.startswith('process_') or source.parent.parent.name != 'sessions':
        raise ValueError('Not a compiler export in an evaluation process session')
    meta, data = read_bundle(source)
    require_complete(meta)
    if source.stat().st_uid != os.getuid():
        raise ValueError('Compiler bundle belongs to another account')
    parent = STORE/meta['fingerprint_sha256']
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent/meta['artifact_sha256']
    if destination.exists():
        existing, _ = read_bundle(destination, meta['fingerprint'])
        if existing['artifact_sha256'] != meta['artifact_sha256']:
            raise ValueError('Published compiler artifact differs')
        return destination
    temporary = Path(tempfile.mkdtemp(prefix='.publish_', dir=parent))
    try:
        (temporary/'artifacts.bin').write_bytes(data)
        (temporary/'bundle.json').write_text(json.dumps(meta, indent=2)+'\n')
        read_bundle(temporary, meta['fingerprint'])
        for path in temporary.iterdir():
            path.chmod(0o400)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def validated_store_path(source):
    source = Path(source)
    if source.is_symlink() or not source.resolve().is_relative_to(STORE.resolve()):
        raise ValueError('Only locally published compiler bundles are accepted')
    if source.stat().st_uid != os.getuid():
        raise ValueError('Compiler bundle belongs to another account')
    meta, _ = read_bundle(source)
    require_complete(meta)
    return source.resolve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print('PUBLISHED_COMPILER_BUNDLE', publish(args.source))

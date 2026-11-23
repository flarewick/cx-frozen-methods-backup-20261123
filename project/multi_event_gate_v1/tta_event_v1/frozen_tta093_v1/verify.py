"""Read-only stdlib SHA verification for TTA093_20261101_v1. Never starts CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('manifest.json'))
    args = parser.parse_args()
    value = json.loads(args.manifest.read_text())
    expected = value.pop('identity')
    if identity(value) != expected:
        raise ValueError('Frozen manifest identity mismatch')
    if value['method_id'] != 'TTA093_20261101_v1':
        raise ValueError('Unexpected frozen method identity')
    root = args.project.resolve()
    checked = 0
    for section in ('source_sha256', 'reference_input_sha256', 'frozen_files_sha256'):
        for name, wanted in value[section].items():
            path = Path(name)
            if not path.is_absolute():
                path = root / path
            actual = digest(path)
            if actual != wanted:
                raise ValueError('Frozen file SHA-256 mismatch: ' + name)
            checked += 1
    print(json.dumps(dict(passed=True, method_id=value['method_id'], identity=expected,
                         checked_bindings=checked, source_files=len(value['source_sha256']),
                         reference_inputs=len(value['reference_input_sha256']),
                         frozen_files=len(value['frozen_files_sha256']),
                         GPU_started=False, torch_imported=False, writes_performed=False),
                     ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        print(json.dumps(dict(passed=False, error_type=type(exc).__name__, error=str(exc),
                              GPU_started=False, writes_performed=False), ensure_ascii=False),
              file=sys.stderr)
        raise SystemExit(1)

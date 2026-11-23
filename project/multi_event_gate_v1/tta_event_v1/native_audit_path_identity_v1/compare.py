"""Compare original audit values exactly, identifying files across project mounts."""
from pathlib import Path
from native_policy_execution_v1.common import T, sha
from native_onpolicy_recovery_v3.common import mapped

def normalized(value):
    body = dict(value)
    bindings = {}
    for name, digest in value['input_sha256'].items():
        assert Path(name).is_absolute()
        file = mapped(name).resolve()
        assert file.is_relative_to(T.resolve())
        key = str(file.relative_to(T.resolve()))
        assert key not in bindings, 'Duplicate logical file'
        assert sha(file) == digest, ('Changed audit input', key)
        bindings[key] = digest
    body['input_sha256'] = bindings
    return body

def equal(actual, recorded):
    return normalized(actual) == normalized(recorded)

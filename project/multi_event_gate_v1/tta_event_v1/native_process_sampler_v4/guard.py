"""Runtime34%/24563MiB unchanged; future admission follows the explicit empty-card/under3 amendment.

Future host entries import this module as guard. Historical hosts stay immutable.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

T = Path(__file__).resolve().parents[1]
POLICY = json.loads((Path(__file__).parent / "policy.json").read_text())
MIN_GPU_UTILIZATION = POLICY["minimum_cumulative_GPU_utilization_percent"]
assert MIN_GPU_UTILIZATION == POLICY["minimum_rolling_300s_GPU_utilization_percent"] == 34
MAX_GPU_MEMORY_MIB = POLICY["GPU_memory_MiB"][1]
assert POLICY["GPU_memory_MiB"] == [4096, 24563]
_guard = None


def _implementation():
    global _guard
    if _guard is None:
        source = Path(__file__).parent / "strict_guard.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != POLICY["terminal_guard_source_sha256"]:
            raise RuntimeError("Frozen terminal guard source changed")
        for folder in (T.parents[1] / "deployment", T / "resource_guard_v1", T / "native_nextchunk_v1"):
            sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location("_tta_floor34_vram24563_direct_nvml_v1_guard", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _guard = module
    return _guard


def guarded_command(*args, min_cumulative_utilization=MIN_GPU_UTILIZATION, max_memory_mib=MAX_GPU_MEMORY_MIB, **kwargs):
    if max_memory_mib != MAX_GPU_MEMORY_MIB:
        raise ValueError("Future TTA hosts must explicitly use the authorized24563MiB cap")
    # Accept the old host's explicit40 only as a declared operational override.
    # Both cumulative and rolling300 checks use this one frozen guard argument.
    if min_cumulative_utilization not in (34, 40):
        raise ValueError("Only the authorized34 floor or legacy40 request is supported")
    print("TTA_RESOURCE_POLICY", json.dumps(dict(
        revision=POLICY["revision"], requested_floor=min_cumulative_utilization,
        cumulative_floor=34, rolling_300s_floor=34, max_memory_mib=MAX_GPU_MEMORY_MIB,
        historical_attempts_unchanged=True)), flush=True)
    return _implementation().guarded_command(
        *args, min_cumulative_utilization=MIN_GPU_UTILIZATION, max_memory_mib=MAX_GPU_MEMORY_MIB, **kwargs)


def __getattr__(name):
    return getattr(_implementation(), name)

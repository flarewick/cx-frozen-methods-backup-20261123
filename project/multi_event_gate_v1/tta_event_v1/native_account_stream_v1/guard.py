"""Runtime34%/24563MiB unchanged; future admission follows the explicit empty-card/under3 amendment.

Future host entries import this module as guard. Historical hosts stay immutable.
"""
import ast,copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

T = Path(__file__).resolve().parents[1]
POLICY = json.loads((T / "native_process_sampler_v13/policy.json").read_text())
MIN_GPU_UTILIZATION = POLICY["minimum_cumulative_GPU_utilization_percent"]
assert MIN_GPU_UTILIZATION == POLICY["minimum_rolling_300s_GPU_utilization_percent"] == 34
MAX_GPU_MEMORY_MIB = POLICY["GPU_memory_MiB"][1]
assert POLICY["GPU_memory_MiB"] == [4096, 24563]
_guard = None


def _implementation():
    global _guard
    if _guard is None:
        source = T / "native_process_sampler_v13/strict_guard.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != POLICY["terminal_guard_source_sha256"]:
            raise RuntimeError("Frozen terminal guard source changed")
        for folder in (T.parents[1] / "deployment", T / "resource_guard_v1", T / "native_nextchunk_v1"):
            sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location("_tta_unchanged_guard_persistent_account_v1", source)
        module = importlib.util.module_from_spec(spec)
        tree=ast.parse(source.read_text());original=copy.deepcopy(tree)
        hits=[node for node in ast.walk(tree)if isinstance(node,ast.ImportFrom)and node.module=='native_fast_inventory_v1.inventory']
        assert len(hits)==1 and [x.name for x in hits[0].names]==['inventory']
        hits[0].module='native_account_stream_v1.transport'
        restored=copy.deepcopy(tree)
        for node in ast.walk(restored):
            if isinstance(node,ast.ImportFrom)and node.module=='native_account_stream_v1.transport':node.module='native_fast_inventory_v1.inventory'
        assert ast.dump(restored)==ast.dump(original),'Only fresh account transport import may change'
        exec(compile(tree,str(source),'exec'),module.__dict__)
        from native_metadata_stream_v1.transport import DoctorProxy
        module.gpu_doctor = DoctorProxy(module.gpu_doctor)
        original_release=module.record_release
        def release(*args,**kwargs):
            proxy=module.gpu_doctor
            if proxy.report is None:
                report=args[3] if len(args)>3 else kwargs['report'];proxy.report=Path(report).with_name('NVML_queries.jsonl')
            # Independent fresh account scan only AFTER original stop_job; never resume a failed scientific measurement.
            with proxy.context('release'):
                return original_release(*args,**kwargs)
        module.record_release=release
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
    implementation = _implementation()
    report = args[2] if len(args)>2 else kwargs["report"]
    implementation.gpu_doctor.report = Path(report).with_name("NVML_queries.jsonl")
    from native_process_sampler_v13.metrics import sampling_session
    from .transport import context
    with context(Path(report).with_name('NVML_account_monitor_actual_exit.json')), implementation.gpu_doctor.context('monitor'), sampling_session(kwargs['gpu'],Path(report).with_name('NVML_stream_actual_exit.json')):
        return implementation.guarded_command(
            *args, min_cumulative_utilization=MIN_GPU_UTILIZATION, max_memory_mib=MAX_GPU_MEMORY_MIB, **kwargs)


def __getattr__(name):
    return getattr(_implementation(), name)

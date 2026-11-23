"""Storage-only wrapper around the frozen official structured-action evaluator.

Compiler caches are private process scratch. Large static runtime files are
immutable, content-addressed project objects; policy/rollout code stays unchanged.
"""
from __future__ import annotations

import atexit
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import sys
import tempfile

from cx_tta.audit import digest
from cx_tta.collect_smoke import ROBOTWIN
from .paths import WORKSPACE_ROOT as ROOT, output_path, write_json


CACHE_KEYS = (
    "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
    "CUDA_CACHE_PATH", "NUMBA_CACHE_DIR", "WARP_CACHE_PATH",
    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
)
SOURCE_SUFFIXES = {".py", ".json", ".yml", ".yaml", ".sh"}
STATIC_THRESHOLD = 256 * 1024


def storage_policy():
    return dict(
        schema_version=1, entry="cx_tta.evaluate_structured_low_storage",
        helper_sha256=digest(Path(__file__)),
        cache_storage="private /tmp/cxsa_* per process; never shared",
        cleanup="atexit after compiler worker teardown; SIGTERM exits gracefully",
        tokenizer="verified local scratch copy; no retained per-run copy",
        runtime="ordinary source copies; immutable project objects for static files >256KiB",
        static_object_threshold_bytes=STATIC_THRESHOLD,
        static_object_identity="SHA256, original mode without write bits, original modification time",
        queries="query 0 for each episode seed; diagnostics still generated on every query",
        retained="manifest, provenance, runtime snapshot, hardware/config, episodes/results, first queries",
        frozen_policy_and_rollout="unchanged",
    )


def first_query_recorder(recorder):
    """Filter persistence only, after the original policy generated diagnostics."""
    def record(value):
        if value["query"] == 0:
            return recorder(value)
    return record


def first_query_policy(original):
    class FirstQueryPolicy(original):
        def __init__(self, base, checkpoint, condition, recorder):
            super().__init__(base, checkpoint, condition, first_query_recorder(recorder))
    return FirstQueryPolicy


def _event_path(path, dir_fd=None):
    if isinstance(path, int):
        return None
    path = Path(os.fsdecode(path))
    if not path.is_absolute() and dir_fd is not None and dir_fd >= 0:
        path = Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / path
    return path


def make_write_guard(scratch, objects_dir, protected_inodes):
    """Allow project/private scratch writes, protecting immutable static objects.

This is the original Python audit boundary extended to this process's scratch;
native compiler output is also explicitly redirected. It is not an OS sandbox.
"""
    scratch, objects_dir = Path(scratch).resolve(), Path(objects_dir).resolve()
    project = ROOT.resolve()

    def guard(event, args):
        paths = []
        content_write = False
        if event == "open":
            path, mode, flags = args
            writable = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            if writable:
                paths, content_write = [(path, None)], True
        elif event in {"os.mkdir", "os.chmod", "os.utime"}:
            paths = [(args[0], args[2] if len(args) > 2 else None)]
            if event == "os.utime":
                paths = [(args[0], args[3] if len(args) > 3 else None)]
            content_write = event != "os.mkdir"
        elif event in {"os.remove", "os.rmdir"}:
            paths = [(args[0], args[1] if len(args) > 1 else None)]
        elif event == "os.truncate":
            paths, content_write = [(args[0], None)], True
        elif event in {"os.rename", "os.link"}:
            paths = [(args[0], args[2] if len(args) > 2 else None),
                     (args[1], args[3] if len(args) > 3 else None)]
        elif event == "os.symlink":
            paths = [(args[1], args[2] if len(args) > 2 else None)]
        for raw, dir_fd in paths:
            path = _event_path(raw, dir_fd)
            if path is None:
                if content_write:
                    info = os.fstat(raw)
                    if (info.st_dev, info.st_ino) in protected_inodes:
                        raise PermissionError("Immutable runtime object file descriptor")
                continue
            # Unlink/rename mutate the directory entry, not a symlink's target.
            resolved = (Path(os.path.abspath(path.parent.resolve() / path.name))
                        if event in {"os.remove", "os.rmdir", "os.rename"} else path.resolve())
            if resolved != Path("/dev/null") and not (
                resolved.is_relative_to(project) or resolved.is_relative_to(scratch)):
                raise PermissionError(f"Evaluation write boundary: {resolved}")
            if resolved == objects_dir or resolved.is_relative_to(objects_dir):
                raise PermissionError(f"Immutable runtime object: {resolved}")
            if content_write and path.exists():
                info = path.stat()
                if (info.st_dev, info.st_ino) in protected_inodes:
                    raise PermissionError(f"Immutable runtime object alias: {resolved}")
    return guard


class PrivateEvalStorage:
    def __init__(self, *, runtime_source=ROBOTWIN, objects_dir=None):
        self.runtime_source = Path(runtime_source)
        self.objects_dir = Path(objects_dir or ROOT / "cache/runtime_objects")
        if not self.objects_dir.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Runtime objects must stay in the new project")
        self.protected_inodes = set()
        self.static_objects = {}
        self.run = None
        self.scratch = None
        self.cleaned = False

    def _copy_runtime_file(self, source, destination):
        source, destination = Path(source), Path(destination)
        source_info = source.stat()
        if source.suffix.lower() in SOURCE_SUFFIXES or source_info.st_size <= STATIC_THRESHOLD:
            return shutil.copy2(source, destination)
        sha = digest(source)
        object_mode = stat.S_IMODE(source_info.st_mode) & ~0o222
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        # Permissions/mtime are inode metadata: separate objects when these differ.
        obj = self.objects_dir / f"{sha}_m{object_mode:o}_t{source_info.st_mtime_ns}"
        if obj.is_symlink():
            raise ValueError("Runtime object must not be a symlink")
        if not obj.exists():
            descriptor, temporary = tempfile.mkstemp(prefix=".object_", dir=self.objects_dir)
            os.close(descriptor)
            temporary = Path(temporary)
            try:
                shutil.copyfile(source, temporary)
                if digest(temporary) != sha:
                    raise ValueError("Static runtime source changed while copying")
                temporary.chmod(object_mode)
                os.utime(temporary, ns=(source_info.st_atime_ns, source_info.st_mtime_ns))
                try:
                    os.link(temporary, obj)
                except FileExistsError:
                    pass  # Another evaluation atomically published the same object.
            finally:
                temporary.unlink()
        if not obj.is_file() or digest(obj) != sha:
            raise ValueError("Static runtime object hash mismatch")
        object_info = obj.stat()
        if stat.S_IMODE(object_info.st_mode) != object_mode:
            raise ValueError("Static runtime object permissions mismatch")
        os.link(obj, destination)
        # This workspace filesystem updates mtime on hardlink operations. Restore
        # it after publication/temp unlink and final alias creation, before load.
        os.utime(obj, ns=(source_info.st_atime_ns, source_info.st_mtime_ns))
        info = obj.stat()
        self.protected_inodes.add((info.st_dev, info.st_ino))
        self.static_objects[str(destination)] = dict(object=str(obj), sha256=sha, bytes=info.st_size,
            mode=object_mode, mtime_ns=source_info.st_mtime_ns)
        return str(destination)

    def prepare_runtime(self, run):
        """Keep the frozen runtime layout and bytes, sharing only immutable objects."""
        self.run = output_path(Path(run))
        runtime = self.run / "robotwin"
        runtime.mkdir()
        for name in ("envs", "task_config", "description"):
            shutil.copytree(self.runtime_source / name, runtime / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".cache"),
                copy_function=self._copy_runtime_file)
        (runtime / "assets").symlink_to(self.runtime_source / "assets", target_is_directory=True)
        write_json(self.run / "runtime_hashes.json", dict(
            base_task_sha256=digest(runtime / "envs/_base_task.py"),
            task_config_sha256=digest(runtime / "task_config/demo_randomized.yml")))
        return runtime

    def configure_runtime(self, run, runtime, tokenizer_path):
        self.run = output_path(Path(run))
        runtime, tokenizer_path = Path(runtime).resolve(), Path(tokenizer_path).resolve()
        if not runtime.is_relative_to(self.run):
            raise ValueError("Runtime must be the new run's retained snapshot")
        self.scratch = Path(tempfile.mkdtemp(prefix="cxsa_", dir="/tmp")).resolve()
        environment = {}
        for key in CACHE_KEYS:
            folder = self.scratch / key.lower()
            folder.mkdir()
            environment[key] = str(folder)
        temp = self.scratch / "tmp"
        temp.mkdir()
        if len(os.fsencode(temp)) >= 120:
            raise ValueError("Scratch path too long for NVRTC")
        environment.update({key: str(temp) for key in ("TMPDIR", "TMP", "TEMP")})
        environment["OPENPI_DATA_HOME"] = str(self.scratch / "openpi")
        token_target = self.scratch / "openpi/big_vision/paligemma_tokenizer.model"
        token_target.parent.mkdir(parents=True)
        shutil.copyfile(tokenizer_path, token_target)
        tokenizer_sha = digest(tokenizer_path)
        if digest(token_target) != tokenizer_sha:
            raise ValueError("Scratch tokenizer copy mismatch")
        token_target.chmod(0o444)
        token_info = token_target.stat()
        self.protected_inodes.add((token_info.st_dev, token_info.st_ino))
        os.environ.update(environment)
        tempfile.tempdir = str(temp)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.chdir(runtime)
        sys.path.insert(0, str(runtime))
        write_json(self.run / "compiler_temp.json", dict(
            path=str(temp), run=str(self.run), path_bytes=len(os.fsencode(temp))))
        write_json(self.run / "storage.json", dict(**storage_policy(),
            scratch=str(self.scratch), environment=environment,
            tokenizer_source=str(tokenizer_path), tokenizer_sha256=tokenizer_sha,
            static_objects={str(Path(path).relative_to(runtime)): value
                for path, value in self.static_objects.items()}))
        sys.addaudithook(make_write_guard(self.scratch, self.objects_dir, self.protected_inodes))

    def cleanup(self):
        if self.cleaned or self.scratch is None:
            return
        self.cleaned = True
        result = dict(scratch=str(self.scratch), phase="after_process_workers", removed=False)
        try:
            if self.scratch.is_symlink() or self.scratch.parent != Path("/tmp").resolve() or not self.scratch.name.startswith("cxsa_"):
                raise ValueError("Unsafe private scratch cleanup path")
            if self.scratch.exists():
                shutil.rmtree(self.scratch)
            result["removed"] = True
        except BaseException as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
            print(f"Evaluation scratch cleanup failed: {exc}", file=sys.stderr, flush=True)
        try:
            write_json(self.run / "storage_cleanup.json", result)
        except BaseException as exc:
            print(f"Could not record evaluation scratch cleanup: {exc}", file=sys.stderr, flush=True)


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


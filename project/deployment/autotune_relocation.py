"""Relocate validated PyTorch 2.7 portable tuning records before worker dispatch."""
import functools
from pathlib import Path
import shutil


class AutotuneRelocation:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.copied = 0
        self.original = None

    def stage(self, source_code):
        from torch._inductor.codecache import code_hash, get_path
        from torch._inductor.runtime.autotune_cache import AutotuneCache
        _, dirname, filename = get_path(code_hash(source_code.strip()), 'py')
        key = AutotuneCache._prepare_key(filename) + '.best_config'
        source, destination = self.root/key, Path(dirname)/key
        if not source.exists():
            return False
        if source.is_symlink() or not source.is_file():
            raise ValueError('Restored tuning record must be a regular private file')
        if destination.parent.resolve().parent != self.root:
            raise ValueError('Tuning destination escaped the private cache')
        if destination.is_symlink():
            raise ValueError('Tuning destination is a symlink')
        if destination.exists():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        # This is the original record, including configs_hash and coordesc result.
        # The unchanged PyTorch reader still validates it against live configs.
        shutil.copyfile(source, destination)
        self.copied += 1
        return True

    def install(self):
        from torch._inductor.async_compile import AsyncCompile
        original = AsyncCompile.triton
        relocation = self

        @functools.wraps(original)
        def triton(instance, kernel_name, source_code, device_str='cuda'):
            relocation.stage(source_code)
            return original(instance, kernel_name, source_code, device_str=device_str)

        AsyncCompile.triton = triton
        self.original = original

    def uninstall(self):
        if self.original is not None:
            from torch._inductor.async_compile import AsyncCompile
            AsyncCompile.triton = self.original

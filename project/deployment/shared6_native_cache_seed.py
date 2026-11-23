"""Host-only, checked copies of CPU-precompiled native cache into private scratch."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from compiler_bundle import read_bundle
from finish_scratch import scratch_target
import user_container as container


class Shared6NativeCacheSeed:
    def __init__(self, proof, bundle):
        self.proof = json.loads(Path(proof).read_text())
        meta, _ = read_bundle(bundle)
        record = self.proof
        if not record.get('passed') or record.get('cuda_initialized') is not False:
            raise ValueError('CPU cache proof did not pass')
        for key in ('artifact_sha256','fingerprint_sha256'):
            if record[key] != meta[key]: raise ValueError('CPU cache belongs to another compiler bundle')
        helper = container.HERE/'shared6_native_cache_cpu.py'
        if hashlib.sha256(helper.read_bytes()).hexdigest() != record['helper_sha256']:
            raise ValueError('CPU compiler helper changed since proof')
        if not record['kernels'] or not all(k.get('official_request_exact') for k in record['kernels']):
            raise ValueError('Official compiler request equality was not established')
        if len(record['kernels'])+len(record['unselected'])!=record['source_count'] or record['source_count']<1 or record['original_key_count']<1:
            raise ValueError('Original multi-graph source inventory is incomplete')
        if any(k.get('reason')!='No unique original tuning selection; leave to official runtime' or k.get('configs')==1 for k in record['unselected']):
            raise ValueError('Unselected original kernel was guessed offline')
        hashes=[k['source_sha256'] for k in record['kernels']+record['unselected']]
        if len(hashes)!=len(set(hashes)):
            raise ValueError('Original source inventory repeats a kernel')
        relative=Path(record.get('native_root_relative','.work/native_cpu_inspect/triton_cache_dir'))
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='.work':
            raise ValueError('Native cache root escapes deployment/.work')
        self.root = container.HERE/relative
        self.files = {}
        for relative, expected in record['files'].items():
            path = Path(relative)
            if len(path.parts) != 2 or not re.fullmatch('[A-Z2-7]+',path.parts[0]) or path.name in ('.','..'):
                raise ValueError('Invalid native cache path')
            source = self.root/path
            if source.is_symlink() or source.parent.is_symlink(): raise ValueError('Symlink in CPU cache')
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected: raise ValueError('CPU native cache bytes changed')
            self.files[relative] = data

    def seed(self, session):
        start=time.monotonic()
        storage=json.loads((session/'storage.json').read_text())
        scratch=scratch_target(storage['scratch'],container.WORK/'run-tmp')
        canonical=Path(storage['scratch'])/'triton_cache_dir'
        if storage['environment']['TRITON_CACHE_DIR'] != str(canonical): raise ValueError('Unexpected private Triton cache')
        destination=scratch/'triton_cache_dir'
        if destination.is_symlink() or not destination.is_dir(): raise ValueError('Private cache not ready')
        if list(destination.iterdir()): raise ValueError('Private cache already in use; refusing late seeding')
        stage=Path(tempfile.mkdtemp(prefix='native_seed_',dir=scratch))
        try:
            for relative, data in self.files.items():
                target=stage/relative;target.parent.mkdir(exist_ok=True)
                if target.name.startswith('__grp__') and target.suffix=='.json':
                    data=data.replace(self.proof['native_root'].encode(),str(canonical).encode())
                target.write_bytes(data)
            for source in stage.iterdir():
                os.rename(source,destination/source.name)
        finally: shutil.rmtree(stage)
        result=dict(files=len(self.files),seconds=round(time.monotonic()-start,4),
                    artifact_sha256=self.proof['artifact_sha256'],official_requests_exact=True,unique_selected_sources=len(self.proof['kernels']),unselected_original_runtime_sources=len(self.proof['unselected']))
        (session/'native_cache_seed.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

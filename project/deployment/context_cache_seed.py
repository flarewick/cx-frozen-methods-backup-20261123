"""Checked same-GPU driver/NVRTC cache copies, before private context use."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from compiler_bundle import read_bundle
from finish_scratch import scratch_target
import user_container as container


class ContextCacheSeed:
    def __init__(self,proof,bundle):
        self.proof=json.loads(Path(proof).read_text());meta,_=read_bundle(bundle)
        for key in ('artifact_sha256','fingerprint_sha256'):
            if self.proof[key]!=meta[key]:raise ValueError('Context cache fingerprint differs')
        relative=Path(self.proof['root_relative'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='.work':raise ValueError('Unsafe context cache root')
        root=container.HERE/relative;self.files={}
        for name,expected in self.proof['files'].items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or path.parts[:2] not in (('nvidia','GLCache'),('torch','kernels')):
                raise ValueError('Unsafe context cache entry')
            source=root/path
            if source.is_symlink() or any(p.is_symlink() for p in source.parents if p.is_relative_to(root)):
                raise ValueError('Symlink in context cache')
            data=source.read_bytes()
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Context cache was altered')
            self.files[name]=data
        if not self.files or sum(map(len,self.files.values()))>32*2**20:raise ValueError('Context cache size invalid')

    def seed(self,session):
        record=json.loads((session/'storage.json').read_text());scratch=scratch_target(record['scratch'],container.WORK/'run-tmp')
        if record['environment']['XDG_CACHE_HOME']!=record['scratch']+'/xdg_cache_home':raise ValueError('Unexpected context cache location')
        destination=scratch/'xdg_cache_home'
        if destination.is_symlink() or not destination.is_dir():raise ValueError('Unsafe private context cache')
        tops={Path(name).parts[0] for name in self.files}
        if any((destination/name).exists() for name in tops):raise ValueError('Context cache already used; late seeding rejected')
        stage=Path(tempfile.mkdtemp(prefix='context_seed_',dir=scratch))
        try:
            for name,data in self.files.items():
                target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            for name in tops:os.rename(stage/name,destination/name)
        finally:shutil.rmtree(stage)
        result=dict(files=len(self.files),bytes=sum(map(len,self.files.values())),fingerprint_sha256=self.proof['fingerprint_sha256'])
        (session/'context_cache_seed.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

"""Private source copies and guarded read-only aliases to existing static files.

Only the new native-entry probe uses this storage policy. Frozen evaluators and
their source identities remain unchanged; compiler outputs stay private.
"""
import os
from pathlib import Path
import shutil
import sys

from .audit import digest
from .evaluate_structured_low_storage import PrivateEvalStorage, SOURCE_SUFFIXES, STATIC_THRESHOLD, _event_path
from .paths import write_json


def source_write_guard(source):
    source=Path(source).resolve()
    def guard(event,args):
        paths=[]
        if event=='open':
            path,mode,flags=args
            if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
                paths=[(path,None)]
        elif event in ('os.mkdir','os.chmod','os.utime'):
            paths=[(args[0],args[3] if event=='os.utime' and len(args)>3 else args[2] if len(args)>2 else None)]
        elif event in ('os.remove','os.rmdir'):
            paths=[(args[0],args[1] if len(args)>1 else None)]
        elif event=='os.truncate':paths=[(args[0],None)]
        elif event in ('os.rename','os.link'):
            paths=[(args[0],args[2] if len(args)>2 else None),(args[1],args[3] if len(args)>3 else None)]
        elif event=='os.symlink':paths=[(args[1],args[2] if len(args)>2 else None)]
        for raw,fd in paths:
            path=_event_path(raw,fd)
            if path is None:continue  # Protected static inode descriptors also guarded by parent.
            resolved=(path.parent.resolve()/path.name if event in ('os.remove','os.rmdir','os.rename') else path.resolve())
            if resolved==source or resolved.is_relative_to(source):
                raise PermissionError('Native source runtime is immutable: '+str(resolved))
    return guard


class NativeEntryStorage(PrivateEvalStorage):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.source_metadata={}

    def _copy_runtime_file(self,source,destination):
        source,destination=Path(source),Path(destination);info=source.stat()
        if source.suffix.lower() in SOURCE_SUFFIXES or info.st_size<=STATIC_THRESHOLD:
            return shutil.copy2(source,destination)
        absolute=source.resolve();sha=digest(source)
        destination.symlink_to(absolute)
        self.protected_inodes.add((info.st_dev,info.st_ino))
        self.source_metadata[str(absolute)]={'sha256':sha,'bytes':info.st_size,
            'mtime_ns':info.st_mtime_ns,'mode':info.st_mode}
        self.static_objects[str(destination)]={'object':str(absolute),'sha256':sha,
            'bytes':info.st_size,'mtime_ns':info.st_mtime_ns,'sharing':'read-only guarded symlink; no copy or hardlink'}
        return str(destination)

    def configure_runtime(self,run,runtime,tokenizer_path):
        super().configure_runtime(run,runtime,tokenizer_path)
        sys.addaudithook(source_write_guard(self.runtime_source))
        write_json(self.run/'native_storage.json',{'source':str(self.runtime_source.resolve()),
            'static_source_metadata':self.source_metadata,'large_static_files_copied':0,
            'python_configs':'Private byte-identical copies','source_write_guard_installed':True})

    def verify_source(self):
        for path,expected in self.source_metadata.items():
            actual=Path(path).stat()
            if (actual.st_size!=expected['bytes'] or actual.st_mtime_ns!=expected['mtime_ns']
                or actual.st_mode!=expected['mode'] or digest(Path(path))!=expected['sha256']):
                raise ValueError('Shared static source changed: '+path)
        return {'passed':True,'static_files_verified':len(self.source_metadata),'large_static_files_copied':0}

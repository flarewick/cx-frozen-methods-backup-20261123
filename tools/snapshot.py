"""Pack, verify, and restore the frozen workspace with standard Python."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PART_LIMIT = 40 * 1024**2


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write('\n')


def pack():
    archive = ROOT / 'evidence'
    archive.mkdir(exist_ok=False)
    stage = ROOT / '_evidence_staging'
    with tempfile.TemporaryFile(dir=ROOT) as combined:
        with gzip.GzipFile(fileobj=combined, mode='wb', filename='', compresslevel=5, mtime=0) as stream:
            with tarfile.open(fileobj=stream, mode='w|') as tar:
                for path in sorted(stage.rglob('*')):
                    if path.is_file():
                        tar.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)
        combined.seek(0)
        parts=[]
        while block := combined.read(PART_LIMIT):
            path=archive / f'evidence.tar.gz.part{len(parts):04}'
            with path.open('xb') as out:
                out.write(block)
            parts.append(dict(path=path.name, bytes=len(block), sha256=hashlib.sha256(block).hexdigest()))
    write(archive / 'parts.json', parts)


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.stream = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.stream = path.open('rb')
            size = self.stream.readinto(buffer)
            if size:
                return size
            self.stream.close()
            self.stream = None

    def close(self):
        if self.stream:
            self.stream.close()
        super().close()


@contextmanager
def verified_parts():
    paths=[]
    for row in json.loads((ROOT / 'evidence/parts.json').read_text()):
        path = ROOT / 'evidence' / row['path']
        assert path.parent == ROOT / 'evidence' and path.is_file()
        assert path.stat().st_size == row['bytes'] and sha(path)==row['sha256'], path
        paths.append(path)
    with io.BufferedReader(PartsReader(paths)) as source:
        yield source


def verify_or_unpack(destination=None):
    index = json.loads((ROOT / 'inventory.json').read_text())
    rows = {v['path']:v for v in index['files']}
    assert len(rows) == len(index['files'])
    for name,row in rows.items():
        p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts and '.git' not in p.parts, name
        if row['layer'] != 'git':
            continue
        src=ROOT / 'project' / p
        assert not src.is_symlink() and src.is_file() and src.stat().st_size==row['bytes'] and sha(src)==row['sha256'],name
        if destination is not None:
            target=destination / p
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,target)
    expected={name for name,value in rows.items() if value['layer']=='evidence'}
    remaining=expected.copy()
    with verified_parts() as source:
        with tarfile.open(fileobj=source,mode='r|gz') as tar:
            for member in tar:
                name=member.name
                assert member.isfile() and name in remaining, name
                row=rows[name]
                assert member.size==row['bytes'],name
                original=tar.extractfile(member)
                if destination is None:
                    digest=hashlib.file_digest(original,'sha256').hexdigest()
                else:
                    path=destination / name
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('xb') as restored:
                        shutil.copyfileobj(original,restored)
                    os.chmod(path,member.mode & 0o755)
                    os.utime(path,(member.mtime,member.mtime))
                    digest=sha(path)
                assert digest==row['sha256'],name
                remaining.remove(name)
    assert not remaining,sorted(remaining)
    return dict(passed=True,files=len(rows),git_files=len(rows)-len(expected),evidence_files=len(expected),
                unresolved_bindings=len(index['unresolved_bindings']),destination=str(destination) if destination else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('pack','verify','restore'))
    parser.add_argument('destination',type=Path,nargs='?')
    args=parser.parse_args()
    if args.action=='pack':
        pack()
        return
    if args.action=='restore':
        if args.destination is None:
            parser.error('restore requires a new destination')
        path=args.destination.absolute()
        if path.exists() or path.is_symlink():
            parser.error('destination already exists')
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.cx-restore-',dir=path.parent) as temp:
            working=Path(temp)/'project'
            working.mkdir(mode=0o700)
            result=verify_or_unpack(working)
            if path.exists() or path.is_symlink():
                raise FileExistsError(path)
            # Claim an empty, new directory exclusively before moving its verified files.
            path.mkdir(mode=0o700)
            for item in working.iterdir():
                item.rename(path/item.name)
        result['destination']=str(path)
    else:
        result=verify_or_unpack()
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()

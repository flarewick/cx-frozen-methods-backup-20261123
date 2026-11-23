from pathlib import Path
import hashlib
import json

CODE=Path(__file__).resolve().parents[1]
TTA=CODE.parent
PROJECT=TTA.parents[1]
OUTPUT=TTA/'outputs/native_feedback_confirm_v5'
CANONICAL=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def output(path):
    path=Path(path).resolve()
    if path==OUTPUT.resolve() or not path.is_relative_to(OUTPUT.resolve()):raise ValueError('New factor outputs only')
    return path

def write(path,value):
    path=output(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def verify(spec):
    for field in ('input_sha256','source_sha256'):
        for key,sha in spec[field].items():
            path=Path(key) if Path(key).is_absolute() else PROJECT/key
            if digest(path)!=sha:raise ValueError('Frozen dependency changed: '+key)

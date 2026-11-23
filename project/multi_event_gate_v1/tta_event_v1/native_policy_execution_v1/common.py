from pathlib import Path
import json,hashlib
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,data):
 p=Path(p).resolve();assert p.is_relative_to((T/'outputs').resolve())
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

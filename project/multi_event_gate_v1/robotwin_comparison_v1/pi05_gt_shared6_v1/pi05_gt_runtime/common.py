from pathlib import Path
import json,hashlib,os
C=Path(__file__).resolve().parents[2];W=C.parent;T=W/'tta_event_v1';P=W.parent;O=C/'pi05_gt_shared6_v1/native_runs'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(p,x):
 p=Path(p).resolve();assert p.is_relative_to(O.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def replace(p,x):
 p=Path(p).resolve();assert p.is_relative_to(O.resolve());p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def verify(spec):
 for field in ['source_sha256','input_sha256']:
  for p,h in spec[field].items():assert sha(p)==h,('Changed qualified source/input',p)

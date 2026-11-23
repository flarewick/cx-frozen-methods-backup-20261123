from pathlib import Path
import json,os,hashlib
T=Path(__file__).resolve().parents[1];P=T.parents[1];C=Path(__file__).resolve().parent;O=T/'outputs/native_batched_prefix_graph_v1'
from native_cross_task100_v5.common import verify,mapped,identity

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p).resolve();assert p.is_relative_to(O.resolve());p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.publish')
 with tmp.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
 try:os.link(tmp,p)
 finally:tmp.unlink()

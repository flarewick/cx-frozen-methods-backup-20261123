from pathlib import Path
import json,os
from native_policy_execution_v1.common import T,read,sha
from native_onpolicy_recovery_v3.common import mapped,verify,identity
C=Path(__file__).resolve().parent;P=T.parents[1];O=T/'outputs/native_object_state_storage_v2'
def write(p,v):
 p=Path(p).resolve();assert p.is_relative_to(O.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.'+str(os.getpid())+'.publish')
 with tmp.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
 try:os.link(tmp,p)
 finally:tmp.unlink()
def replace(p,v):
 p=Path(p).resolve();assert p.is_relative_to(O.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(v,allow_nan=False));os.replace(tmp,p)

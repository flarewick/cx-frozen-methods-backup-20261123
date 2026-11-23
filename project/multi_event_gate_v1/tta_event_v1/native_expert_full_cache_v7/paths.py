from pathlib import Path
import hashlib,json
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1];O=T/'outputs/native_expert_full_cache_v7'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(p,x):
 p=Path(p).resolve();assert p!=O.resolve()and p.is_relative_to(O.resolve())
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify(spec):
 for field in('source_sha256','input_sha256'):
  for p,h in spec[field].items():assert sha(Path(p)if Path(p).is_absolute()else P/p)==h,p
def dataset(spec,gpu_uuid):
 assert gpu_uuid==spec['required_gpu_uuid']
 m=dict(protocol_identity=spec['identity'],gpu_uuid=gpu_uuid,case_ids=[r['case_id']for r in spec['scenes']]);m['identity']=identity(m)
 root=O/'collections'/m['identity'][:24]
 if(root/'manifest.json').exists():assert read(root/'manifest.json')==m
 else:write(root/'manifest.json',m)
 return root,m

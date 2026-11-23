from pathlib import Path
import hashlib,json
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1];O=T/'outputs/native_trained_runtime_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(p,x):
 p=Path(p).resolve();assert p!=O.resolve()and p.is_relative_to(O.resolve())
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify(spec):
 for name in('source_sha256','input_sha256'):
  for p,h in spec[name].items():
   p=Path(p)if Path(p).is_absolute()else P/p
   assert sha(p)==h,('Changed native episode dependency',str(p))
def dataset(spec,gpu_uuid):
 assert gpu_uuid==spec['required_gpu_uuid']
 manifest=dict(version=1,protocol_identity=spec['identity'],gpu_uuid=gpu_uuid,case_ids=[r['case_id']for r in spec['scenes']])
 manifest['identity']=identity(manifest);root=O/'collections'/manifest['identity'][:24]
 if(root/'manifest.json').exists():assert read(root/'manifest.json')==manifest
 else:write(root/'manifest.json',manifest)
 return root,manifest

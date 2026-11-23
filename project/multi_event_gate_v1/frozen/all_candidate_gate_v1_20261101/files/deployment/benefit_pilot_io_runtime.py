"""Lossless JSON formatting releases GIL; policy/physics sources unchanged."""
import ast,json,sys,types
from pathlib import Path
import benefit_pilot_lazy_runtime as lazy
from benefit_pilot_protocol import D,sha
EXTRA=('benefit_pilot_io_runtime.py','benefit_json_writer.cpp','build_benefit_json_cpu.py','.work/benefit_json/build.json')
def install_sources():
 lazy.install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_pilot_io_sources',False):return
 before=backend.sources;build=json.loads((D/'.work/benefit_json/build.json').read_text());binary=D/'.work/benefit_json'/Path(build['binary']).name
 backend.sources=lambda:dict(before(),**{n:sha(D/n)for n in(*EXTRA,str(binary.relative_to(D)))})
 backend._pilot_io_sources=True

def load_writer():
 build=json.loads((D/'.work/benefit_json/build.json').read_text());binary=D/'.work/benefit_json'/Path(build['binary']).name
 if sha(D/'benefit_json_writer.cpp')!=build['source_sha256']or sha(binary)!=build['binary_sha256']:raise RuntimeError('Passive JSON reader build changed')
 sys.path.insert(0,str(binary.parent));import cx_benefit_json
 return cx_benefit_json

def install():
 lazy.install();install_sources()
 import benefit_pilot_trace as trace
 native=load_writer()
 def dumps(record,*,allow_nan):
  if allow_nan is not False:raise ValueError('Only strict passive record encoding allowed')
  return native.dumps(record)
 factory=trace.chunk_recorder;namespace=dict(factory.__globals__);namespace['json']=types.SimpleNamespace(dumps=dumps)
 trace.chunk_recorder=types.FunctionType(factory.__code__,namespace,factory.__name__,factory.__defaults__,factory.__closure__)

def main():
 from shared6_eval_prepared import main as prelude
 def edit(node):
  index=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='torch.set_num_threads')
  node.body[index+1:index+1]=ast.parse("__import__('benefit_pilot_io_runtime').install()").body
 lazy.fast.original.transform(prelude,edit)()
if __name__=='__main__':main()

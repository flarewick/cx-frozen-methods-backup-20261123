"""Lossless recording acceleration; original physics/policy/cache sources intact."""
import ast,gzip,hashlib,inspect,json,sys,textwrap,struct
from pathlib import Path
import benefit_pilot_runtime as original
from benefit_pilot_protocol import D,sha
EXTRA=('benefit_pilot_fast_runtime.py','benefit_native_contacts.cpp','build_benefit_contacts_cpu.py','.work/benefit_contacts/build.json')
def install_sources():
 original.install_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_pilot_fast_sources',False):return
 before=backend.sources;build=json.loads((D/'.work/benefit_contacts/build.json').read_text());binary=D/'.work/benefit_contacts'/Path(build['binary']).name;relative=str(binary.relative_to(D))
 backend.sources=lambda:dict(before(),**{n:sha(D/n)for n in(*EXTRA,relative)});backend._pilot_fast_sources=True

def load_reader():
 import sapien
 build=json.loads((D/'.work/benefit_contacts/build.json').read_text());binary=D/'.work/benefit_contacts'/Path(build['binary']).name
 if sha(D/'benefit_native_contacts.cpp')!=build['source_sha256']or sha(binary)!=build['binary_sha256']:raise RuntimeError('Contact reader build changed')
 sys.path.insert(0,str(binary.parent));import cx_benefit_contacts
 return cx_benefit_contacts

def point_key(p):return struct.pack('!10d',*p['position'],*p['normal'],*p['impulse'],p['separation'])
def canonical_contacts(rows):
 # Canonical ordering only: every original pair and signed point value remains.
 # Binary keys avoid formatting every point as JSON several times per action.
 for row in rows:row['points'].sort(key=point_key)
 rows.sort(key=lambda r:(tuple(r['bodies']),tuple(point_key(p)for p in r['points'])))
 return rows

def install():
 original.install();install_sources()
 import benefit_pilot_trace as trace
 native=load_reader()
 def contacts(task):
  return canonical_contacts(native.contacts(task.scene.physx_system.get_contacts()))
 trace.contacts=contacts
 # Reuse the exact original capture, action-count and signed-array checks;
 # only the NPZ storage compression and gzip level differ on disk.
 node=ast.parse(textwrap.dedent(inspect.getsource(trace.chunk_recorder))).body[0];changed=0
 for call in[n for n in ast.walk(node)if isinstance(n,ast.Call)]:
  if ast.unparse(call.func)=='np.savez_compressed':call.func.attr='savez';changed+=1
  if ast.unparse(call.func)=='gzip.open':call.keywords.append(ast.keyword(arg='compresslevel',value=ast.Constant(1)));changed+=1
 if changed!=2:raise RuntimeError('Original trace storage boundaries changed')
 ns=dict(trace.chunk_recorder.__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(D/'benefit_pilot_fast_runtime.py'),'exec'),ns);trace.chunk_recorder=ns['chunk_recorder']

def main():
 from shared6_eval_prepared import main as prelude
 def edit(node):
  index=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='torch.set_num_threads')
  node.body[index+1:index+1]=ast.parse("__import__('benefit_pilot_fast_runtime').install()").body
 original.transform(prelude,edit)()
if __name__=='__main__':main()

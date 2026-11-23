"""Preserve every original physics tick/recorder call, releasing GIL only in PhysX."""
import ast,hashlib,inspect,json,sys,textwrap,weakref
from pathlib import Path
D=Path(__file__).resolve().parent

def load():
 import sapien
 proof=json.loads((D/'reports/target_collection_step_cpu_20261023.json').read_text());build=json.loads((D/'.work/collection_step/build.json').read_text());binary=Path(build['binary'])
 if not proof['passed']or proof['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()or proof['binary_sha256']!=hashlib.sha256(binary.read_bytes()).hexdigest()or build['source_sha256']!=hashlib.sha256((D/'collection_step.cpp').read_bytes()).hexdigest():raise RuntimeError('Collection physics CPU proof stale')
 sys.path.insert(0,str(binary.parent));import cx_collection_step
 return cx_collection_step

def install(task,gate,native):
 scene=task.scene
 expected=ast.parse('def step(self):\n    self.physx_system.step()')
 actual=ast.parse(textwrap.dedent(inspect.getsource(scene.step.__func__)))
 if ast.dump(actual,include_attributes=False)!=ast.dump(expected,include_attributes=False):raise RuntimeError('Original Scene.step changed')
 scene_ref=weakref.ref(scene)
 def step():
  current=scene_ref()
  if current is None:raise RuntimeError('Physics scene already released')
  with gate.read():native.step(current.physx_system)
 scene.step=step

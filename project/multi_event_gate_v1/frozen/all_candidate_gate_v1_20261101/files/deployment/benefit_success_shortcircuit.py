"""Exact original lift success predicate; reject impossible height first."""
import ast,inspect,json,textwrap,types
from pathlib import Path
from benefit_stage_protocol import ROOT,sha

def build(original):
 node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
 if ast.unparse(node.body[0])!='pot_pose = self.pot.get_pose()':raise RuntimeError('Original pose read changed')
 value=node.body[-1].value
 if not isinstance(value,ast.BoolOp)or not isinstance(value.op,ast.And)or ast.unparse(value.values[0])!='pot_pose.p[2] > 0.82':raise RuntimeError('Original necessary height condition changed')
 node.body[1:1]=ast.parse('if not (pot_pose.p[2] > 0.82):\n return pot_pose.p[2] > 0.82').body
 ns=dict(original.__globals__)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),inspect.getsourcefile(original),'exec'),ns)
 return ns[node.name]

def install(task):
 if type(task).__name__!='lift_pot'or getattr(task,'_benefit_shortcircuit',False):return
 proof=json.loads((ROOT/'success_shortcircuit_cpu_proof.json').read_text())
 if not proof['passed']or proof['helper_sha256']!=sha(__file__):raise RuntimeError('Success shortcircuit proof stale')
 original=task.check_success.__func__
 if sha(inspect.getsourcefile(original))!=proof['task_source_sha256']:raise RuntimeError('Original lift task changed')
 for method in (task.robot._trans_endpose,task.pot.get_point,task.pot.get_pose,original.__globals__['get_face_prod']):
  source=Path(inspect.getsourcefile(method));expected=proof['pure_getter_sources'].get(source.name)
  if not expected or sha(source)!=expected:raise RuntimeError('Unverified getter side effects')
 task.check_success=types.MethodType(build(original),task);task._benefit_shortcircuit=True

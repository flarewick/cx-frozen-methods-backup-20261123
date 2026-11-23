"""Recreate the original expert's deterministic success-bookkeeping prefix.

No expert actions, trajectories, labels, or features enter the policy. For
open_laptop, its original arm_tag initialization is needed by check_success in
all independently recreated scenes, including lanes with shared screening rows.
"""
import ast,copy,functools,hashlib,inspect,json,textwrap,types
from pathlib import Path
D=Path(__file__).resolve().parent

def initializer(method):
    function=method.__func__;tree=ast.parse(textwrap.dedent(inspect.getsource(function)));fn=tree.body[0]
    if fn.name!='play_once' or len(fn.body)<3:raise RuntimeError('Original laptop initialization changed')
    nodes=fn.body[:3]
    expected=ast.parse("face_prod = get_face_prod(self.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])\narm_tag = ArmTag('left' if face_prod > 0 else 'right')\nself.arm_tag = arm_tag")
    if ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)!=ast.dump(expected,include_attributes=False):raise RuntimeError('Unexpected original task bookkeeping prefix')
    fn.body=copy.deepcopy(nodes);fn.name='_original_task_bookkeeping_prefix';ast.fix_missing_locations(tree);namespace={}
    exec(compile(tree,function.__code__.co_filename,'exec'),function.__globals__,namespace)
    return types.MethodType(namespace[fn.name],method.__self__)

def install_task(task,verify_proof=True):
    if type(task).__name__!='open_laptop' or getattr(task,'_cx_original_bookkeeping_installed',False):return
    if verify_proof:
        proof=json.loads((D/'reports/target_shared6_task_bookkeeping_cpu_20261024.json').read_text())
        if not proof.get('passed')or proof['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()or proof['task_source_sha256']['open_laptop']!=hashlib.sha256(Path(inspect.getsourcefile(type(task))).read_bytes()).hexdigest():raise RuntimeError('Original task bookkeeping CPU proof missing/stale')
    prefix=initializer(task.play_once);original=task.setup_demo
    @functools.wraps(original)
    def setup(*args,**kwargs):
        result=original(*args,**kwargs);prefix();return result
    task.setup_demo=setup;task._cx_original_bookkeeping_installed=True

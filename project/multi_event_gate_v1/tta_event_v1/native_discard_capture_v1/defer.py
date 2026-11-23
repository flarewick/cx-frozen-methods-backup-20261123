"""Omit only an unread picture capture; keep the original update and all guards."""
import ast,copy,inspect,sys,textwrap,weakref
TASKS={}

def configure(task,enabled,case_id):
 task._cx_defer_unread_capture=bool(enabled)
 task._cx_deferred_capture_count=0
 TASKS[case_id]=weakref.ref(task)

def build(original):
 node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0];before=copy.deepcopy(node)
 positions=[i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and ast.unparse(n.value)=='task.cameras.update_picture()']
 assert len(positions)==1
 index=positions[0];removed=node.body.pop(index);restored=copy.deepcopy(node);restored.body.insert(index,removed)
 assert ast.dump(restored)==ast.dump(before),'Original modality/update/RNG path changed'
 namespace=dict(original.__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)
 reduced=namespace[node.name]
 def discard(task):
  if not getattr(task,'_cx_defer_unread_capture',False):return original(task)
  result=reduced(task)
  task._cx_deferred_capture_count+=1
  return result
 return discard

def install():
 from cx_tta_feedback_confirm5 import throughput
 original=throughput.discard_observation;candidate=build(original);changed=[]
 for name,module in list(sys.modules.items()):
  if module is None or not(name.startswith('native_')or name.startswith('cx_tta')):continue
  namespace=vars(module)
  if namespace.get('discard_observation')is original:
   namespace['discard_observation']=candidate;changed.append(name)
 assert 'cx_tta_feedback_confirm5.throughput'in changed
 return changed

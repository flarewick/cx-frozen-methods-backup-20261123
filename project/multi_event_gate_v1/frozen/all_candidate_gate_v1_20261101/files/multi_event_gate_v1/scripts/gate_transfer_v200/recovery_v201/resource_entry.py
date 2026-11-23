"""Resource-only CPU proof path alias; all frozen code and scientific sources exact."""
import json,os,runpy,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;sys.path.insert(0,str(PARENT));sys.path.insert(0,str(HERE))
import entry;entry.install()
import benefit_followup_protocol_v2 as p
class ProofHome(type(p.HOME)):
 def __truediv__(self,key):
  if str(key)=='deferred_finalize_cpu_proof.json':return p.ROOT/'deferred_finalize_cpu_proof.json'
  return super().__truediv__(key)
p.HOME=ProofHome(p.HOME)
# Preserve every other lookup under HOME; no model, gate, task, scene or source-table changes.
assert p.HOME==PARENT.parents[2]/'multi_event_gate_v1/outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4'
for name in ('fixed_ensemble_TRAIN472_v118','namespace_amendment.json','supervision_growth_canonical_v96/train2'):
 assert p.HOME/name==Path(str(p.HOME))/name
assert p.HOME/'deferred_finalize_cpu_proof.json'==p.ROOT/'deferred_finalize_cpu_proof.json'
# New tasks have no historical TRAIN qualification in the old two-task corpus.
# The discarded L0/Zzero parity probe uses an archived index0 expert-qualified seed only.
# It executes no policy actions; formal seeds and formal setup/hash checks are unchanged.
REFERENCES={task:p.HOME/'cup_bowls_discarded_probe_references_v202'/(task+'.json')for task in p.TASKS}
probe_rows={task:json.loads(path.read_text())['qualified_row']for task,path in REFERENCES.items()}
if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
 for path in REFERENCES.values():
  reference=json.loads(path.read_text());assert p.sha(Path(reference['source']))==reference['source_sha256']and reference['no_policy_success_outcome_imported']
assert all(row['seed']==100000 and row['expert_plan_success']and row['expert_success']for row in probe_rows.values())
import benefit_followup_mixed_core_v3 as mixed
original_probe=mixed.probe_first
mixed.probe_first=lambda task:dict(probe_rows[task])if task in probe_rows else original_probe(task)
# Policy uses the original uninstrumented base; expert qualification uses the
# instrumented base. Verify either exact file and the SAME update-render AST.
import ast,functools,inspect
import shared6_render_sync as visual
certified=visual.install;original_visual=inspect.unwrap(certified)
proof=json.loads((p.ROOT/'transfer_task_runtime_CPU.json').read_text())
plan=json.loads((p.TRAINING/'training_plan.json').read_text())
from evaluate import host
protocol=json.loads((host(plan['dataset'])/'protocol.json').read_text());rawbase=Path(protocol['runtime_source'])/'envs/_base_task.py'
if not rawbase.exists():rawbase=p.D/'.work/rootfs'/rawbase.relative_to('/')
assert p.sha(rawbase)==proof['raw_base_sha256']
def update_ast(path):return ast.dump(next(n for n in ast.walk(ast.parse(path.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='_update_render'),include_attributes=False)
reference_update=update_ast(rawbase)
@functools.wraps(original_visual)
def certified_policy(task,note=lambda *a,**kw:None,verify_proof=True):
 name=type(task).__name__
 if name not in proof['task_source_sha256']:return certified(task,note,verify_proof)
 assert p.sha(Path(inspect.getsourcefile(type(task))))==proof['task_source_sha256'][name]
 file=Path(inspect.getsourcefile(inspect.unwrap(task._update_render)))
 assert p.sha(file)in (proof['raw_base_sha256'],proof['instrumented_base_sha256'])
 assert update_ast(file)==reference_update
 assert p.sha(p.D/'reports/target_shared6_render_sync_cpu_20261024.json')==proof['original_renderer_CPU_proof_sha256']
 assert p.sha(p.D/'reports/target_collection_control_cpu_20261024.json')==proof['original_control_CPU_proof_sha256']
 result=original_visual(task,note,verify_proof=False)
 __import__('success_shortcircuit_v206').install(task)
 return result
visual.install=certified_policy
# Passive trace extension only: old logger assumes every non-pot task is a bin.
# New tasks record their unchanged official predicate solely under offline_only.
if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')=='1':
 import benefit_pilot_trace as trace
 old_parts=trace.success_parts
 for task_name in p.TASKS:
  source=rawbase.parent/(task_name+'.py');tree=ast.parse(source.read_text());fn=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='check_success')
  assert not any(isinstance(n,(ast.AugAssign,ast.Delete,ast.Yield,ast.Await))for n in ast.walk(fn))
  assert all(isinstance(target,ast.Name)or(isinstance(target,ast.Tuple)and all(isinstance(t,ast.Name)for t in target.elts))for n in ast.walk(fn)if isinstance(n,ast.Assign)for target in n.targets)
  assert not any(isinstance(n,ast.Call)and any(v in ast.unparse(n.func)for v in ('random','seed','render','step','set_'))for n in ast.walk(fn))
 def parts(task):
  if type(task).__name__ in p.TASKS:return dict(official_success=bool(task.check_success()),task=type(task).__name__,recording_only=True)
  return old_parts(task)
 trace.success_parts=parts
__import__('parallel_v203').install()
import benefit_growth_runtime as runtime
old_sources=runtime.install_sources
def install_resource_manifest():
 old_sources()
 import shared6_eval_backend as backend
 if getattr(backend,'_transfer_probe202',False):return
 old=backend.decorate
 def decorate(*a,**kw):
  raw=old(*a,**kw)
  raw['deployment_transfer_resource_scope_v202']=dict(helper_sha256=p.sha(Path(__file__)),discarded_probe_seed=100000,discarded_probe_policy_actions=0,formal_scene_seeds_unchanged=True,control_branch_time_order='independent_A0_A1_B_may_overlap_G_after_full_control_audit',old_protocol_sequential_descriptor_explicitly_superseded=True,CPU_proof_pointer_only_other_path_change=True,original_gate_and_adapter_unchanged=True,archived_expert_reference_sha256={task:p.sha(path)for task,path in REFERENCES.items()})
  raw['source_sha256'].update({str(f):p.sha(f)for f in [Path(__file__),HERE/'parallel_v203.py',HERE/'balanced_v207.py',HERE/'success_shortcircuit_v206.py',p.ROOT/'success_shortcircuit_CPU_v206.json',p.ROOT/'balanced_task_admission_CPU_v207.json',*REFERENCES.values()]})
  from cx_tta.evaluation_resume import identity
  raw['evaluation_identity']=identity(raw);return raw
 backend.decorate=decorate;backend._transfer_probe202=True
runtime.install_sources=install_resource_manifest
if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
 import user_container as c
 from evaluate import CANONICAL_PROJECT
 before=c.command;old=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_v200/entry.py');new=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_transfer_v200/recovery_v201/resource_entry.py')
 def route(*a,**kw):
  cmd=before(*a,**kw)
  for i,s in enumerate(cmd):
   if s==old:cmd[i]=new
   elif isinstance(s,str)and "__import__('transfer_entry' if 'transfer_entry' in sys.modules else 'entry').install();"in s:
    needle="__import__('transfer_entry' if 'transfer_entry' in sys.modules else 'entry').install();"
    cmd[i]=s.replace(needle,needle+"sys.path.insert(0,"+repr(str(Path(new).parent))+");__import__('resource_entry');",1)
  return cmd
 c.command=route
 popen=subprocess.Popen
 class WrappedPopen(popen):
  def __init__(self,args,*a,**kw):
   if isinstance(args,(tuple,list)):
    args=list(args)
    for i,x in enumerate(args):
     if str(x)in (str(PARENT/'entry.py'),str(p.D/'benefit_event_full_resources_v174.py')):args[i]=str(HERE/'resource_entry.py');break
     elif str(x)in (str(PARENT/'cpu_host.py'),str(PARENT/'collector.py')):
      args[i:i]=[str(HERE/'resource_entry.py')];break
   super().__init__(args,*a,**kw)
 subprocess.Popen=WrappedPopen
if __name__=='__main__':entry.execute(Path(sys.argv[1]),sys.argv[2:])

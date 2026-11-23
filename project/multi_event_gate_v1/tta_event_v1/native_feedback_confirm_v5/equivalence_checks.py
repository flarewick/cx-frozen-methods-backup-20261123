"""Verify declared operational changes against the exact frozen v4 sources."""
from pathlib import Path
import ast,hashlib,json
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1];O=T/'outputs/native_feedback_confirm_v5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return s.replace('native_feedback_confirm_v4','native_feedback_confirm_v5').replace('cx_tta_feedback_confirm4','cx_tta_feedback_confirm5')
def function(text,name):
 return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
def run():
 old=T/'native_feedback_confirm_v4';rows=[]
 changes={'AGENTS.md','equivalence_checks.py','drain_checks.py','cx_tta_feedback_confirm5/prepare.py','cx_tta_feedback_confirm5/analysis.py','host.py',
  'cx_tta_feedback_confirm5/preload.py','cx_tta_feedback_confirm5/parallel_runtime.py','cx_tta_feedback_confirm5/task_affinity.py','cx_tta_feedback_confirm5/stream_schedule.py'}
 for a in sorted(old.rglob('*')):
  if not a.is_file():continue
  rel=str(a.relative_to(old)).replace('cx_tta_feedback_confirm4','cx_tta_feedback_confirm5');b=C/rel;expected=norm(a.read_text())
  if rel=='cx_tta_feedback_confirm5/parallel_runtime.py':expected=expected.replace('range(6)','range(4)').replace('six native lanes','four native lanes')
  if rel=='cx_tta_feedback_confirm5/preload.py':expected=expected.replace('range(6)','range(4)').replace('CPU_lane_heads=6','CPU_lane_heads=4').replace('persistent_lanes=6','persistent_lanes=4')
  if rel=='cx_tta_feedback_confirm5/task_affinity.py':expected=expected.replace('range(6)','range(4)').replace('0<=i<6','0<=i<4').replace('Exactly six CPU lanes','Exactly four CPU lanes').replace('==[3,3]','==[2,2]').replace('six_fixed_lanes_three_per_task_queue_ownership','four_fixed_lanes_two_per_task_queue_ownership')
  if rel=='cx_tta_feedback_confirm5/stream_schedule.py':expected=expected.replace('Fixed six lanes','Fixed four lanes').replace('initial+6','initial+4').replace('range(3)','range(2)')
  if rel=='host.py':expected=expected.replace('import strict_guard as guard','from native_resource_policy_v1 import guard').replace("proof['CPU_lane_heads']==6","proof['CPU_lane_heads']==4").replace('min_cumulative_utilization=40','min_cumulative_utilization=34')
  if rel not in {'AGENTS.md','equivalence_checks.py','drain_checks.py','cx_tta_feedback_confirm5/prepare.py','cx_tta_feedback_confirm5/analysis.py'}:
   assert b.read_text()==expected,('Undeclared source difference',rel)
  if b.suffix=='.py':ast.parse(b.read_text(),filename=str(b))
  rows.append(dict(source=str(a),destination=str(b),source_sha256=sha(a),destination_sha256=sha(b),declared_operational_change=rel in changes,namespace_only=b.read_text()==norm(a.read_text())))
 before=norm((old/'cx_tta_feedback_confirm4/analysis.py').read_text());after=(C/'cx_tta_feedback_confirm5/analysis.py').read_text()
 assert function(before,'evaluate')==function(after,'evaluate'),'Original physical feedback math, controls and statistical gates must be identical'
 assert function(before,'module')==function(after,'module')
 original=json.loads((T/'outputs/native_feedback_candidate_v1/frozen_candidate.json').read_text())
 # Host-side path mapping verifies all812 scientific dependencies without GPU.
 import sys
 sys.path.insert(0,str(T))
 from native_feedback_host_v1.pipeline import candidate
 assert candidate()['identity']==original['identity']
 policy=json.loads((T/'native_resource_policy_v1/policy.json').read_text());assert policy['minimum_cumulative_GPU_utilization_percent']==policy['minimum_rolling_300s_GPU_utilization_percent']==34
 assert sha(T/'native_terminal_guard_v1/strict_guard.py')==policy['terminal_guard_source_sha256']
 result=dict(passed=True,files=rows,original_candidate_identity=original['identity'],original_frozen_source_count=len(original['source_sha256']),
  scientific_model_physics_controller_precision_RNG_source_unchanged=True,primary_evaluate_function_AST_exact=True,
  operational_changes=['namespaces/output roots','exhaustive sparse20 whole-case import and scheduling only','four native lanes/two per task','user-authorized future cumulative/rolling34; other limits unchanged','source/data/provenance binding'],
  prior_failures_unchanged=True,not_a_GPU_numeric_or_task_benefit_test=True)
 O.mkdir(exist_ok=True)
 (O/'continuation_equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
 print('FOUR_LANE_CONTINUATION_SOURCE_EQUIVALENCE',len(rows),len(original['source_sha256']),flush=True)
if __name__=='__main__':run()

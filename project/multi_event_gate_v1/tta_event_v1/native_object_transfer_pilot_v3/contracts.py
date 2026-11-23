"""Declared substitutions only; scientific episodes/queue/optimizer stay original."""
from pathlib import Path
import ast
C=Path(__file__).resolve().parent;OLD=C.parent/'native_frozen100_v25'
def tree(s):return ast.dump(ast.parse(s),include_attributes=False)
def check():
 for name in ['analysis.py','baseline.py','cache.py','cached_screen.py','completed_planners.py','episode.py','forward_cache.py','initial_integrity.py','ordered_screen.py','owner_precision.py','prefetch.py','preflight.py','preload.py','python_scheduling.py','runtime.py','scene_cache.py','screen.py','semantic.py','semantic_worker.py','setup_fairness.py','startup.py']:
  actual=(C/name).read_text();expected=(OLD/name).read_text().replace('native_frozen100_v25','native_object_transfer_pilot_v3')
  if name=='semantic_worker.py':expected=expected.replace('native_transfer_semantics_v2.inference','native_object_feedback_v5.inference').replace('from native_transfer_semantics_v2 import semantic','from native_object_feedback_v5 import semantic')
  if name=='semantic.py':expected=expected.replace("report=dict(available=row['available']", "report=dict(feedback_interface=row['metadata'].get('interface_version','native_transfer_semantics_v2'),actual_object_count=row['metadata'].get('actual_object_count'),available=row['available']")
  if name=='runtime.py':
   expected=expected.replace("assert spec['scope']=='frozen_best_TTA_cross_task_official_seed0_paired100'", "assert spec['scope']=='new_object_feedback_independent_TRAIN_interface_pilot'").replace("role='frozen_official_EVAL_no_fit'", "role=spec['scene_role']")
   expected=expected.replace("      return SourceStream(mask_model,masks,history,item['instruction']", "      from native_object_feedback_v5.source import SourceStream as ObjectSource\n      selected_source=ObjectSource if seed_report['feedback_interface']=='native_object_feedback_v5'else SourceStream\n      return selected_source(mask_model,masks,history,item['instruction']")
  if name=='runtime.py':expected=expected.replace("c.check_cancel();task=c.call(lane,lambda:c.prepare_task(cls(),lane));raw=lambda:c.call(lane,observe,task)","c.check_cancel()\n    def create_task():\n     from shared6_task_bookkeeping import install_task\n     value=cls();install_task(value);return c.prepare_task(value,lane)\n    task=c.call(lane,create_task);raw=lambda:c.call(lane,observe,task)")
  if name=='screen.py':expected=expected.replace('assert 100000<=seed<101000 and index<100',"assert self.spec['actual_seed_start']<=seed<self.spec['actual_seed_start']+self.spec['max_screen_attempts_per_task']and index<self.spec['episodes_per_task']")
  if name=='analysis.py':expected=expected.replace('formal_outcomes_never_used_for_training=True',"formal_EVAL_not_used=True,scene_role=spec['scene_role'],interface_qualification_only=True")
  if name=='episode.py':expected=expected.replace('from native_best100_endpoint_v1.episode import execute as frozen_execute','from native_feedback_application_v1.episode import execute as derived_execute').replace('return frozen_execute(','return derived_execute(').replace('full/TTA still call the frozen implementation','full/TTA use the explicit derived application version')
  if name=='preload.py':expected=expected.replace('  checked=audit(spec,root)',"  checked=audit(spec,root)\n  from native_feedback_application_v1.audit import audit as audit_application\n  audit_application(spec,root,response_model=prepared['response_model'],predictor=prepared['predictor'])")
  assert tree(actual)==tree(expected),name
 print('OBJECT_PILOT_ORIGINAL_ENGINEERING_AND_DECLARED_SCIENTIFIC_SUBSTITUTIONS_PASS',flush=True)

"""Carry only pure preparation from the diagnosed bounded C5 second session."""
from .common import T,CANON,P,read,sha
from .adoption import failed_second,check as check_adoption,IMPORTED_PLAIN_SESSIONS

def recovered(spec,initial):
 root,old,release,deps=failed_second()
 assert old['initial_screen_state']==initial and old['tasks']==spec['tasks']
 from .preparation import collect
 value=collect(spec,0,initial)
 assert value and value['cached_valid_cases']>0
 canonical=lambda p:str(CANON/p.resolve().relative_to(P))
 assert set(value['source_roots'])=={canonical(root)}
 assert not value['primary_policy_results_imported']and not value['old_masks_or_teacher_outputs_imported']
 assert all('/cases/'not in p and '/answers/'not in p for p in value['input_sha256'])
 spent=round(old['inherited_stage_GPU_seconds']+release['cleanup_seconds_after_start'],3)
 return value,spent,deps

def check():
 assert check_adoption()
 from .adoption import failed_current
 _,_,release,_=failed_current()
 assert release['cleanup_seconds_after_start']==1776.919
 if IMPORTED_PLAIN_SESSIONS==1:
  root,spec,release,deps=failed_second()
  value,spent,_=recovered(spec,spec['initial_screen_state'])
  assert spent==round(spec['inherited_stage_GPU_seconds']+release['cleanup_seconds_after_start'],3)
  assert all(len(v['items'])<=6 for v in value['task_cache'].values())
 return True

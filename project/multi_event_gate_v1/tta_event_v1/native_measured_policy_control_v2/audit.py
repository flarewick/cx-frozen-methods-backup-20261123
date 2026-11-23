"""Actual measured-control usage must match all original25 native clocks."""
from pathlib import Path
from collections import Counter
from native_policy_execution_v1.common import read,sha

def validate(rows,tasks):
 assert len(rows)==25 and set(Counter(r['task']for r in rows))==set(tasks)
 assert all(sum(r['task']==task for r in rows)==5 for task in tasks)
 for r in rows:
  u=r['usage'];assert u['passed']and u['eligible_commands']>=0 and u['clock_ticks']==r['physical_ticks']>=0
  assert u['clock_ticks']==u['native_clock_ticks']+u['fallback_clock_ticks']and u['original_randomized_light_and_unsupported_cases_preserved']
  assert u['original_clock_trace_and_feedback_every_physics_tick']and u['original_inactive_path_preserved']and u['original_reference_not_installed']and u['no_extra_commands_or_steps']
 assert all(sum(r['usage']['native_clock_ticks']for r in rows if r['task']==task)>0 for task in tasks),'Each task must actually use native control somewhere'
 return True

def audit(root,spec):
 root=Path(root);rows=[];files=[]
 for case in sorted((root/'cases').iterdir()):
  usage=case/'M_event_full/measured_control_usage.json';result=case/'M_event_full/result.json'
  value=read(result);assert value['native_episode_complete']
  rows.append(dict(task=case.name.split(':')[0],usage=read(usage),physical_ticks=value['physical_ticks']));files.extend([usage,result])
 assert validate(rows,spec['tasks'])
 whole=root/'CPU_storage_whole25_audit.json';assert read(whole)['passed'];files.append(whole)
 return dict(passed=True,all25_original_whole_raw_controls_required=True,actual_cases=25,actual_native_eligible_commands=sum(r['usage']['eligible_commands']for r in rows),actual_original_physics_clock_ticks=sum(r['physical_ticks']for r in rows),input_sha256={str(p):sha(p)for p in files})

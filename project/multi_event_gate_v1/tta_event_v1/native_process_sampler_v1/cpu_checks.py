"""Real read-only NVIDIA acquisition and fail-closed parser/guard contracts."""
from pathlib import Path
import ast,json,hashlib,subprocess
from unittest.mock import patch
from . import metrics
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent;O=T/'outputs/native_process_sampler_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(first='99 99',last='2 1',gpu=0):
 return '# header\n20261030 19:10:00 %d 123 C %s - - python\n20261030 19:10:01 %d 123 C %s - - python\n'%(gpu,first,gpu,last)
def main():
 checks=[]
 for a,b,expected in [('99 99','2 1',{123:2}),('- -','0 0',{123:0}),('99 99','- -',{}),('99 99','- 1',{123:1}),('- -','- -',{})]:
  result,frames=metrics.parse_frames(fixture(a,b),0);assert result==expected;checks.append(dict(first=a,last=b,expected=expected))
 for raw in [fixture().splitlines()[1]+'\n',fixture().replace('19:10:01','19:10:00'),fixture().replace('19:10:01','19:09:59'),fixture(gpu=1),fixture(last='101 0'),'bad row']:
  try:metrics.parse_frames(raw,0)
  except ValueError:pass
  else:raise AssertionError('Invalid sampling frame accepted')
 for exc in(subprocess.TimeoutExpired('pmon',3),subprocess.CalledProcessError(1,'pmon')):
  with patch.object(metrics.subprocess,'run',side_effect=exc)as mocked:
   try:metrics.process_utilization(0)
   except type(exc):pass
   else:raise AssertionError('Sampling failure suppressed')
   assert mocked.call_count==1 and mocked.call_args.kwargs['timeout']==3
 # Complete guard identical, including missing-stop and existing zero-on-recovery.
 original=ast.parse((T/'native_resource_policy_v4/strict_guard.py').read_text());new=ast.parse((C/'strict_guard.py').read_text())
 for node in ast.walk(new):
  if isinstance(node,ast.ImportFrom)and node.module=='native_process_sampler_v1.metrics':node.module='gpu_metrics_strict'
 assert ast.dump(original)==ast.dump(new)
 records=[]
 for gpu in(2,3,4):
  values=metrics.process_utilization(gpu);d=metrics.diagnostics();assert d['collection_seconds']<3 and len(d['fixed_interval_frames'])==2 and d['selected_frame_index']==1
  assert d['fixed_interval_frames'][-1]['values']==values and all(r['gpu']==gpu for r in d['components']);records.append(d)
 report=dict(passed=True,all_strict_guard_AST_unchanged_except_metric_import=True,same_three_second_call_timeout=True,fixed_last_frame_not_max_or_available_selection=True,double_missing_still_unavailable=True,actual_read_only_NVIDIA_collections=records,parser_negative_checks=6,retained_parser_cases=checks,no_GPU_work_started=True,source_sha256={str(p):sha(p)for p in C.rglob('*')if p.is_file()and p.suffix in('.py','.json')},original_parser_sha256=sha(T.parents[1]/'deployment/gpu_metrics_strict.py'))
 O.mkdir(exist_ok=True)
 with(O/'CPU_fresh_sample_checks.json').open('x')as f:json.dump(report,f,indent=2)
 print('FRESH_PM0N_LAST_INTERVAL_STRICT_GUARD_CPU_PASS',len(records),flush=True)
if __name__=='__main__':main()

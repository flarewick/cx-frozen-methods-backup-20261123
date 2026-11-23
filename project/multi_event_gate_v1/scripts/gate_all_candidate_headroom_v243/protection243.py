"""Additional conservative early stop; original33% guard remains untouched."""
MINIMUM_HEADROOM=40.
CHECK_AFTER=240.
def reason(row):
 if row.get('seconds',0)<CHECK_AFTER or not row.get('gpu_pids'):return None
 fields=['mean_utilization','time_weighted_mean_utilization','rolling_300s_utilization']
 values=[row.get(k,0.)for k in fields]
 for pid in row['gpu_pids']:
  values.extend(row.get(k,{}).get(str(pid),row.get(k,{}).get(pid,0.))for k in ['process_mean_utilization','process_rolling_300s_utilization'])
 if min(values)<MINIMUM_HEADROOM:return 'Additional243 GPU headroom below40% after240s; stop before risking original33% floor'
 if row.get('metric_missing_conservative_zero_pids'):return 'Additional243 stop on unresolved PID metrics after240s'
 return None

def install(guard):
 original=guard.guarded_command
 def guarded(*a,**kw):
  from pathlib import Path
  import json,hashlib
  report=Path(a[2]);marker=report.with_name(report.name+'.protection243.json')
  marker.write_text(json.dumps(dict(protection_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),original33pct_guard_retained=True,additional40pct_after240s=True,telemetry_basename=report.name))+'\n')
  prior=kw.get('on_sample')
  def sample(row):return (prior(row)if prior else None)or reason(row)
  kw['on_sample']=sample
  return original(*a,**kw)
 guard.guarded_command=guarded
 return original

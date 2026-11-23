"""Extra resource margin; original strict guard still owns admission and cleanup."""
from pathlib import Path
import json,sys
T=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(T.parents[1]/'deployment'))
from gpu_utilization_window import UtilizationWindow
from native_process_missing_guard_v2 import guard as original
from native_process_missing_guard_v2.empty import zero_previous
class Monitor:
 def __init__(self):self.window=UtilizationWindow(180);self.recent=UtilizationWindow(60)
 def observe(self,row):
  elapsed=row.get('lower_bound_exact_elapsed',row['seconds']);pids=row['gpu_pids']
  missing={int(k):v for k,v in row.get('missing_process_zero_lower_bound',{}).items()}
  zero_previous(self.window,self.recent,missing)
  def value(table,pid):return table.get(pid,table.get(str(pid)))
  values={pid:value(row['process_utilization'],pid)or 0 for pid in pids}
  _,long=self.window.add(elapsed,0,values);_,short=self.recent.add(elapsed,0,values)
  metrics=dict(seconds=row['seconds'],process_rolling180={str(pid):long.get(pid,0)for pid in pids},process_rolling60={str(pid):short.get(pid,0)for pid in pids},device_recent60=row['recent_60s_utilization'],cumulative_floor=35,short_floor=36,unchanged_original_guard=True)
  reason=None
  if row['seconds']>=280 and pids:
   if min(row['mean_utilization'],row['time_weighted_mean_utilization'],row['rolling_300s_utilization'])<35:reason='Additional governor cumulative device35 margin not met'
   if row['recent_60s_utilization']<36:reason='Additional governor recent60 device36 margin not met'
   for pid in pids:
    key=str(pid)
    if value(row['process_mean_utilization'],pid)is None or value(row['process_mean_utilization'],pid)<35:reason='Additional governor cumulative process35 margin not met'
    if min(long.get(pid,0),short.get(pid,0))<36:reason='Additional governor process60/180 margin36 not met'
  return reason,metrics

def guarded_command(*args,on_sample=None,**kwargs):
 report=Path(args[2]if len(args)>2 else kwargs['report']);path=report.with_name('governor_margin.jsonl');assert not path.exists();monitor=Monitor()
 def checked(row):
  reason,metrics=monitor.observe(row)
  with path.open('a')as out:out.write(json.dumps(dict(**metrics,stop_reason=reason))+'\n')
  prior=on_sample(row)if on_sample else None
  return prior or reason
 return original.guarded_command(*args,on_sample=checked,**kwargs)

def __getattr__(name):return getattr(original,name)

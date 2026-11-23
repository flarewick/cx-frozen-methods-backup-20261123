"""Physical device samples, fixed startup clock, conservative covered-time means."""
from bisect import bisect_right

class DeviceWindow:
 def __init__(self,start_unix_us,gpu_uuid):
  assert type(start_unix_us)is int and type(gpu_uuid)is str and gpu_uuid.startswith('GPU-')
  self.start=start_unix_us;self.uuid=gpu_uuid;self.times=[];self.values=[];self.areas=[];self.lookup={};self.total=0;self.max_gap=0;self.last_now=None;self.checkpoints=[]
 def area(self,t):
  if not self.times or t<=self.times[0]:return 0
  if t>=self.times[-1]:return self.areas[-1]
  k=bisect_right(self.times,t)-1
  return self.areas[k]+self.values[k]*(t-self.times[k])
 def update(self,data,now_unix_us,elapsed,*,steady_required=False):
  if data.get('gpu_uuid')!=self.uuid:raise ValueError('Physical device UUID mismatch')
  if type(now_unix_us)is not int or now_unix_us<=self.start:raise ValueError('Invalid device accounting clock')
  if abs((now_unix_us-self.start)-elapsed*1e6)>100000:raise ValueError('Host/monotonic accounting clocks diverged')
  if self.last_now is not None and now_unix_us<=self.last_now:raise ValueError('Accounting time did not advance')
  frame=data['frames'][-1];raw=frame['device_samples'];tables=[]
  for name in ['SM','bandwidth']:
   table={}
   for r in raw[name]:
    t,v=r['timestamp_us'],r['value']
    if type(t)is not int or t<=0 or t>now_unix_us or t in table:raise ValueError('Invalid/duplicate/future device timestamp')
    if type(v)is not int or not 0<=v<=100:raise ValueError('Unavailable/invalid device component')
    table[t]=v
   if not table:raise ValueError('Missing physical device samples')
   tables.append(table)
  sm,bw=tables;common=sorted(sm.keys()&bw.keys())
  if not common:raise ValueError('No paired physical samples')
  lo,hi=common[0],common[-1]
  if {t for t in sm if lo<=t<=hi}!={t for t in bw if lo<=t<=hi}:raise ValueError('Unpaired internal device component')
  if not 0<=now_unix_us-hi<=750000:raise ValueError('Physical driver samples stale')
  if self.times and hi<=self.times[-1]:raise ValueError('No new physical device sample')
  old_last=self.times[-1]if self.times else None
  late=0
  for t in common:
   if t<self.start:continue
   pair=(sm[t],bw[t])
   if t in self.lookup:
    if self.lookup[t]!=pair:raise ValueError('Conflicting driver sample history')
   else:
    if old_last is not None and t<old_last:late+=1
    self.lookup[t]=pair
  self.late_records=getattr(self,'late_records',0)+late
  items=sorted(self.lookup.items());self.times=[];self.values=[];self.areas=[];self.total=0;self.max_gap=0
  for t,pair in items:
   v=max(pair)
   if not self.times:
    if t-self.start>250000:raise ValueError('Initial device coverage missing')
    area=0
   else:
    gap=t-self.times[-1]
    if not 0<gap<=250000:raise ValueError('Missing internal device time interval')
    self.max_gap=max(self.max_gap,gap);area=self.areas[-1]+self.values[-1]*gap
   self.times.append(t);self.values.append(v);self.areas.append(area);self.total+=v
  if not self.times:raise ValueError('No device records since fixed guard start')
  self.last_now=now_unix_us
  def mean(seconds=None):
   begin=self.start if seconds is None else max(self.start,now_unix_us-int(seconds*1e6))
   return(self.area(now_unix_us)-self.area(begin))/(now_unix_us-begin)
  result=dict(started_unix_us=self.start,now_unix_us=now_unix_us,gpu_uuid=self.uuid,first_driver_timestamp_us=self.times[0],last_driver_timestamp_us=self.times[-1],actual_driver_records=len(self.times),backfilled_actual_records=getattr(self,'late_records',0),maximum_internal_gap_us=self.max_gap,unobserved_prefix_us=self.times[0]-self.start,unobserved_tail_us=now_unix_us-self.times[-1],observed_fraction=(self.times[-1]-self.times[0])/(now_unix_us-self.start),boundary_unobserved_time_credited_zero_only_as_lower_bound=True,no_raw_samples_invented=True,all_internal_components_paired=True,sample_mean=self.total/len(self.times),cumulative_lower_bound=mean(),rolling300_lower_bound=mean(300),rolling60_lower_bound=mean(60),integral_percent_us=self.areas[-1],fixed_start_never_reset=True)
  for previous in self.checkpoints:
   past=self.retrospective(previous)
   if min(past['sample_mean'],past['cumulative_lower_bound'],past['rolling300_lower_bound'])<34:raise ValueError('Backfilled physical samples invalidate an earlier34percent device check')
  if steady_required and min(result['sample_mean'],result['cumulative_lower_bound'],result['rolling300_lower_bound'])>=34:self.checkpoints.append(now_unix_us)
  result['prior_steady_checkpoints_rechecked']=len(self.checkpoints)
  return result


 def retrospective(self,now_unix_us):
  """Recompute an earlier decision from all finally published raw samples."""
  count=bisect_right(self.times,now_unix_us)
  if not count:raise ValueError('No actual samples at retrospective checkpoint')
  def mean(seconds=None):
   begin=self.start if seconds is None else max(self.start,now_unix_us-int(seconds*1e6))
   return(self.area(now_unix_us)-self.area(begin))/(now_unix_us-begin)
  return dict(sample_mean=sum(self.values[:count])/count,cumulative_lower_bound=mean(),rolling300_lower_bound=mean(300),rolling60_lower_bound=mean(60))

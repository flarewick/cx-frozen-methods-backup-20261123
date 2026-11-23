"""CPU-only conservative eligibility estimates, separate from GPU accounting."""
import time

class History:
 def __init__(self):self.rows={};self.idle_since=None;self.observed=None;self.fleet=None
 def update(self,fleet,now,ages):
  self.fleet=fleet;self.observed=now;active=set()
  own=[c for c in fleet['cards']if c['own_pids']]
  if not own:
   if self.idle_since is None:self.idle_since=now
  else:self.idle_since=None
  for card in own:
   key=card['uuid'];active.add(key);pids=tuple(sorted(card['own_pids']));identities=tuple((p,ages[p][0])for p in pids)
   fraction=card['own_fraction'];util=card['attributable_utilization'];oldest=max(ages[p][1]for p in pids);row=self.rows.get(key)
   if row is None or row['identities']!=identities:
    row=dict(identities=identities,first=now,last=now,last_fraction=fraction,last_util=util,memory_integral=0.,util_integral=0.,credible_seconds=0.,oldest_age=oldest);self.rows[key]=row
   else:
    dt=now-row['last']
    # No credit for missing/long observation spans or unknown attribution.
    if 0<dt<=8:
     if fraction is not None and row['last_fraction']is not None:row['memory_integral']+=min(fraction,row['last_fraction'])*dt
     if util is not None and row['last_util']is not None:row['util_integral']+=min(util,row['last_util'])*dt
     row['credible_seconds']+=dt
    row.update(last=now,last_fraction=fraction,last_util=util,oldest_age=oldest)
  for key in list(self.rows):
   if key not in active:del self.rows[key]
 def estimates(self,now):
  result={}
  for key,row in self.rows.items():
   age=row['oldest_age']+max(0,now-row['last']);duration=max(0,now-row['first'])
   memory=row['memory_integral']/age if age>0 else 0.;util=row['util_integral']/age if age>0 else 0.
   result[key]=dict(observed_seconds=duration,process_age_seconds=age,observed_memory_time_integral=row['memory_integral'],observed_utilization_time_integral=row['util_integral'],memory_lifetime_estimate=memory,utilization_lifetime_estimate=util,eligible=duration>=120 and row['credible_seconds']>=120 and memory>=.55,unobserved_prefix_credit=0,sparse_samples_are_admission_estimates_only=True)
  return result
 def decision(self,selected_uuid,now,*,running=False):
  if self.observed is None or now-self.observed>8:return False,'No fresh conservative rule4 history',{}
  fleet=self.fleet;cards=fleet['cards'];selected=next((c for c in cards if c['uuid']==selected_uuid),None)
  if selected is None:return False,'Selected GPU absent',{}
  count=fleet['own_gpu_count']
  if count>(3 if running else 2):return False,'Account GPU limit',{}
  if not running and selected['pids']:return False,'Selected GPU is occupied',{}
  estimates=self.estimates(now);own=[c for c in cards if c['own_pids']]
  if running:
   if len(own)<=1:return True,'At most one account GPU; no additional-card condition',estimates
   ordered=sorted(own,key=lambda c:(estimates[c['uuid']]['memory_lifetime_estimate'],c['index']))
   return all(estimates[c['uuid']]['eligible']for c in ordered[1:]),'Conservative mean memory for every GPU except the least-filled estimate',estimates
  prior=[c for c in own if c['uuid']!=selected_uuid]
  if not prior:
   okay=self.idle_since is not None and now-self.idle_since>=30
   return okay,'First GPU requires30s account-idle observation',estimates
  return all(estimates[c['uuid']]['eligible']for c in prior),'Prior GPUs require120s history and lifetime own-memory>=55%; utilization is diagnostic only',estimates

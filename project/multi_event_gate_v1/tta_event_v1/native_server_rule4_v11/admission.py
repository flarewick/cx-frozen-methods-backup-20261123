"""Preserve strict new-card admission; later account jobs cannot revoke it."""
import copy, hashlib, json, time
from native_server_rule4_v10.admission import Monitor as OriginalMonitor

def digest(value):
 canonical=json.loads(json.dumps(value,allow_nan=False))
 return hashlib.sha256(json.dumps(canonical,sort_keys=True,allow_nan=False).encode()).hexdigest()

class Monitor(OriginalMonitor):
 def __init__(self):
  self.admitted={}
  super().__init__()

 def require(self,uuid,*,running=False):
  if not running:
   # Original 120s/55%, unknown-zero, fresh instantaneous rule4, PID/startticks.
   original=super().require(uuid,running=False)
   with self.lock:self.admitted[uuid]=copy.deepcopy(original)
   return original
  with self.lock:
   if self.failed:raise RuntimeError('Conservative rule4 metadata failed: '+self.failed)
   if uuid not in self.admitted:raise RuntimeError('No successful strict admission for this GPU')
   if self.history.observed is None or time.monotonic()-self.history.observed>8:
    raise RuntimeError('No fresh account observation')
   fleet=self.history.fleet
   if not any(c['uuid']==uuid for c in fleet['cards']):raise RuntimeError('Admitted GPU absent')
   if fleet['own_gpu_count']>3:raise RuntimeError('Account GPU limit')
   original=self.admitted[uuid]
   return dict(passed=True,reason='Original strict admission preserved; live account limit checked independently of later jobs',
    original_strict_admission_sha256=digest(original),original_strict_admission=original,
    latest_observation=copy.deepcopy(self.latest),checked_monotonic=time.monotonic(),no_CUDA_or_GPU_work=True,
    extra_card_condition_applies_before_new_allocation=True,
    later_account_jobs_do_not_revoke_prior_legal_admission=True,
    process_device_utilization_CPU_memory_and_account_limits_still_enforced_by_original_guard=True)

_monitor=None
def require(uuid):
 global _monitor
 if _monitor is None:_monitor=Monitor()
 return _monitor.require(uuid)
def ongoing(uuid):
 if _monitor is None:raise RuntimeError('Rule4 pre-admission history absent')
 return _monitor.require(uuid,running=True)
def close():
 global _monitor
 if _monitor is not None:
  monitor=_monitor;_monitor=None;monitor.close()
def prime():
 global _monitor
 if _monitor is None:_monitor=Monitor()
 return _monitor.root

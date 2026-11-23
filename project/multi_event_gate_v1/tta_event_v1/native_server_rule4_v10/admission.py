"""Fresh scheduled metadata only; preserve original rule4 at permit."""
from pathlib import Path
import json,os,threading,time,uuid as uuid_module
from native_server_rule4_v1.admission import decision as instantaneous
from native_account_stream_v1.transport import snapshot,context
from .history import History
T=Path(__file__).resolve().parents[1]
class Monitor:
 def __init__(self):
  self.history=History();self.lock=threading.Lock();self.failed=None;self.latest=None;self.stop=threading.Event()
  self.root=T/'outputs/native_server_rule4_v10'/('host_'+str(os.getpid())+'_'+uuid_module.uuid4().hex[:12]);self.root.mkdir(parents=True,exist_ok=False)
  from .handoff import seed
  seed(self.history,self.root)
  self.history.idle_since=None # Require a freshly observed first-card quiet period.
  self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def run(self):
  with context(self.root/'NVML_account_rule4_actual_exit.json'):
   self.run_metadata()
 def run_metadata(self):
  while not self.stop.is_set():
   try:
    fleet=snapshot();now=time.monotonic();uptime=float(Path('/proc/uptime').read_text().split()[0]);hz=os.sysconf('SC_CLK_TCK');ages={}
    for card in fleet['cards']:
     for pid in card['own_pids']:
      ticks=int(Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split()[19]);ages[pid]=(ticks,max(0,uptime-ticks/hz))
    with self.lock:
     if self.history.observed is not None and now-self.history.observed>8:self.history.idle_since=None
     self.history.update(fleet,now,ages);self.latest=dict(observed_unix_ns=time.time_ns(),monotonic=now,fleet=fleet,process_ages=ages,estimates=self.history.estimates(now),no_CUDA_or_GPU_work=True,scientific_GPU_clock_not_reset=True)
     with(self.root/'observations.jsonl').open('a')as f:f.write(json.dumps(self.latest,allow_nan=False)+'\n')
   except FileNotFoundError as e:
    # A PID can retire between NVML and /proc. Discard this eligibility
    # observation, never credit the missing interval, wait for next scheduled sample.
    with self.lock:
     self.history.observed=None;self.history.idle_since=None
     with(self.root/'identity_races.jsonl').open('a')as f:f.write(json.dumps(dict(reason=repr(e),observation_discarded=True,no_scientific_metric_or_clock_change=True))+'\n')
   except BaseException as e:
    with self.lock:self.failed=repr(e)
    (self.root/'metadata_failure.json').write_text(json.dumps(dict(error=repr(e),automatic_retry=False,no_GPU_work=True))+'\n');return
   self.stop.wait(5)
 def close(self):
  self.stop.set();self.thread.join(timeout=7)
  note=dict(thread_stopped=not self.thread.is_alive(),metadata_failure=self.failed,no_CUDA_or_GPU_work=True)
  with(self.root/'actual_monitor_exit.json').open('x')as f:json.dump(note,f,indent=2)
  if self.thread.is_alive():raise RuntimeError('Rule4 CPU metadata monitor did not close')
 def require(self,uuid,*,running=False):
  with self.lock:
   if self.failed:raise RuntimeError('Conservative rule4 metadata failed: '+self.failed)
   okay,reason,estimates=self.history.decision(uuid,time.monotonic(),running=running)
   if not okay:raise RuntimeError(reason)
   proof=dict(passed=True,reason=reason,estimates=estimates,latest_observation=self.latest,observation_file=str(self.root/'observations.jsonl'),no_CUDA_or_GPU_work=True)
   observed_prior={c['uuid']:tuple(sorted(c['own_pids']))for c in self.history.fleet['cards']if c['own_pids']and c['uuid']!=uuid}
   observed_ticks={pid:identity[0]for pid,identity in self.latest['process_ages'].items()}
  if not running:
   # Fresh original rules still apply; history never substitutes for them.
   fleet=snapshot();okay,reason=instantaneous(fleet,uuid)
   if not okay:raise RuntimeError(reason)
   fresh_prior={c['uuid']:tuple(sorted(c['own_pids']))for c in fleet['cards']if c['own_pids']and c['uuid']!=uuid}
   if fresh_prior!=observed_prior:raise RuntimeError('Own GPU population changed since historical eligibility; await a fresh scheduled observation')
   for pids in fresh_prior.values():
    for pid in pids:
     try:ticks=int(Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split()[19])
     except FileNotFoundError:raise RuntimeError('Prior own GPU process retired during admission')
     if observed_ticks.get(pid)!=ticks:raise RuntimeError('Prior own GPU process identity changed')
   proof['fresh_original_rule4_fleet']=fleet
  return proof
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

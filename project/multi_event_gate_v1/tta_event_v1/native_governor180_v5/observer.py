"""Read an existing LIBERO observer, with source, PID birth and freshness verification."""
import hashlib,json,os,time
from pathlib import Path
from .policy import L,verify_rules,require_admission as policy_require_admission
T=Path(__file__).resolve().parents[1];C=Path(__file__).parent
OWN=T/'outputs/native_governor180_v5/observer'
STATE=OWN/'current.json'
HISTORY=OWN/'raw.jsonl'

def verify_reference():
 verify_rules();manifest=json.loads((C/'reference.json').read_text())
 for name,digest in manifest['source_sha256'].items():
  assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,('Shared LIBERO reference changed',name)
 proof=L/'outputs/resource_guard_v15_CPU_acceptance.json'
 assert hashlib.sha256(proof.read_bytes()).hexdigest()==manifest['CPU_proof_sha256']
 assert json.loads(proof.read_text())['passed']

def read_current(path,now=None):
 v=json.loads(Path(path).read_text());now=time.time()if now is None else now
 if v.get('error')or now-v['wall_time']>8 or v['wall_time']>now+1:raise RuntimeError('Fleet observer stale or failed')
 return v

def latest():
 verify_reference();v=read_current(STATE)
 launch=json.loads((OWN/'launch.json').read_text())
 for name,digest in launch['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest;pid=launch['pid'];p=Path('/proc',str(pid))
 try:
  st=(p/'stat').read_text().rsplit(')',1)[1].split();assert p.stat().st_uid==os.getuid()and st[0]!='Z'
  assert int(st[19])==int(launch['start_ticks'])and v['observer_pid']==pid
 except (OSError,AssertionError)as e:raise RuntimeError('TTA-owned observer PID birth unverified')from e
 assert v['schema']=='governor180_v15'and len(v['cards'])==7 and [c['index']for c in v['cards']]==list(range(7))
 return v

def admission_snapshot():
 from native_account_stream_v1.transport import inventory
 v=latest();fresh=inventory()
 for c in v['cards']:
  cur=next(x for x in fresh['cards']if x['uuid']==c['uuid'])
  if set(c['pids'])!=set(cur['pids'])or set(c['own_pids'])!=set(cur['own_pids']):raise RuntimeError('Fleet changed since180s observation; wait for fresh observer')
  for p in c['processes']:
   try:birth=int(Path('/proc',str(p['pid']),'stat').read_text().rsplit(')',1)[1].split()[19])
   except (FileNotFoundError,ProcessLookupError):raise RuntimeError('Fleet changed since180s observation; wait for fresh observer')
   if birth!=p['birth']:raise RuntimeError('Fleet changed since180s observation; wait for fresh observer')
  c['available']=cur['available']
 v['available_indices']=[c['index']for c in v['cards']if c['available']];v['wall_time']=time.time()
 return v

_root=None

def prime():
 global _root
 import uuid
 assert _root is None
 _root=T/'outputs/native_governor180_v5'/('host_'+str(os.getpid())+'_'+uuid.uuid4().hex[:12]);_root.mkdir(parents=True)
 (_root/'reference.json').write_bytes((C/'reference.json').read_bytes());latest()
 return _root

def require(uuid):
 assert _root is not None
 fleet=admission_snapshot();policy_require_admission(fleet,uuid)
 row=dict(passed=True,selected_uuid=uuid,fleet=fleet,no_CUDA_or_GPU_work=True,server_window_seconds=180,source='TTA_owned_exact_LIBERO_v15_sampling_and_statistics')
 with(_root/'admissions.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
 return row

def ongoing(uuid):
 v=latest();assert any(c['uuid']==uuid for c in v['cards'])and v['own_gpu_count']<=3
 return dict(passed=True,no_CUDA_or_GPU_work=True,server_window_seconds=180,current_fleet=v,actual_per_process_and_account_rule8_enforced_in_live_guard=True)

def close():
 global _root
 if _root is None:return
 p=_root/'actual_monitor_exit.json';assert not p.exists();p.write_text(json.dumps(dict(passed=True,no_CUDA_or_GPU_work=True,readonly_existing_LIBERO_observer_not_stopped=True,local_reader_closed=True,TTA_owned_observer_continues_CPU_only=True))+'\n');_root=None

def check_admission(info):
 return require(info['uuid'])

def require_admission(fleet,uuid):
 from native_resource_policy_v4.admission import require_admission as empty_under3
 return empty_under3(fleet,uuid) # Full180 policy stays in the caller's explicit CPU-wait try/except.

"""TTA-owned CPU sampler; exact v15 per-process NVML/CPU sampling, no GPU work."""
import fcntl,json,os,subprocess,time
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET
from .statistics import parse_pmon,aggregate
from .observer import OWN,STATE,HISTORY,C,verify_reference

def replace(path,value):
 temp=path.with_suffix('.'+str(os.getpid())+'.tmp');temp.write_text(json.dumps(value,allow_nan=False)+'\n');os.replace(temp,path)

def proc(pid, uptime, now, previous):
    p = Path('/proc', str(pid))
    st = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    hz = os.sysconf('SC_CLK_TCK')
    birth = int(st[19]); ticks = int(st[11]) + int(st[12])
    old = previous.get((pid, birth)); previous[(pid, birth)] = (now, ticks)
    cpu = 0. if old is None else 100 * (ticks - old[1]) / hz / max(now - old[0], .001)
    return dict(pid=pid, birth=birth, uid=p.stat().st_uid, age=max(0, uptime - birth / hz),
                cpu=cpu, rss=int(st[21]) * os.sysconf('SC_PAGE_SIZE'))

def sample(previous):
    cmd = lambda args: subprocess.run(['nvidia-smi', *args], check=True, capture_output=True,
                                     text=True, timeout=4).stdout
    components = parse_pmon(cmd(['pmon', '-s', 'u', '-c', '1']))
    xml = ET.fromstring(cmd(['-q', '-x']))
    now = time.monotonic(); wall = time.time(); uptime = float(Path('/proc/uptime').read_text().split()[0])
    cards = []
    for i, g in enumerate(xml.findall('gpu')):
        number = lambda path: float(g.findtext(path).split()[0])
        ps = []
        for node in g.findall('./processes/process_info'):
            try:
                pid = int(node.findtext('pid')); p = proc(pid, uptime, now, previous)
            except (FileNotFoundError, ProcessLookupError):
                continue
            sm, bd = components.get((i, pid), (None, None))
            token = node.findtext('used_memory', '').split()
            memory = float(token[0]) if token and token[0].replace('.', '', 1).isdigit() else 0.
            p.update(sm=sm, bandwidth=bd, memory_mib=memory,
                     metrics_available=(sm is not None and bd is not None and memory > 0))
            ps.append(p)
        mem = number('./fb_memory_usage/used'); sm = number('./utilization/gpu_util')
        cards.append(dict(index=i, uuid=g.findtext('uuid'), sm=sm,
                          bandwidth=number('./utilization/memory_util'), memory_mib=mem,
                          total_mib=number('./fb_memory_usage/total'), processes=ps,
                          pids=[p['pid'] for p in ps], own_pids=[p['pid'] for p in ps if p['uid'] == os.getuid()],
                          available=not g.findall('./processes/process_info') and mem <= 64 and sm <= 5))
    return dict(monotonic=now, wall_time=wall, observer_pid=os.getpid(), cards=cards)

def record_collection_error(error,root=OWN):
 # A collector failure is not a new resource sample. Preserve the last valid
 # snapshot byte-for-byte (including its original timestamp); the reader's
 # unchanged8s freshness bound still stops stale/unverified monitoring.
 now=time.time();state=root/'current.json'
 row=dict(wall_time=now,monotonic=time.monotonic(),observer_pid=os.getpid(),error_type=type(error).__name__,error=repr(error),invalid_collection_not_a_valid_sample=True)
 if state.exists():
  previous=json.loads(state.read_text());row['last_valid_wall_time']=previous.get('wall_time')if not previous.get('error')else None
 else:row['last_valid_wall_time']=None
 with(root/'errors.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
 replace(root/'last_error.json',row)
 if not state.exists():replace(state,dict(wall_time=now,error=repr(error),observer_pid=os.getpid()))

def main():
 verify_reference();OWN.mkdir(parents=True,exist_ok=True)
 lock=(OWN/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not(OWN/'launch.json').exists(),'Inspect existing observer, never duplicate/restart unchanged'
 import hashlib
 proof=C.parent/'outputs/native_governor180_v4/CPU_checks_v1.json';actual=proof.with_name('CPU_actual_exit_v1.json')
 assert json.loads(proof.read_text())['passed']and json.loads(actual.read_text())['exit_code']==0
 birth=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
 launch=dict(pid=os.getpid(),start_ticks=birth,no_CUDA_or_GPU_work=True,lifetime_seconds=35*3600,original_LIBERO_process_untouched=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in C.glob('*')if p.is_file()})
 launch['CPU_proof_sha256']=hashlib.sha256(proof.read_bytes()).hexdigest()
 (OWN/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 history=deque();previous={};start=time.monotonic()
 while time.monotonic()-start<35*3600:
  begin=time.monotonic()
  try:
   raw=sample(previous);history.append(raw)
   while history and history[0]['monotonic']<raw['monotonic']-180:history.popleft()
   current=aggregate(json.loads(json.dumps(raw)),history)
   replace(STATE,current);replace(OWN/'window.json',list(history))
   with HISTORY.open('a')as f:f.write(json.dumps(raw,allow_nan=False)+'\n')
  except Exception as exc:record_collection_error(exc)
  time.sleep(max(.1,3-(time.monotonic()-begin)))
 (OWN/'normal_CPU_loop_exit.json').write_text(json.dumps(dict(normal=True,pid=os.getpid(),elapsed=time.monotonic()-start,no_GPU_work=True))+'\n')

if __name__=='__main__':main()

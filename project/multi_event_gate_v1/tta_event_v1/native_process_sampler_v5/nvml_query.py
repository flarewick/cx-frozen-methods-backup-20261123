"""Guard-private fresh NVML queries with the original killable three-second bound."""
import json,subprocess,sys,time,csv,io
from pathlib import Path
SINGLE={'--query-gpu=memory.total','--query-gpu=utilization.gpu,utilization.memory,memory.used'}
APPS={'--query-compute-apps=gpu_uuid,pid','--query-compute-apps=gpu_uuid,pid,used_gpu_memory'}
def form(a):
 assert type(a)is list and all(type(x)is str for x in a)
 if len(a)==3 and a[0]=='-i'and a[2]in SINGLE:
  i=int(a[1]);assert str(i)==a[1]and 0<=i<7;return 'device'
 assert len(a)==1 and a[0]in APPS,'Unsupported guard query; no CLI fallback';return 'apps'
def mib(v):
 assert type(v)is int and 0<=v<2**50,'Unavailable or invalid memory';return(v+2**20-1)//2**20

def render(data,a):
 kind=form(a);assert data['arguments']==a and data['read_only_NVML']is True and data['CUDA_context_created']is False
 devices=data['devices'];assert len(devices)==(1 if kind=='device'else 7);assert [d['index']for d in devices]==([int(a[1])]if kind=='device'else list(range(7)))
 rows=[];seen=set()
 for d in devices:
  assert type(d['uuid'])is str and d['uuid'].startswith('GPU-')and d['uuid']not in seen;seen.add(d['uuid'])
  if kind=='device':
   total=mib(d['memory_total_bytes']);assert total>0
   if a[2]=='--query-gpu=memory.total':rows.append([total])
   else:
    assert all(type(d[k])is int and 0<=d[k]<=100 for k in ['SM','bandwidth'])
    used=mib(d['memory_used_bytes']);assert used<=total;rows.append([d['SM'],d['bandwidth'],used])
  else:
   pids=set()
   for p in d['compute_processes']:
    pid=p['pid'];assert type(pid)is int and pid>0 and pid not in pids;pids.add(pid)
    row=[d['uuid'],pid]
    if a[0].endswith(',used_gpu_memory'):row.append('N/A'if p['used_GPU_memory_bytes']is None else mib(p['used_GPU_memory_bytes']))
    rows.append(row)
 text=io.StringIO();csv.writer(text,lineterminator='\n').writerows(rows);return text.getvalue()

class DoctorProxy:
 def __init__(self,original):self.original=original;self.report=None
 def __getattr__(self,name):return getattr(self.original,name)
 def record(self,row):
  if self.report is not None:
   with self.report.open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 def query(self,arguments):
  form(arguments);started=time.monotonic();cmd=[sys.executable,'-B',str(Path(__file__).with_name('nvml_query_worker.py')),json.dumps(arguments)]
  try:r=subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=3)
  except(subprocess.TimeoutExpired,subprocess.CalledProcessError)as exc:
   self.record(dict(arguments=arguments,error=type(exc).__name__,timeout_seconds=3,elapsed_seconds=time.monotonic()-started,automatic_retry=False));raise
  raw=json.loads(r.stdout);self.record(dict(arguments=arguments,raw_NVML=raw,elapsed_seconds=time.monotonic()-started,timeout_seconds=3,automatic_retry=False,CLI_used=False))
  return render(raw,arguments)

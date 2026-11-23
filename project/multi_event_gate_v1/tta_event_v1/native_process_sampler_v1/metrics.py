"""Read a fresh fixed NVIDIA interval; no temporal maximum or missing fallback."""
import copy,datetime,re,subprocess,sys,time
from pathlib import Path
T=Path(__file__).resolve().parents[1];P=T.parents[1]
sys.path.insert(0,str(P/'deployment'))
from gpu_metrics_strict import parse as original_parse
_last={}
def parse_frames(raw,index):
 frames=[]
 for line in raw.splitlines():
  if not line.strip()or line.lstrip().startswith('#'):continue
  fields=line.split()
  if len(fields)<7 or not re.fullmatch(r'\d{8}',fields[0])or not re.fullmatch(r'\d{2}:\d{2}:\d{2}',fields[1]):raise ValueError('Malformed timestamped pmon row')
  stamp=datetime.datetime.strptime(fields[0]+' '+fields[1],'%Y%m%d %H:%M:%S')
  if not fields[2].isdigit()or int(fields[2])!=index:raise ValueError('Unexpected pmon GPU index')
  if not frames or stamp!=frames[-1]['timestamp']:
   if frames and stamp<=frames[-1]['timestamp']:raise ValueError('Nonmonotonic pmon timestamps')
   frames.append(dict(timestamp=stamp,lines=[]))
  frames[-1]['lines'].append(' '.join(fields[2:]))
 if len(frames)!=2 or (frames[1]['timestamp']-frames[0]['timestamp']).total_seconds()<1:raise ValueError('Two distinct completed pmon sampling intervals required')
 parsed=[]
 for f in frames:
  values,rows=original_parse('\n'.join(f['lines']));parsed.append(dict(timestamp=f['timestamp'].isoformat(),values=values,components=rows))
 return parsed[-1]['values'],parsed

def process_utilization(index):
 global _last
 if type(index)is not int or not 0<=index<=6:raise ValueError('Physical GPU index required')
 started=time.monotonic();command=['nvidia-smi','pmon','-i',str(index),'-s','u','-c','2','-d','1','-o','DT']
 result=subprocess.run(command,check=True,stdout=subprocess.PIPE,text=True,timeout=3)
 values,frames=parse_frames(result.stdout,index);_last=dict(unix_ns=time.time_ns(),gpu=index,components=frames[-1]['components'],fixed_interval_frames=frames,selected_frame_index=1,raw_pmon=result.stdout,collection_seconds=time.monotonic()-started,timeout_seconds=3,previous_call_data_used=False,device_wide_substitution=False)
 return values

def diagnostics():return copy.deepcopy(_last)

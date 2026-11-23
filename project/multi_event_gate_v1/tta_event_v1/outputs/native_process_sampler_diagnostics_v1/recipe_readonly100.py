from pathlib import Path
import sys,time,json,traceback,signal
T=Path('/data/qiufw/cx_migration/cx_new_direction_guide_TTA/multi_event_gate_v1/tta_event_v1');sys.path.insert(0,str(T));from native_process_sampler_v8 import worker
O=T/'outputs/native_process_sampler_diagnostics_v1';O.mkdir(parents=True,exist_ok=True)
original=worker.frame;current=[]
def frame(h,last_seen):
 wall=time.monotonic();cpu=time.process_time();value=original(h,last_seen);current.append(dict(wall_seconds=time.monotonic()-wall,CPU_seconds=time.process_time()-cpu,actual_full_frame=value));return value
worker.frame=frame
for i in range(100):
 current=[];began=time.monotonic()
 try:
  signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Original3second acquisition deadline')));signal.setitimer(signal.ITIMER_REAL,3)
  value=worker.collect(4);signal.setitimer(signal.ITIMER_REAL,0);row=dict(passed=True,trial=i,seconds=time.monotonic()-began,frames=current,selected=value['selected_frame_indices'])
 except BaseException:
  signal.setitimer(signal.ITIMER_REAL,0);row=dict(passed=False,trial=i,seconds=time.monotonic()-began,frames=current,traceback=traceback.format_exc())
 with(O/'readonly_sampler8_trials.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
 if not row['passed']or i%10==0:print(json.dumps({k:v for k,v in row.items()if k!='frames'}),flush=True)
 if not row['passed']:break
print('READONLY_TRIALS_FINISHED_NO_CUDA_OR_GPU_WORK',flush=True)

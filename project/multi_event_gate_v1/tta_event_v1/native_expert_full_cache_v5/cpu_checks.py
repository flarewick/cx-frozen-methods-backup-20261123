"""Actual prior36 query/trace replay plus native tail/unsupported-phase fixture checks."""
from pathlib import Path
import json,numpy as np,torch
from .paths import T,O,read,write,sha
from .audit import policy_check,trace_check,load
from . import capture as m
from cx_tta_feedback_confirm5.query import build_head
from cx_tta_clock.clock import ACTIVE as CLOCK,ClockComplete
from actuation_telemetry import ACTIVE as TRACE
from stage_feedback_bridge_v1.bridge import ACTIVE as FEEDBACK
class Task:
 def __init__(self):self.take_action_cnt=0;self.eval_success=False;self.step_lim=100;self.command=np.zeros(14,np.float32);self.command[[6,13]]=1
 def take_action(self,command):
  self.take_action_cnt+=1;self.command=command.copy()
  def step():
   row=dict(arm_position_target=np.r_[command[:6],command[7:13]].astype(float),arm_velocity_target=np.arange(12,dtype=float),gripper_logical_command=command[[6,13]].astype(float))
   TRACE.get().append(row,float(np.float32(.004)));CLOCK.get().after_step(command,float(np.float32(.004)))
  for _ in range(3):
   hook=FEEDBACK.get()
   if hook is None:step()
   else:hook.call(step)
 def read(self):return dict(rgb=tuple(np.zeros((240,320,3),np.uint8)for _ in range(3)),command_state=self.command.copy(),physical_sha256='0'*64,rng_sha256='1'*64)
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high');assert not torch.cuda.is_initialized()
 policy=read(T/'outputs/native_expert_policy_v1/protocol.json')
 with torch.random.fork_rng(devices=[]):head,robots=build_head(policy['training'])
 projection=load(policy['projection']);pdir=T/'outputs/native_expert_policy_v1';proots=[p.parent for p in(pdir/'collections').glob('*/manifest.json')];assert len(proots)==1
 rows=read(proots[0]/'GPU_cache_complete.json')['rows'];prior={};checks=0
 for row in rows:
  key=row['case_id'];_,prior[key],n=policy_check(row,load(row['observation_file']),prior.get(key),head,robots,projection);checks+=n
 fdir=T/'outputs/native_expert_frames_v1';froots=[p.parent for p in(fdir/'collections').glob('*/manifest.json')];assert len(froots)==1
 frames=read(froots[0]/'GPU_frames_complete.json')['rows'];spec=read(fdir/'protocol.json');chunks=0;dense=0
 for item,row in zip(spec['scenes'],frames,strict=True):
  source=load(item['file'])['absolute_source_commands']
  for q,chunk in enumerate(row['chunks']):
   _,valid,partial,nframes,n=trace_check(chunk,row['observations'][q],row['observations'][q+1],source[q*32:q*32+32]);chunks+=1;dense+=nframes;checks+=n
 old=m.discard_observation;m.discard_observation=lambda task:None
 try:
  task=Task();commands=np.zeros((7,14),np.float32);commands[:,13]=1
  a,r,d=m.capture(task,commands,reader=task.read,initial=task.read(),kind='close_left',collect_dense=True)
  assert len(r['receipts'])==7 and r['committed_ticks']==21 and r['phase_unavailable']=='partial_expert_reference_chunk_no_complete32_phase'and not r['completion']and d['ticks'].tolist()==[21]
  task=Task();commands=np.ones((32,14),np.float32)
  a,r,d=m.capture(task,commands,reader=task.read,initial=task.read(),kind='close_left',collect_dense=True)
  assert len(r['receipts'])==32 and 'excursion'in r['phase_unavailable']and not r['completion']and d['ticks'].tolist()==[32,64,96]
  task=Task();commands=np.zeros((32,14),np.float32);commands[:,13]=1
  a,r,d=m.capture(task,commands,max_ticks=5,reader=task.read,initial=task.read(),kind='close_left',collect_dense=True)
  assert r['reason']=='measurement_limit'and r['completion'][0]['status']=='censored'and d['ticks'].tolist()==[1,5]
  assert CLOCK.get()is None and TRACE.get()is None and FEEDBACK.get()is None
 finally:m.discard_observation=old
 assert len(rows)==36 and chunks==36 and dense==1007
 write(O/'CPU_interface_checks.json',dict(passed=True,actual_prior_queries=36,actual_prior_chunks=36,actual_prior_arrived_frames=dense,checks=checks,
  new_partial_and_unsupported_and_censor_fixtures_pass=True,new96_GPU_execution_not_claimed=True,CUDA_initialized=False,
  source_sha256={str(p):sha(p)for p in(T/'native_expert_full_cache_v5').rglob('*.py')}))
 print('ORIGINAL96_CPU_REAL_CACHE_AND_TAIL_PASS',checks,flush=True)
if __name__=='__main__':main()

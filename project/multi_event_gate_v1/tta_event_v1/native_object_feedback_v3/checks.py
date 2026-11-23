"""Meaningful CPU contracts. Fixtures are not simulator or efficacy evidence."""
from pathlib import Path
import json,hashlib,itertools,re
import numpy as np
import torch
from .language import new_prompt,prompt
from native_transfer_semantics_v2.semantic import prompt as previous
from native_seven_task_inventory_v1.inventory import R
from .local import LocalFeedbackStream,old
from native_delayed_query_v1.local_feedback import partner_features
T=Path(__file__).resolve().parents[1];O=T/'outputs/native_object_feedback_v3'
def language_check():
 models={'beat_block_hammer':('020_hammer',[0]),'turn_switch':('056_switch',range(8)),'adjust_bottle':('001_bottle',[13,16]),'grab_roller':('102_roller',[0,2]),'open_laptop':('015_laptop',range(11))};counts={};bound={}
 def load(p):bound[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
 for task,(model,ids)in models.items():
  desc=[]
  for i in ids:
   d=load(R/'description/objects_description'/model/('base'+str(i)+'.json'));desc.extend('the '+v for v in(d.get('unseen')or d['seen']))
  pars={'A':desc,'a':['the left arm','the right arm']};total=supported=0
  for template in load(R/'description/task_instruction'/(task+'.json'))['unseen']:
   keys=sorted(set(re.findall(r'{([^}]+)}',template)))
   for values in itertools.product(*(pars[k]for k in keys)):
    text=template
    for k,v in zip(keys,values):text=text.replace('{'+k+'}',v)
    roles,phrases=prompt(text);total+=1;supported+=bool(phrases)
    assert roles['actual_object_count']==len(phrases)in(1,2)
  assert supported==total;counts[task]=dict(supported=supported,total=total)
 old=['Stack the red bowl on the blue bowl.','Stack the two blocks.','Place the mouse onto the pad.','Put the fan on the mat.','Place the bread in the skillet.','Move the stapler onto the pad.']
 for text in old:assert prompt(text)==previous(text)
 for text in ['Move that.','Grab the bottle beside the laptop.','Open the computer.','Move the block onto the switch.']:assert not new_prompt(text)[1]
 return dict(counts=counts,old_dispatch_examples=len(old),negative_language_cases=4,input_sha256=bound)
def local_check():
 # Real OpenCV corner/flow functions; only hardware FK is a CPU fixture.
 saved={k:getattr(old,k)for k in('tcp','context','action')}
 old.tcp=lambda *a:np.asarray([[[140.,120.],[260.,200.]]])
 old.context=lambda *a:np.zeros(44)
 old.action=lambda *a:np.zeros(197)
 try:
  rng=np.random.default_rng(723);rgb=np.repeat(rng.integers(35,220,(240,320,1),dtype=np.uint8),3,axis=2);m=np.zeros((2,240,320),bool);m[0,90:150,110:170]=True;presence=np.array([1.,0.]);q=np.zeros(14)
  v=LocalFeedbackStream(rgb,m,'Grab the roller.',robots=None,cfg=None,camera=None,real_object_slots=np.array([True,False]));v.start(rgb,m,presence,q,tick=0,elapsed_s=0.)
  assert v.previous['valid'].tolist()==[True,False]and v.previous['good'][0].sum()>=3
  assert not v.previous['mask'][1].any()and not v.previous['good'][1].any()
  initial=v.previous['p'][0,v.previous['good'][0]];assert np.all(np.linalg.norm(initial-[140.,120.],axis=1)<=48.001)
  moved=np.roll(rgb,2,axis=1);mm=np.roll(m,2,axis=2);window=dict(arm_position_target=np.zeros((25,12)),arm_velocity_target=np.zeros((25,12)),gripper_logical_command=np.zeros((25,2)),dt_s=np.full(25,.004))
  rows,missing=v.arrive(moved,mm,presence,window,q,tick=25,elapsed_s=.1,segment='origin',segment_start_tick=0)
  assert len(rows)==1 and rows[0]['object_index']==0,(rows,missing)
  row=rows[0];assert np.allclose(row['target'],[2/320,0],atol=.0003),row['target']
  assert row['context'].shape==(53,)and row['command'].shape==(197,)
  assert np.array_equal(row['context'][-9:],np.zeros(9))and not row['relative_available']and np.isnan(row['relative_target']).all()
  assert len(missing)==1 and missing[0]['object_index']==1
  bad=dict(window,dt_s=np.full(25,.005));previous_frame=v.previous
  try:v.arrive(rgb,m,presence,bad,q,tick=50,elapsed_s=.2,segment='following',segment_start_tick=25)
  except ValueError as e:assert str(e)=='Only the exact arrived own-command prefix'
  else:raise AssertionError('Incorrect actual duration accepted')
  assert v.previous is previous_frame and v.rows==1
  # The ROI's first frame must use the new path, including gray objects.
  qbad=q.copy();qbad[0]=1
  try:v.arrive(rgb,m,presence,window,qbad,tick=50,elapsed_s=.2,segment='following',segment_start_tick=25)
  except ValueError as e:assert str(e)=='Current own-command state and committed prefix disagree'
  else:raise AssertionError('Borrowed command accepted')
  return dict(real_OpenCV_achromatic_flow=True,observed_target=row['target'].tolist(),actual_one_target_only=True,absent_partner_zero_and_relative_missing=True,trace_duration_and_command_mismatch_rejected=True,initial_frame_new_ROI=True,fixture_hardware_geometry_only=True)
 finally:
  for k,v in saved.items():setattr(old,k,v)
def main():
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();value=dict(language=language_check(),local=local_check(),GPU_started=False,scientific_efficacy_claim=False)
 O.mkdir(parents=True,exist_ok=True);p=O/'CPU_interface_checks.json'
 with p.open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(passed=True,language=value['language']['counts'],local=value['local'],CUDA_initialized=torch.cuda.is_initialized())))
if __name__=='__main__':main()

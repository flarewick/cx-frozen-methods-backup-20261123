"""CPU-only synthetic control-flow fixture, never experiment data."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .common import O,TASKS,read,write,sha

def check():
 from . import screen
 from .future_preparation import prepare,audit
 from .prepared_recovery import collect
 root=O/'CPU_synthetic_future_fixture_v2';root.mkdir(parents=True,exist_ok=False)
 state={t:dict(accepted=3,next_seed=100010)for t in TASKS};write(root/'screening_complete.json',dict(state=state))
 write(root/'protocol.json',dict(CPU_fixture_only=True));write(root/'terminal_audit.json',dict(passed=True,CPU_fixture_only=True))
 oldspec=dict(future_preparation_each_task=1,episodes_per_task=3,tasks=list(TASKS));oldcallback=object()
 class Schedule:
  def __init__(self):self.root=root;self.spec=oldspec;self.on_initial=oldcallback;self.indices={t:dict(x)for t,x in state.items()}
  def next(self,*a):raise AssertionError('Atomic/cache primary next must not be used for pure future preparation')
  def finish(self):write(self.root/'screening_complete.json',dict(state=self.indices))
  def allow_future(self):return True
 schedule=Schedule();saved=screen.Schedule.next;calls=[]
 def next_explicit(self,name):
  assert self.root==root/'next_preparation'and self.spec['episodes_per_task']==4
  st=self.indices[name];seed=st['next_seed'];index=st['accepted'];instruction='CPU_fixture_only '+name
  task=SimpleNamespace(set_instruction=lambda **kw:None)
  self.on_initial(task,instruction,name,seed,index)
  case=name+':'+str(seed);item=dict(task=name,seed=seed,episode_index=index,instruction=instruction,case_id=case,initial_physical_sha256='CPU_fixture_physics')
  write(self.root/'accepted'/(case+'.json'),item)
  write(self.root/'screening'/name/(str(seed)+'.json'),dict(task=name,seed=seed,episode_index=index,eligible=True,expert_plan_success=True,expert_success=True,initial_physical_sha256='CPU_fixture_physics'))
  command=self.root/'expert_commands'/(case+'.npz');command.parent.mkdir(exist_ok=True)
  with command.open('xb')as f:np.savez_compressed(f,CPU_fixture_only=np.ones((1,),np.int8))
  write(self.root/'expert_references'/(case+'.json'),dict(case_id=case,complete=True,expert_plan_success=True,expert_task_success=True,file=str(command),sha256=sha(command),CPU_fixture_only=True))
  st['next_seed']+=1;st['accepted']+=1;calls.append((name,index));return item
 screen.Schedule.next=next_explicit
 try:
  observe=lambda task:dict(rgb=[np.zeros((240,320,3),np.uint8)for _ in range(3)],command_state=np.zeros(14,np.float32),native_action_count=0,physical_sha256='CPU_fixture_physics',rng_sha256='CPU_fixture_rng')
  prepare(schedule,SimpleNamespace(check_cancel=lambda:None),root,oldspec,observe,lambda t:None,lambda t:None)
 finally:screen.Schedule.next=saved
 assert schedule.root==root and schedule.spec is oldspec and schedule.on_initial is oldcallback
 assert calls==[(t,3)for t in TASKS]
 assert audit(root,oldspec)['passed']
 cache=collect((root,{}),state,1);assert cache['cached_valid_cases']==len(TASKS)and not cache['primary_policy_results_imported']and not cache['old_masks_or_teacher_outputs_imported']
 assert all('/next_preparation/'in p or p.endswith(('protocol.json','terminal_audit.json'))for p in cache['input_sha256'])
 return True

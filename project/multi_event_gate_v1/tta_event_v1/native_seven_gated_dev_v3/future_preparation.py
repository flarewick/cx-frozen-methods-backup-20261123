"""Only required next fixed TRAIN/VAL experts and passive q0 inputs, no actor."""
import numpy as np
from .common import read,write,sha,verify,mapped
from .semantic import content_key
def prepare(schedule,c,root,spec,observe,install_render,discard_observation):
 count=spec['future_preparation_each_task'];assert count in(1,2);current=read(root/'screening_complete.json')['state'];end=spec['episodes_per_task'];assert all(v['accepted']==end for v in current.values())
 future=root/'next_preparation';future.mkdir(parents=True,exist_ok=False);old_callback=schedule.on_initial
 schedule.root=future;schedule.spec=dict(spec,episodes_per_task=end+count)
 write(future/'initial_state.json',dict(state=current,first_index=end,last_exclusive=end+count,required_next_fixed_dataset_work=True,no_policy_queries_or_masks=True))
 def export_initial(task,instruction,name,seed,index):
  install_render(task);task.set_instruction(instruction=instruction);discard_observation(task);initial=observe(task);assert initial['native_action_count']==0 and end<=index<end+count
  key=content_key(initial['rgb'],initial['command_state'],instruction);f=future/'semantic/frames'/(key+'.npz');f.parent.mkdir(parents=True,exist_ok=True)
  with f.open('xb')as h:np.savez_compressed(h,**{'cam'+str(i):v for i,v in enumerate(initial['rgb'])},command_state=initial['command_state'])
  write(future/'semantic/requests'/(key+'.json'),dict(frame_file=str(f),frame_sha256=sha(f),instruction=instruction,content_key=key))
  write(future/'initial_prefetch'/(name+':'+str(seed)+'.json'),dict(content_key=key,only_RGB_own_command_language_submitted=True,passive_physics_action_count_RNG_unchanged=True,no_CPU_answer_wait=True,diagnostic_identity={k:initial[k]for k in('physical_sha256','rng_sha256')}))
 schedule.on_initial=export_initial;items=[]
 try:
  for _ in range(count):
   for task in spec['tasks']:
    if not getattr(schedule,'allow_future',lambda:True)():break
    c.check_cancel();item=schedule.next(task);assert end<=item['episode_index']<end+count;items.append(item)
   else:continue
   break
  schedule.finish();assert len(items)<=len(spec['tasks'])*count
  files=[p for p in future.rglob('*')if p.is_file()];write(future/'preparation_complete.json',dict(complete=True,expert_valid_inputs=len(items),planned_maximum=len(spec['tasks'])*count,first_index=end,last_exclusive=end+count,not_dispatched_inputs_remain_required_in_next_fixed_stage=True,stopped_dispatch_before_current_full_tail=len(items)<len(spec['tasks'])*count,required_future_fixed_dataset_work=True,no_full_policy_outputs_masks_or_teacher_targets=True,input_sha256={str(f):sha(f)for f in files}));print('NEXT_FIXED_STAGE_REAL_PREPARATION_COMPLETE',len(items),flush=True)
 finally:schedule.on_initial=old_callback

def audit(root,spec):
 count=spec['future_preparation_each_task']
 if not count:return dict(passed=True,required_future_cases=0)
 future=root/'next_preparation';value=read(future/'preparation_complete.json');assert value['complete'];verify(value);items=[read(f)for f in(future/'accepted').glob('*.json')];end=spec['episodes_per_task'];assert len(items)==value['expert_valid_inputs']<=count*len(spec['tasks'])
 assert value['planned_maximum']==count*len(spec['tasks'])and value['not_dispatched_inputs_remain_required_in_next_fixed_stage']
 expected=[(task,i)for i in range(end,end+count)for task in spec['tasks']][:len(items)]
 assert {(i['task'],i['episode_index'])for i in items}==set(expected)
 for task in spec['tasks']:
  indices=sorted(i['episode_index']for i in items if i['task']==task)
  assert indices==list(range(end,end+len(indices)))
 for item in items:
  meta=read(future/'initial_prefetch'/(item['case_id']+'.json'));req=read(future/'semantic/requests'/(meta['content_key']+'.json'));assert sha(mapped(req['frame_file']))==req['frame_sha256']and req['instruction']==item['instruction']
  with np.load(mapped(req['frame_file']),allow_pickle=False)as z:assert content_key([z['cam'+str(i)]for i in range(3)],z['command_state'],item['instruction'])==req['content_key']
  ref=read(future/'expert_references'/(item['case_id']+'.json'));assert ref['expert_plan_success']and ref['expert_task_success']and sha(mapped(ref['file']))==ref['sha256']
 out=dict(passed=True,required_future_cases=len(items),no_policy_outputs_or_masks_imported=True,input_sha256={str(future/'preparation_complete.json'):sha(future/'preparation_complete.json'),**value['input_sha256']});write(root/'CPU_future_preparation_audit.json',out);return out

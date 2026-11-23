"""Reuse only independently validated pure inputs; fresh screening for laptop."""
from pathlib import Path
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify

def proof(spec):
 file=T/'outputs/native_fixed_FIT_inputs_v1/verified_old_FIT0_prefix_v1.json'
 assert mapped(spec['carried_FIT0_input_proof']).resolve()==file.resolve()
 value=read(file);verify(value)
 assert value['passed']and value['completed_pure_initial_inputs']==4 and value['missing_tasks']==['open_laptop']
 assert value['failed_parent_scientific_outputs_NOT_qualified']and value['no_policy_masks_or_response_teacher_labels_imported']
 assert value['initial_state_after_carried_prefix']['open_laptop']==dict(next_seed=7124000,accepted=0)
 return file,value

def install(schedule,spec):
 if not spec.get('carried_FIT0_input_proof'):return
 assert spec['dataset_split']=='adapter_validation'and spec['future_preparation_population']==dict(split='fit',seed_start=7124000,first_index=0)
 assert spec['future_preparation_each_task']==1
 source,value=proof(spec);items={row['task']:row for row in value['items']};original=schedule.next;used=set()
 def next_case(name):
  if name not in items:return original(name)
  assert name not in used and schedule.state[name]['accepted']==0 and schedule.state[name]['next_seed']==7124000
  item=dict(items[name]);prefix=mapped(item.pop('source_prefix'));root=schedule.root
  assert root.name=='next_preparation'and root.is_relative_to(T/'outputs')
  bindings={str(source):sha(source)}
  def copy_json(old,new):
   old=mapped(old);write(new,read(old));bindings[str(old)]=sha(old)
  rows=sorted((prefix/'screening'/name).glob('*.json'),key=lambda p:int(p.stem))
  expected=7124000
  for file in rows:
   row=read(file);assert row['seed']==expected and row['episode_index']==0;expected+=1
   copy_json(file,root/'screening'/name/file.name)
   if row['eligible']:break
  assert expected==value['initial_state_after_carried_prefix'][name]['next_seed']
  case=item['case_id'];copy_json(prefix/'accepted'/(case+'.json'),root/'accepted'/(case+'.json'))
  meta=read(prefix/'initial_prefetch'/(case+'.json'));key=meta['content_key']
  for path in ['initial_prefetch/'+case+'.json','semantic/requests/'+key+'.json','expert_references/'+case+'.json']:
   copy_json(prefix/path,root/path)
  schedule.state[name].update(value['initial_state_after_carried_prefix'][name]);schedule.state[name]['rows']=[read(p)for p in rows if int(p.stem)<expected]
  used.add(name)
  write(root/'carried_inputs'/(case+'.json'),dict(passed=True,case_id=case,pure_initial_input_only=True,
   no_policy_masks_or_response_teacher_labels_imported=True,input_sha256=bindings))
  return item
 schedule.next=next_case

def audit(root,spec):
 if not spec.get('carried_FIT0_input_proof'):return None
 source,value=proof(spec);future=root/'next_preparation';done=read(future/'preparation_complete.json');verify(done)
 assert done['complete']and done['expert_valid_inputs']==done['planned_maximum']==5
 assert done['first_index']==0 and done['last_exclusive']==1
 bindings={str(source):sha(source),str(future/'preparation_complete.json'):sha(future/'preparation_complete.json')}
 for item in value['items']:
  p=future/'carried_inputs'/(item['case_id']+'.json');v=read(p);verify(v)
  assert v['passed']and v['pure_initial_input_only']and v['no_policy_masks_or_response_teacher_labels_imported']
  bindings[str(p)]=sha(p);bindings.update(v['input_sha256'])
  old=mapped(item['source_prefix']);case=item['case_id']
  for path in ['accepted/'+case+'.json','initial_prefetch/'+case+'.json','expert_references/'+case+'.json']:
   assert read(future/path)==read(old/path);bindings[str(future/path)]=sha(future/path)
 laptop=list((future/'accepted').glob('open_laptop:*.json'));assert len(laptop)==1
 fresh=read(laptop[0]);assert fresh['task']=='open_laptop'and fresh['episode_index']==0 and fresh['seed']>=7124000
 assert not list((future/'carried_inputs').glob('open_laptop:*.json'))
 switch=sorted((future/'screening/turn_switch').glob('*.json'),key=lambda p:int(p.stem));assert [int(p.stem)for p in switch]==[7124000,7124001,7124002,7124003]
 assert [read(p)['eligible']for p in switch]==[False,False,False,True]
 result=dict(passed=True,carried_pure_inputs=4,fresh_required_task='open_laptop',all_required_inputs=5,
  original_negative_validity_rows_preserved=True,no_failed_policy_masks_or_labels_imported=True,input_sha256=bindings)
 write(root/'CPU_carried_FIT0_inputs_audit.json',result);return result

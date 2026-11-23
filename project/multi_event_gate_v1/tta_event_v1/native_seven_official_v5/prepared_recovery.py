"""Only real next-stage expert/q0 work from the preceding qualified batch."""
from .common import T,P,CANON,read,mapped,sha,verify,TASKS
from native_seven_plain_dev_v3.preparation import validate_records

def collect(previous,initial,count):
 if previous is None:return None
 root,spec=previous;future=root/'next_preparation'
 if not future.exists():return None
 complete=read(future/'preparation_complete.json');assert complete['complete'];verify(complete)
 assert complete['required_future_fixed_dataset_work']and complete['no_full_policy_outputs_masks_or_teacher_targets']
 original=read(future/'initial_state.json');assert original['state']==initial
 canonical=lambda p:str(CANON/p.resolve().relative_to(P))
 deps={canonical(p):sha(p)for p in(future/'preparation_complete.json',future/'initial_state.json',root/'protocol.json',root/'terminal_audit.json')}
 tasks={};requests=[];keys={};diagnostics={};references={}
 for task in TASKS:
  rows=sorted([read(p)for p in(future/'screening'/task).glob('*.json')],key=lambda r:r['seed'])
  items=sorted([read(p)for p in(future/'accepted').glob(task+':*.json')],key=lambda r:r['episode_index'])[:count]
  if not items:continue
  rows=[r for r in rows if r['seed']<=items[-1]['seed']];entry=validate_records(initial[task],rows,items,task);tasks[task]=entry
  for row in rows:
   p=future/'screening'/task/(str(row['seed'])+'.json');deps[canonical(p)]=sha(p)
  for item in items:
   i=item['case_id'];a=future/'accepted'/(i+'.json');receipt=future/'initial_prefetch'/(i+'.json');meta=read(receipt);key=meta['content_key'];req=future/'semantic/requests'/(key+'.json');request=read(req);frame=mapped(request['frame_file']);reference=future/'expert_references'/(i+'.json');teacher=read(reference);commands=mapped(teacher['file'])
   assert teacher['complete']and teacher['expert_plan_success']and teacher['expert_task_success']and sha(commands)==teacher['sha256']
   assert set(request)=={'frame_file','frame_sha256','instruction','content_key'}and request['instruction']==item['instruction']and request['content_key']==key and sha(frame)==request['frame_sha256']
   assert meta['only_RGB_own_command_language_submitted']and meta['passive_physics_action_count_RNG_unchanged']
   for p in(a,receipt,req,frame,reference,commands):deps[canonical(p)]=sha(p)
   requests.append(request);keys[i]=key;diagnostics[i]=meta['diagnostic_identity'];references[i]=canonical(reference)
 if not requests:return None
 assert len({r['content_key']for r in requests})==len(requests)
 assert all(not'/cases/'in p and not'/answers/'in p for p in deps)
 return dict(version='qualified_previous_future_inputs_only_v1',source_roots=[canonical(future)],batch_index=0,initial_screen_state=initial,task_cache=tasks,requests=requests,case_request_keys=keys,case_diagnostics=diagnostics,expert_reference_files=references,cached_valid_cases=len(requests),completed_expert_rows=sum(len(v['rows'])for v in tasks.values()),primary_policy_results_imported=False,old_masks_or_teacher_outputs_imported=False,input_sha256=deps)

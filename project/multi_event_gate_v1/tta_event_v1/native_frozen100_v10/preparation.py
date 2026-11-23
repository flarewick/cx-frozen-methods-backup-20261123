"""Reuse verified completed preparation only; never read primary case outcomes."""
from pathlib import Path
from .common import T,P,O,CANON,read,sha,mapped,verify
SIGNATURE=('scope','tasks','official_seed_group','actual_seed_start','candidate_checkpoint_sha256','trained_qualification','training','runtime_source','projection','event_config','actor_sampler_seed','perception_interface','conditions')
def canonical(path):return str(CANON/Path(path).resolve().relative_to(P))
def validate_records(initial,rows,items,task):
 state=dict(initial);accepted=[];byseed={x['seed']:x for x in items}
 assert len(byseed)==len(items)
 prefix=[]
 for row in rows:
  assert row['task']==task and row['seed']==state['next_seed']and row['episode_index']==state['accepted'],'Noncontiguous expert preparation'
  if row['eligible']:
   # A completed expert row without a generated instruction is not reusable.
   if row['seed']not in byseed:break
   item=byseed[row['seed']]
   assert row['expert_plan_success']and row['expert_success']
   assert set(item)=={'task','seed','episode_index','instruction','case_id','initial_physical_sha256'}
   assert item['task']==task and item['seed']==row['seed']and item['episode_index']==state['accepted']
   assert item['initial_physical_sha256']==row['initial_physical_sha256']and item['case_id']==task+':'+str(row['seed'])and isinstance(item['instruction'],str)and item['instruction']
   accepted.append(item);state['accepted']+=1
  state['next_seed']+=1;prefix.append(row)
  assert state['accepted']<=initial['accepted']+4
 assert len(accepted)==len(items),'Accepted item lacks a complete contiguous expert record'
 return dict(initial_state=initial,end_state=state,rows=prefix,items=accepted)
def collect_root(root,study,batch,initial):
 root=Path(root);proto=root/'protocol.json';old=read(proto)
 assert old['batch_index']==batch and old['initial_screen_state']==initial
 assert all(old[k]==study[k]for k in SIGNATURE),'Preparation scientific signature changed'
 verify(old)
 resources=root/'attempt/resources.jsonl';records=[__import__('json').loads(x)for x in resources.read_text().splitlines()]
 assert records[-1].get('own_gpu_released')and not records[-1]['remaining_own_gpu_pids'],'Old GPU subtree not released'
 assert not(root/'terminal_audit.json').exists(),'Qualified results must be reused whole, not as preparation'
 deps={canonical(proto):sha(proto),canonical(resources):sha(resources)};tasks={};requests=[];keys=set();case_keys={}
 for task in study['tasks']:
  files=sorted((root/'screening'/task).glob('*.json'),key=lambda p:int(p.stem));rows=[read(p)for p in files]
  accepted_files=sorted((root/'accepted').glob(task+':*.json'),key=lambda p:read(p)['episode_index']);items=[read(p)for p in accepted_files]
  entry=validate_records(initial[task],rows,items,task);tasks[task]=entry
  for p in files[:len(entry['rows'])]+accepted_files:deps[canonical(p)]=sha(p)
  for item in entry['items']:
   receipt=root/'initial_prefetch'/(item['case_id']+'.json')
   # Only actual pre-command initial frames, with four legal input fields.
   assert receipt.exists(),'Completed cached case lacks an initial input receipt'
   meta=read(receipt);key=meta['content_key'];req=root/'semantic/requests'/(key+'.json')
   assert req.exists(),'Completed cached case lacks its legal initial input request'
   row=read(req);assert set(row)=={'frame_file','frame_sha256','instruction','content_key'}and row['content_key']==key and row['instruction']==item['instruction']
   frame=mapped(row['frame_file']).resolve();prior=(old.get('cached_preparation')or{}).get('requests',[]);assert (frame.is_relative_to((root/'semantic/requests').resolve())or(row in prior and frame.is_relative_to((T/'outputs').resolve())))and frame.suffix=='.npz'
   assert sha(frame)==row['frame_sha256']and key not in keys
   keys.add(key);requests.append(row);case_keys[item['case_id']]=key
   for p in(receipt,req,frame):deps[canonical(p)]=sha(p)
 n=sum(len(v['items'])for v in tasks.values());count=sum(len(v['rows'])for v in tasks.values());assert n<=8 and len(requests)==n
 return dict(version='verified_completed_preparation_v1',source_root=canonical(root),source_protocol_identity=old['identity'],batch_index=batch,initial_screen_state=initial,task_cache=tasks,requests=requests,case_request_keys=case_keys,cached_valid_cases=n,completed_expert_rows=count,primary_policy_results_imported=False,old_masks_or_teacher_outputs_imported=False,input_sha256=deps)
def collect(study,batch,initial):
 candidates=[]
 for base in(T/'outputs/native_frozen100_v1'/__import__('os').environ.get('CX_TTA100_COHORT','pads'),T/'outputs/native_frozen100_v2'/__import__('os').environ.get('CX_TTA100_COHORT','pads'),O):
  for root in(base/'batches'/('%03d'%batch)/'runs').glob('*'):
   if(root/'terminal_audit.json').exists()or not(root/'attempt/resources.jsonl').exists():continue
   log=root/'GPU_host.log'
   if log.exists()and 'OIDN Error:'in log.read_text(errors='replace'):continue # Entire failed render attempt is unsuitable as a q0 source
   import json
   rows=[json.loads(x)for x in(root/'attempt/resources.jsonl').read_text().splitlines()]
   if not rows or not rows[-1].get('own_gpu_released'):continue
   value=collect_root(root,study,batch,initial)
   if value['completed_expert_rows']:candidates.append(value)
 # Selection depends only on completed preparation, never policy outcomes.
 return max(candidates,key=lambda v:(v['cached_valid_cases'],v['completed_expert_rows'],v['source_root']))if candidates else None

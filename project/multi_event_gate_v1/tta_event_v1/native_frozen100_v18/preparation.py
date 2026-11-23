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
  assert state['accepted']<=initial['accepted']+5
 assert len(accepted)==len(items),'Accepted item lacks a complete contiguous expert record'
 return dict(initial_state=initial,end_state=state,rows=prefix,items=accepted)
def collect(study,batch,initial):
 import json
 selected={};slots=study['mixed_task_slots'][str(batch)]
 # Only new logical groups can match; old roots are read solely for expert/q0 preparation.
 bases=[T/('outputs/native_frozen100_v'+str(v))/c for v in range(1,int(O.parent.name.rsplit('_v',1)[1])+1)for c in('pads','placement','mixed')]
 roots=[r for base in bases for r in(base/'batches').glob('*/runs/*')if(r/'attempt/resources.jsonl').exists()and not(r/'terminal_audit.json').exists()]
 for root in roots:
  old=read(root/'protocol.json')
  if any(old[k]!=study[k]for k in SIGNATURE if k!='tasks'):continue
  if not any(old['initial_screen_state'].get(t)==initial[t]for t in study['tasks']):continue
  if 'OIDN Error:'in(root/'GPU_host.log').read_text(errors='replace'):continue
  for task in study['tasks']:
   if task not in old['initial_screen_state']or old['initial_screen_state'][task]!=initial[task]:continue
   needed=slots.count(task)
   if not needed:continue
   rows=[read(p)for p in sorted((root/'screening'/task).glob('*.json'),key=lambda p:int(p.stem))]
   items=sorted([read(p)for p in(root/'accepted').glob(task+':*.json')],key=lambda x:x['episode_index'])[:needed]
   ready=[]
   for item in items:
    if not(root/'initial_prefetch'/(item['case_id']+'.json')).exists():break
    ready.append(item)
   if not ready:continue
   rows=[r for r in rows if r['seed']<=ready[-1]['seed']];entry=validate_records(initial[task],rows,ready,task)
   if task in selected and (len(selected[task]['entry']['items']),len(selected[task]['entry']['rows']))>=(len(ready),len(rows)):continue
   resource=root/'attempt/resources.jsonl';records=[json.loads(x)for x in resource.read_text().splitlines()];assert records[-1].get('own_gpu_released')and not records[-1]['remaining_own_gpu_pids']
   verify(old);deps={canonical(root/'protocol.json'):sha(root/'protocol.json'),canonical(resource):sha(resource)};requests=[];keys={};diagnostics={}
   for row in rows:
    p=root/'screening'/task/(str(row['seed'])+'.json');deps[canonical(p)]=sha(p)
   for item in ready:
    p=root/'accepted'/(item['case_id']+'.json');deps[canonical(p)]=sha(p)
    receipt=root/'initial_prefetch'/(item['case_id']+'.json');meta=read(receipt);key=meta['content_key'];req=root/'semantic/requests'/(key+'.json');request=read(req)
    assert set(request)=={'frame_file','frame_sha256','instruction','content_key'}and request['instruction']==item['instruction']and request['content_key']==key
    frame=mapped(request['frame_file']).resolve();assert frame.is_relative_to((T/'outputs').resolve())and frame.suffix=='.npz'and sha(frame)==request['frame_sha256']
    for p in(receipt,req,frame):deps[canonical(p)]=sha(p)
    requests.append(request);keys[item['case_id']]=key
    if meta.get('diagnostic_identity')is not None:diagnostics[item['case_id']]=meta['diagnostic_identity']
   selected[task]=dict(entry=entry,requests=requests,keys=keys,diagnostics=diagnostics,deps=deps,root=canonical(root))
 if not selected:return None
 tasks={k:v['entry']for k,v in selected.items()};requests=[r for v in selected.values()for r in v['requests']];assert len({r['content_key']for r in requests})==len(requests)
 return dict(version='verified_mixed_task_preparation_v1',source_roots=[v['root']for v in selected.values()],batch_index=batch,initial_screen_state=initial,task_cache=tasks,requests=requests,case_request_keys={k:x for v in selected.values()for k,x in v['keys'].items()},case_diagnostics={k:x for v in selected.values()for k,x in v['diagnostics'].items()},cached_valid_cases=len(requests),completed_expert_rows=sum(len(v['rows'])for v in tasks.values()),primary_policy_results_imported=False,old_masks_or_teacher_outputs_imported=False,input_sha256={k:x for v in selected.values()for k,x in v['deps'].items()})

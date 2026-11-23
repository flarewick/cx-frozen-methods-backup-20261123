"""Three fixed first TRAIN members; all time from failed density probe retained."""
import copy,json
from .common import T,read,mapped,sha,verify

def recovered(spec,initial):
 from native_seven_plain_density_v2.carry import recovered as previous
 from .preparation import validate_records
 value,spent,deps=previous(spec,initial);assert spent==3016.970
 b=T/'outputs/native_seven_plain_density_v2/plain_000';a=read(b/'actual_host_wait_exit.json')
 assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 root=mapped(read(b/'execution_state.json')['root']);old=read(root/'protocol.json');verify(old)
 assert old['dataset_split']=='TRAIN'and old['actual_seed_start']==7136000
 assert read(root/'failure.json')['traceback'].endswith('RuntimeError: Rule4: rolling180s GPU utilization/VRAM\n')
 r=json.loads((root/'attempt/resources.jsonl').read_text().splitlines()[-1]);assert r['own_gpu_released']and not r['remaining_own_gpu_pids']and r['cleanup_seconds_after_start']==771.612
 assert not(root/'terminal_audit.json').exists()
 value=copy.deepcopy(value);members=set();entries={}
 for task,entry in value['task_cache'].items():
  items=entry['items'][:1];assert len(items)==1 and items[0]['episode_index']==0
  rows=[row for row in entry['rows']if row['seed']<=items[0]['seed']]
  entries[task]=validate_records(initial[task],rows,items,task);members.add(items[0]['case_id'])
 value['task_cache']=entries
 for name in('case_request_keys','case_diagnostics','expert_reference_files'):value[name]={k:v for k,v in value[name].items()if k in members}
 keys=set(value['case_request_keys'].values());value['requests']=[x for x in value['requests']if x['content_key']in keys]
 value['cached_valid_cases']=len(value['requests']);value['completed_expert_rows']=sum(len(e['rows'])for e in entries.values());assert value['cached_valid_cases']==3
 for p in(b/'actual_host_wait_exit.json',root/'protocol.json',root/'attempt/resources.jsonl',root/'failure.json'):deps[str(p)]=sha(p)
 value['input_sha256'].update(deps)
 spent=round(spent+771.612,3);assert spent==3788.582 and round(4926.847-spent,3)==1138.265
 return value,spent,deps

def check():
 from .qualification import qualified_stage
 from .reference_order import include_reference
 root,old=qualified_stage(T/'outputs/native_seven_plain_dev_v3/dev_plain');value,spent,deps=recovered(old,old['initial_screen_state'])
 assert len(value['requests'])==3 and all(len(e['items'])==1 for e in value['task_cache'].values())
 assert {i['case_id']for e in value['task_cache'].values()for i in e['items']}=={'lift_pot:7136000','handover_mic:7136000','stack_bowls_two:7136000'}
 for reference in(False,True):
  branches=['M_event_full','TTA']+(['handoff_reference']if reference else[])
  assert include_reference(branches)==['M_event_full','handoff_reference','TTA']
 return True

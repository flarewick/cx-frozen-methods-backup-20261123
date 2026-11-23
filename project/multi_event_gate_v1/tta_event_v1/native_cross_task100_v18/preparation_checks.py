"""Real seven-case expert-prefix/input binding, negative and original-body checks."""
import ast,copy,threading
from pathlib import Path
import numpy as np,torch
from unittest.mock import patch
from .common import C,T,O,read,write,sha,mapped
from . import preparation
from .cached_screen import schedule_class
from .semantic import content_key
from .prefetch import PrefetchClient
def main():
 old=T/'outputs/native_cross_task100_v13/batches/004/runs/539fba80dbc1420f85329525be8b4b48';study=read(T/'outputs/native_cross_task100_v11/study.json');spec=read(old/'protocol.json');initial=spec['initial_screen_state'];parsed=[]
 def allowed(path):
  p=Path(path);assert 'cases'not in p.parts and p.name!='result.json','Primary outcomes parsed';parsed.append(str(p));return read(p)
 with patch.object(preparation,'read',allowed):cache=preparation.collect_root(old,study,4,initial)
 assert cache['cached_valid_cases']==8 and cache['completed_expert_rows']>=13 and len(cache['requests'])==8
 assert not cache['primary_policy_results_imported']and not cache['old_masks_or_teacher_outputs_imported']
 assert all(sha(mapped(p))==v for p,v in cache['input_sha256'].items())
 class Original:
  def __init__(self,c,prepared,spec,root,planner):
   self.state={task:dict(spec['initial_screen_state'][task],rows=[])for task in spec['tasks']};self.locks={task:threading.Lock()for task in spec['tasks']};self.root=root;self.fallback=[]
  def next(self,name):
   state=self.state[name];value=dict(task=name,seed=state['next_seed'],episode_index=state['accepted'],CPU_fixture_original_fallback=True);self.fallback.append(value);return value
 root=O/'CPU_cached_schedule_fixture';s=dict(spec,cached_preparation=cache);schedule=schedule_class(Original)(None,None,s,root,None)
 expected={task:iter(cache['task_cache'][task]['items'])for task in spec['tasks']}
 for n in range(8):
  task=spec['tasks'][n%2];item=schedule.next(task)
  assert item==next(expected[task])
 assert not schedule.fallback
 for task in spec['tasks']:assert all(schedule.state[task][k]==v for k,v in cache['task_cache'][task]['end_state'].items())
 task=spec['tasks'][0];entry=cache['task_cache'][task];bad=copy.deepcopy(entry['rows']);bad[0]['seed']+=1
 try:preparation.validate_records(initial[task],bad,entry['items'],task)
 except AssertionError:pass
 else:raise AssertionError('Gap accepted')
 bad=copy.deepcopy(entry['items']);bad[0]['initial_physical_sha256']='wrong'
 try:preparation.validate_records(initial[task],entry['rows'],bad,task)
 except AssertionError:pass
 else:raise AssertionError('Wrong initial physics accepted')
 truncated=preparation.validate_records(initial[task],entry['rows'],entry['items'][:-1],task);assert truncated['end_state']['next_seed']==entry['items'][-1]['seed']and truncated['end_state']['accepted']==19
 try:preparation.collect_root(old,dict(study,candidate_checkpoint_sha256='wrong'),4,initial)
 except AssertionError:pass
 else:raise AssertionError('Wrong frozen method accepted')
 preload=O/'CPU_cached_request_fixture';client=PrefetchClient(preload)
 for req in cache['requests']:
  write(preload/'requests'/(req['content_key']+'.json'),req);write(preload/'answers'/(req['content_key']+'.json'),dict(CPU_fixture_only=True));client.register_verified(req)
  with np.load(mapped(req['frame_file']),allow_pickle=False)as z:images=tuple(z['cam'+str(i)].copy()for i in range(3));command=z['command_state'].copy()
  assert content_key(images,command,req['instruction'])==req['content_key']in client.prefetched
 def body(path,name):return next(n for n in ast.walk(ast.parse(path.read_text()))if isinstance(n,ast.FunctionDef)and n.name==name)
 assert ast.dump(body(C/'runtime.py','run_case'))==ast.dump(body(T/'native_cross_task100_v11/runtime.py','run_case'))
 assert (C/'screen.py').read_text()==(T/'native_cross_task100_v11/screen.py').read_text()
 for name in ['semantic.py','semantic_worker.py','preflight.py','preload.py','cache.py','owner_precision.py']:
  assert ast.dump(ast.parse((C/name).read_text().replace('native_cross_task100_v18','native_cross_task100_v11')))==ast.dump(ast.parse((T/'native_cross_task100_v11'/name).read_text())),name
 assert not torch.cuda.is_initialized()
 write(O/'CPU_preparation_checks.json',dict(passed=True,actual_failed_source_root=str(old),cached_valid_cases=8,completed_expert_rows=cache['completed_expert_rows'],all_eight_RGB_command_language_content_keys_exact=True,all_original_cached_expert_seed_order_preserved=True,missing_instruction_does_not_advance_seed=True,gap_wrong_physics_or_checkpoint_fail_closed=True,no_primary_outcome_or_old_mask_parsing=True,original_entire_run_case_expert_and_inference_AST_exact=True,no_GPU_calls=True,source_sha256={str(p):sha(p)for p in C.glob('*.py')},input_sha256=cache['input_sha256']))
 print('VERIFIED_PREPARATION_PREFIX_CPU_PASS',flush=True)
if __name__=='__main__':main()

"""Reproduce the actual partial-prefix boundary; verify multiple native continuations."""
import ast,copy,threading
from unittest.mock import patch
from .common import C,T,O,read,write,sha
from .preparation import collect_root
from .cached_screen import schedule_class
from native_cross_task100_v14.cached_screen import schedule_class as old_schedule

def main():
 old=T/'outputs/native_cross_task100_v14/batches/008/runs/cf26f62de540444e85174fb73833bfd9'
 spec=read(old/'protocol.json');study=read(T/'outputs/native_cross_task100_v14/study.json')
 cache=collect_root(old,study,8,spec['initial_screen_state'])
 assert cache['cached_valid_cases']==4 and all(len(x['items'])==2 for x in cache['task_cache'].values())
 class Original:
  def __init__(self,c,prepared,spec,root,planner):
   self.state={t:dict(spec['initial_screen_state'][t],rows=[])for t in spec['tasks']};self.locks={t:threading.Lock()for t in spec['tasks']};self.root=root;self.fallback=[]
  def next(self,name):
   with self.locks[name]:
    state=self.state[name];value=dict(task=name,seed=state['next_seed'],episode_index=state['accepted'],CPU_fixture_only=True)
    state['next_seed']+=1;state['accepted']+=1;self.fallback.append(value);return value
 s=dict(spec,cached_preparation=cache);before=copy.deepcopy(s)
 # Original defect appears only on the SECOND original fallback after cached depletion.
 root=O/'CPU_partial_old_fixture'
 with patch('native_cross_task100_v14.cached_screen.write',lambda p,v:None):
  broken=old_schedule(Original)(None,None,s,root,None)
  for t in spec['tasks']:
   for _ in range(3):broken.next(t)
  try:broken.next(spec['tasks'][0])
  except AssertionError:pass
  else:raise AssertionError('Original partial-prefix defect was not reproduced')
 fixed=schedule_class(Original)(None,None,s,O/'CPU_partial_fixed_fixture',None)
 got={t:[]for t in spec['tasks']}
 for i in range(12):
  t=spec['tasks'][i%2];got[t].append(fixed.next(t))
 for t in spec['tasks']:
  start=spec['initial_screen_state'][t]['accepted']
  assert [x['episode_index']for x in got[t]]==list(range(start,start+6))
  assert got[t][:2]==cache['task_cache'][t]['items']
  assert fixed.cached_positions[t]==len(cache['task_cache'][t]['rows'])
  assert fixed.cached_items[t]==2 and t in fixed.cached_drained
  assert len([x for x in fixed.fallback if x['task']==t])==4
 assert s==before,'Protocol or cache metadata mutated'
 names=['runtime.py','screen.py','semantic.py','semantic_worker.py','preflight.py','preload.py','cache.py','owner_precision.py','prefetch.py','startup.py','ordered_screen.py']
 for name in names:
  assert ast.dump(ast.parse((C/name).read_text().replace('native_cross_task100_v16','native_cross_task100_v14')))==ast.dump(ast.parse((T/'native_cross_task100_v14'/name).read_text())),name
 proof=T/'outputs/native_cross_task100_v14/CPU_preparation_recompute_checks.json';exitproof=proof.with_name('CPU_preparation_recompute_actual_exit.json')
 parent=read(proof);actual=read(exitproof)
 assert parent['passed'] and actual['actual_exit_code']==0
 write(O/'CPU_partial_cache_checks.json',dict(passed=True,actual_partial_source_root=str(old),cached_cases=4,old_defect_reproduced=True,two_cached_then_four_native_continuations_per_task=True,each_prefix_consumed_once=True,protocol_immutable=True,all_scientific_inference_AST_unchanged=True,parent_CPU_recompute_proof_reused_not_rerun=True,parent_proofs={str(p):sha(p)for p in[proof,exitproof]},source_sha256={str(p):sha(p)for p in C.glob('*.py')},no_GPU_calls=True))
 print('PARTIAL_CACHE_BOUNDARY_CPU_PASS',flush=True)
if __name__=='__main__':main()

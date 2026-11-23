from pathlib import Path
import json,hashlib,weakref,threading
from types import SimpleNamespace
from .lifecycle import plan,assert_released
T=Path(__file__).resolve().parents[1]
def main():
 spec=json.loads((T/'outputs/native_expert_full_cache_v6/protocol.json').read_text());report=json.loads((T/'outputs/native_expert_full_failure_v6/report.json').read_text())
 indices=report['remaining_original96_indices'];waves=plan(spec['scenes'],indices)
 assert [len(w)for w in waves]==[12,12,14]and sorted(i for w in waves for i in w)==indices
 for task in ('stack_blocks_two','stack_bowls_two'):
  assert [i for w in waves for i in w if spec['scenes'][i]['task']==task]==[i for i in indices if spec['scenes'][i]['task']==task]
 class Item:
  def __init__(self):self.cycle=self
 def ended():
  a,b=Item(),Item();return SimpleNamespace(task_lifetimes=[weakref.ref(a)],scene_lifetimes=[weakref.ref(b)])
 dead=ended();proof=assert_released(dead,[],[]);assert proof['live_tasks_after_gc']==proof['live_scenes_after_gc']==0
 live=Item();c=SimpleNamespace(task_lifetimes=[weakref.ref(live)],scene_lifetimes=[])
 rejected=0
 for args in[(c,[],[]),(dead,[],[SimpleNamespace(tasks={'x':live})]),(dead,[threading.current_thread()],[])]:
  try:assert_released(*args)
  except RuntimeError:rejected+=1
  else:raise AssertionError('Unsafe release accepted')
 for bad in [indices+[indices[0]],list(reversed(indices))]:
  try:plan(spec['scenes'],bad)
  except ValueError:rejected+=1
  else:raise AssertionError('Bad coverage accepted')
 out=T/'outputs/native_scene_waves_v1';out.mkdir(exist_ok=True)
 result=dict(passed=True,CPU_only=True,actual_remaining_original_indices=indices,waves=waves,weakreference_cycle_release_verified=True,unsafe_lifecycle_or_coverage_rejections=rejected,GPU_memory_improvement_NOT_claimed=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')})
 with(out/'CPU_checks.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print('SCENE_FREE_WAVE_CPU_CHECKS_PASS',[len(w)for w in waves],rejected)
if __name__=='__main__':main()

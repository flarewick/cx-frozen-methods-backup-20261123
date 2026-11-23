from pathlib import Path
from types import SimpleNamespace
import weakref,json,hashlib
from .release import verify_dead
T=Path(__file__).resolve().parents[1]
class Item:pass
def closed():
 task=Item();scene=Item();task.scene=scene;scene.task=task;task.self=task
 return weakref.ref(task),weakref.ref(scene)
def main():
 other=Item();other.scene=Item();r=weakref.ref(other);task,scene=closed();verify_dead(task,scene);assert r()is other and other.scene is not None
 a=Item();b=Item();a.scene=b
 try:verify_dead(weakref.ref(a),weakref.ref(b))
 except RuntimeError:pass
 else:raise AssertionError('Retained live objects accepted')
 out=T/'outputs/native_case_release_v1';out.mkdir(exist_ok=True)
 report=dict(passed=True,completed_cycle_released=True,live_case_refused=True,other_lane_objects_preserved=True,CPU_only=True,actual_GPU_memory_effect_NOT_verified=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')})
 with(out/'CPU_checks.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('COMPLETED_CASE_LIFETIME_CPU_CHECKS_PASS')
if __name__=='__main__':main()

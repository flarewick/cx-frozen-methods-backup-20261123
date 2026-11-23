"""CPU source/mode/readback contracts; cannot prove actual rendered RGB parity."""
from pathlib import Path
import types,copy,random,uuid,threading
import numpy as np
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import verify
from .defer import build,configure,install
from .probe import compare_npz

def main():
 import torch
 torch.set_num_threads(1)
 from cx_tta_feedback_confirm5.throughput import discard_observation as original
 old=T/'native_onpolicy_arm_collection_v26';new=T/'native_onpolicy_arm_collection_v28'
 proof=T/'outputs/native_onpolicy_arm_collection_v26/CPU_contract_checks_v2.json';v=read(proof);verify(v);assert v['passed']
 runtime=(new/'runtime.py').read_text().replace(' from native_discard_capture_v1.probe import factories\n StagedLanes,actor_class=factories(StagedLanes,actor_class,root,spec)\n','').replace("    from native_discard_capture_v1.defer import configure\n    configure(task,branch=='M_event_full',item['case_id'])\n",'')
 assert runtime==(old/'runtime.py').read_text().replace(old.name,new.name)
 preload=(new/'preload.py').read_text().replace('    from native_discard_capture_v1.defer import install as install_discard_probe\n    install_discard_probe()\n','')
 assert preload==(old/'preload.py').read_text().replace(old.name,new.name)
 host=(new/'host.py').read_text().replace("deadline=min(200,spec['stage_total_GPU_budget_seconds']-spec['inherited_stage_GPU_seconds'])","deadline=min(7200,spec['stage_total_GPU_budget_seconds']-spec['inherited_stage_GPU_seconds'])")
 assert host==(old/'host.py').read_text().replace(old.name,new.name)
 plan=(new/'collection_plan.py').read_text();start=plan.index("  assert split=='val','One bounded render parity diagnostic only'");end=plan.index("  v['independent_required_next_population_preparation']=True",start);plan=plan[:start]+plan[end:];plan=plan.replace('all_failed_seconds_retained=12291.621','all_failed_seconds_retained=11862.916')
 assert plan==(old/'collection_plan.py').read_text().replace(old.name,new.name)
 candidate=build(original);cpu=torch.get_rng_state().clone();before_py=random.getstate();before_np=np.random.get_state()
 class Camera:
  def __init__(self,task):self.task=task;self.count=0;self.last=None
  def update_picture(self):self.count+=1;self.last=np.array([self.task.pose,self.task.light])
 class Task:
  def __init__(self):
   self.pose=0.;self.light=0.;self.data_type={};self.eval_video_path=self.eval_video_ffmpeg=None;self.cameras=Camera(self);self.updates=0
  def _update_render(self):self.updates+=1;self.light=float(np.random.uniform())
  def get_obs(self):self._update_render();self.cameras.update_picture();return self.cameras.last.copy()
 results=[]
 for fast in [False,True]:
  np.random.seed(17);task=Task();configure(task,fast,'CPU_fake_'+str(fast));observed=[]
  for i in range(32):
   task.pose+=.125;candidate(task)
   if i%4==0:observed.append(task.get_obs())
  results.append((task,np.stack(observed),copy.deepcopy(np.random.get_state())))
 a,b=results;assert np.array_equal(a[1],b[1])and a[0].updates==b[0].updates==40
 assert a[0].cameras.count==40 and b[0].cameras.count==8 and b[0]._cx_deferred_capture_count==32
 assert a[2][0]==b[2][0]and np.array_equal(a[2][1],b[2][1])and a[2][2:]==b[2][2:]
 for key in ['depth','pointcloud','actor_segmentation']:
  task=Task();task.data_type[key]=True;configure(task,True,'CPU_bad_'+key)
  errors=[]
  for fn in [original,candidate]:
   try:fn(task)
   except ValueError as error:errors.append(str(error))
   else:raise AssertionError('Original unsupported modality guard bypassed')
  assert errors[0]==errors[1]and task.updates==0 and task.cameras.count==0
 out=T/'outputs/native_onpolicy_arm_collection_v28';out.mkdir(parents=True,exist_ok=True);fixture=out/'CPU_array_fixture'/uuid.uuid4().hex;fixture.mkdir(parents=True)
 x=np.arange(64,dtype=np.float32).reshape(8,8)
 for name in ['reference.npz','same.npz']:
  with(fixture/name).open('xb')as f:np.savez_compressed(f,x=x)
 assert compare_npz(fixture/'reference.npz',fixture/'same.npz')['all_bytes_exact']
 with(fixture/'changed.npz').open('xb')as f:np.savez_compressed(f,x=x+1)
 try:compare_npz(fixture/'reference.npz',fixture/'changed.npz')
 except AssertionError:pass
 else:raise AssertionError('Changed arrays accepted')
 from cx_tta_feedback_confirm5 import throughput
 aliases=install();assert throughput.discard_observation is not original
 np.random.set_state(before_np);random.setstate(before_py)
 assert torch.equal(cpu,torch.get_rng_state())and not torch.cuda.is_initialized()
 files=[p for folder in [new,Path(__file__).parent]for p in folder.rglob('*')if p.is_file()and p.suffix in('.py','.sh','.md')]
 write(out/'CPU_contract_checks_v2.json',dict(passed=True,engineering_parity_only=True,original_runtime_preload_host_plan_restore_exact=True,
  original_update_lighting_draws_guards_and_actual_reads_preserved=True,CPU_fixture_readback_equal=True,
  CPU_fixture_captures_original=40,CPU_fixture_captures_candidate=8,actual_native_RGB_parity_NOT_proven=True,
  changed_array_refused=True,no_GPU_or_scientific_qualification=True,installed_CPU_aliases=aliases,
  source_sha256={str(p):sha(p)for p in files},input_sha256={str(proof):sha(proof)}))
 print('UNREAD_CAPTURE_CPU_CONTRACTS_PASS_NATIVE_PARITY_STILL_REQUIRED',flush=True)

if __name__=='__main__':main()

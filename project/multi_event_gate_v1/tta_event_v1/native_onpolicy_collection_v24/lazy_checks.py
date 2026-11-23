"""Actual rare-light frames, strict old boundaries and exact reversible scope."""
from pathlib import Path
import weakref,gc,random
import numpy as np,torch
from .common import T,O,read,sha,mapped,verify
from .cached_revalidation import decision
from .runtime_scope import transform
from .setup_profile import install,METHODS

def run():
 d=T/'outputs/native_q0_light_alignment_v1/diagnosis';r=mapped(read(d/'execution_state.json')['root']);spec=read(r/'protocol.json');verify(spec);assert read(d/'actual_host_wait_exit.json')['exit_code']==0 and read(r/'terminal_audit.json')['passed']
 rows=read(r/'GPU_q0_diagnostic_complete.json')['records'];assert len(rows)==6
 with np.load(mapped(spec['expected_frame']),allow_pickle=False)as z:oldcommand=z['command_state'].copy()
 for row in rows:
  f=mapped(row['frame_file']);assert sha(f)==row['frame_sha256']
  with np.load(f,allow_pickle=False)as z:initial=dict(rgb=[z['cam'+str(i)].copy()for i in range(3)],command_state=z['command_state'].copy(),native_action_count=0,physical_sha256=row['physical_sha256'],rng_sha256=row['rng_sha256'])
  actual,requires=decision(initial,spec['expected_diagnostic'],oldcommand,spec['expected_key'],spec['case']['instruction']);assert requires and actual==row['content_key']
  key,requires=decision(initial,spec['expected_diagnostic'],oldcommand,actual,spec['case']['instruction']);assert key==actual and not requires
 for field,value in [('native_action_count',1),('physical_sha256','wrong'),('rng_sha256','wrong'),('command_state',oldcommand+1)]:
  bad=dict(initial);bad[field]=value
  try:decision(bad,spec['expected_diagnostic'],oldcommand,spec['expected_key'],spec['case']['instruction'])
  except AssertionError:pass
  else:raise AssertionError('Bad old boundary accepted')
 c=T/'native_onpolicy_collection_v24';old=T/'native_onpolicy_collection_v17'
 assert(c/'runtime.py').read_text()==transform((old/'runtime.py').read_text())
 assert(c/'episode.py').read_text()==(old/'episode.py').read_text()
 expected=(old/'cached_screen.py').read_text().replace("self.root/'initial_prefetch'/(item['case_id']+'.json')","self.root/'cached_initial_prefetch'/(item['case_id']+'.json')")
 assert(c/'cached_screen.py').read_text()==expected
 class Task:pass
 def operation(self,x=1):return np.r_[np.random.rand(3),torch.rand(3).numpy(),random.random(),x]
 for name in METHODS:setattr(Task,name,operation)
 def run(profiled):
  np.random.seed(17);torch.manual_seed(17);random.seed(17);task=Task()
  if profiled:install(task,O/'CPU_setup_profile_fixture.jsonl')
  results=[getattr(task,n)(x=i)for i,n in enumerate(METHODS)];rng=(np.random.rand(),float(torch.rand(())),random.random());ref=weakref.ref(task);task=None;gc.collect();assert ref()is None
  return results,rng
 a,ar=run(False);b,br=run(True);assert ar==br and all(np.array_equal(x,y)for x,y in zip(a,b))
 return dict(passed=True,actual_rare_light_frames_checked=6,new_current_key_requires_whole_fresh_full_control=True,matching_old_inputs_keep_exact_key=True,invalid_old_boundaries_rejected=4,production_runtime_exact_reversible_C17_edits=True,original_full_episode_unchanged=True,profile_calls_and_Torch_numpy_python_RNG_exact=True,profile_does_not_retain_task=True,no_extra_policy_query_or_native_step=True,no_GPU_work=True,input_sha256={str(f):sha(f)for f in[d/'actual_host_wait_exit.json',r/'terminal_audit.json',r/'protocol.json',r/'GPU_q0_diagnostic_complete.json',*[mapped(row['frame_file'])for row in rows]]})

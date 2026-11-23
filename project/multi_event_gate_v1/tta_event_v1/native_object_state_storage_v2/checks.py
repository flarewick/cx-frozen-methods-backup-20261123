"""Real CPU SAM2 one/two-object regression; GPU byte parity still required."""
import ast,inspect
import numpy as np
from .common import C,T,O,read,write,sha,mapped,verify
def main():
 import torch
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
 for p in C.rglob('*.py'):ast.parse(p.read_text())
 host_scope=O/'CPU_host_admission_scope_v1.json';scope=read(host_scope);verify(scope)
 assert scope['passed']and scope['actual_choose_code_object_identical']and scope['all_other_globals_identity_retained']
 manifest=O/'input_manifest.json';v=read(manifest);verify(v)
 assert len(v['cases'])==6 and {x['actual_object_count']for x in v['cases']}=={1,2}
 from native_object_feedback_v3.tracking import MaskCore,video
 from .core import CPUStateCore,CPUStateStream
 from .runtime import compare
 with torch.random.fork_rng(devices=[]):model=video.build('cpu')
 import sam2.sam2_video_predictor as official
 official.tqdm=lambda x,**kw:x
 results=[];cpu_rng=torch.get_rng_state().clone()
 for item in [v['cases'][0],v['cases'][-1]]:
  with np.load(mapped(item['file']),allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
  outputs=[]
  for cls in [MaskCore,CPUStateCore]:
   core=cls(model)
   with torch.inference_mode():
    first=core.initialize(a['initial'],a['masks'],tick=0,decision_tick=0)
    tick=int(a['ticks'][0]);second=core.append(a['frames'][0],tick=tick,decision_tick=tick)
   assert core.count==item['actual_object_count'];assert core.stream.state['offload_state_to_cpu']is(cls is CPUStateCore)
   outputs.append([first,second])
  for x,y in zip(*outputs,strict=True):compare(x,y)
  changed=dict(outputs[1][1],presence=outputs[1][1]['presence'].copy());changed['presence'][0]+=1e-6
  try:compare(outputs[0][1],changed)
  except AssertionError:pass
  else:raise AssertionError('Changed SAM2 confidence accepted')
  results.append(dict(actual_object_count=item['actual_object_count'],actual_CPU_frames=2,all_bytes_exact=True))
 assert torch.equal(cpu_rng,torch.get_rng_state())and not torch.cuda.is_initialized()
 sources=[p for p in C.rglob('*')if p.is_file()and p.suffix in('.py','.sh','.md')]
 inputs=[host_scope,manifest,T/'native_object_feedback_v3/tracking.py',T/'native_object_feedback_v3/source.py',T/'native_onpolicy_arm_collection_v29/host.py',T/'outputs/native_onpolicy_full_ops_v2/ARM29_actual_first280_failure_v1.json']
 write(O/'CPU_contract_checks_v2.json',dict(passed=True,actual_SAM2_CPU_one_two_object_outputs=results,
  original_GPU_admission_choose_bytecode_and_guard_functions_retained=True,changed_confidence_refused=True,
  original_science_math_unchanged=True,CUDA_initialized=False,GPU_parity_NOT_proven=True,
  source_sha256={str(p):sha(p)for p in sources},input_sha256={str(p):sha(p)for p in inputs}))
 print('SAM2_STORAGE_CPU_ONE_TWO_OBJECT_BYTES_EXACT_GPU_PARITY_STILL_REQUIRED',flush=True)
if __name__=='__main__':main()

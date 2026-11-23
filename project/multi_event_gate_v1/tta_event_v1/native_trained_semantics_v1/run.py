"""Same frozen CPU semantic inference on newly captured actual developer q0."""
from pathlib import Path
import json,hashlib,time,random,pickle,resource,sys
import numpy as np,torch
from native_policy_execution_v1.common import T,read,write,sha
from native_trained_initial_observer_v1.audit import audit
from native_expert_full_seed_v6.batch import infer_or_missing,rng_digest
from seed import semantic
from probe import source_check
from adapter_cache_cpu import state_digest
C=Path(__file__).resolve().parent;O=T/'outputs/native_trained_semantics_v1'
def image_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
 assert not O.exists(),'Preserve prior semantic attempt'
 terminal=audit();assert terminal['passed'];root=Path(terminal['root']);frames=read(root/'GPU_frames_complete.json')['rows']
 assert len(frames)==4
 teacherfile=T/'outputs/native_arrived_mask_v3/protocol.json';teacher=read(teacherfile)
 for rel,h in teacher['source_sha256'].items():assert sha(T.parents[1]/rel)==h
 contract=source_check();inputs=dict(terminal['input_sha256']);inputs[str(teacherfile)]=sha(teacherfile)
 modelsfile=T/'outputs/native_arrived_mask_v3/cpu_models.json';inputs[str(modelsfile)]=sha(modelsfile)
 sources={str(p):sha(p)for p in C.glob('*.py')}
 for p in(T/'native_cpu_seed_v1/seed.py',T/'native_cpu_seed_v1/probe.py',T/'native_expert_full_seed_v6/batch.py',T/'native_trained_initial_observer_v1/audit.py'):sources[str(p)]=sha(p)
 protocol=dict(stage='developer_eval_mode_actual_q0_CPU_semantics',requested=4,jobs=frames,input_sha256=inputs,source_sha256=sources,source_contract=contract,CPU_only=True,new_independent_scenes=0,thresholds_and_decision_math_unchanged=True)
 write(O/'protocol.json',protocol)
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');cpu=torch.random.get_rng_state().clone();py=random.getstate();npstate=np.random.get_state();rng=rng_digest()
 try:
  with torch.random.fork_rng(devices=[]):detector=semantic.FrozenInstances('cpu')
 finally:random.setstate(py);np.random.set_state(npstate)
 assert torch.equal(cpu,torch.random.get_rng_state())and rng_digest()==rng and not torch.cuda.is_initialized()
 models={'DINO':detector.detector,'SAM':detector.sam};expected=read(modelsfile)['model_state_sha256'];digests={}
 for name,model in models.items():
  assert not model.training and not any(p.requires_grad for p in model.parameters())
  digests[name]=state_digest(model);assert digests[name]==expected[name]
 torch.set_num_threads(2);rows=[]
 for i,frame in enumerate(frames):
  assert sha(frame['file'])==frame['sha256']
  with np.load(frame['file'],allow_pickle=False)as z:
   images=tuple(z['cam'+str(j)].copy()for j in range(3));command=z['command_state'].copy()
  before=rng_digest();cpu=torch.random.get_rng_state().clone();began=time.monotonic()
  value,metadata,inferred=infer_or_missing(detector,images[0],frame['instruction']);seconds=time.monotonic()-began
  assert before==rng_digest()and torch.equal(cpu,torch.random.get_rng_state())and not torch.cuda.is_initialized()
  row=dict(case_id=frame['case_id'],frame_file=frame['file'],frame_sha256=frame['sha256'],instruction_sha256=hashlib.sha256(frame['instruction'].encode()).hexdigest(),RGB_sha256=[image_sha(im)for im in images],command_state_sha256=image_sha(command),available=bool(value['available']),reason=value['reason'],actual_CPU_inference=inferred,cold_CPU_inference_seconds=seconds,models_state_sha256=digests,metadata=metadata,query=0,tick=0,initialization_eval_mode=True)
  if value['available']:
   file=O/('case%03d_masks.npz'%i)
   with file.open('xb')as f:np.savez_compressed(f,masks=value['masks'])
   row.update(mask_file=str(file),mask_sha256=sha(file))
  write(O/('case%03d.json'%i),row);rows.append(row)
  print('ACTUAL_DEVELOPER_EVAL_Q0_CPU_SEMANTIC',frame['case_id'],row['available'],seconds,flush=True)
 for name,model in models.items():assert state_digest(model)==digests[name]
 for field in('source_sha256','input_sha256'):
  for p,h in protocol[field].items():assert sha(p)==h
 write(O/'result.json',dict(complete=True,requested=4,rows=rows,available=sum(r['available']for r in rows),models_state_sha256=digests,CPU_models_unchanged=True,CUDA_initialized=False,all_RNG_preserved=True,protocol_sha256=sha(O/'protocol.json'),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_independent_scenes=0,actual_actor_rollouts=0,TTA_utility_proven=False))
 print('DEVELOPER_EVAL_Q0_CPU_SEMANTICS_COMPLETE',sum(r['available']for r in rows),flush=True)
if __name__=='__main__':main()

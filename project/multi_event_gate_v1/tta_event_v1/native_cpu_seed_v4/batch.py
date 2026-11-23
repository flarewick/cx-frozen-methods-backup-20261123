"""Generic frozen q0 CPU model preparation; no task/outcome/late-frame forward."""
from pathlib import Path
import argparse,random,pickle,time,resource,sys
import numpy as np,torch
from native_cpu_seed_v4.common import C,T,P,read,sha,write,initial,rgb_sha,language_sha
sys.path.insert(0,str(T/'native_cpu_seed_v1'))
from seed import semantic,infer_pair_cpu
from probe import source_check
from adapter_cache_cpu import state_digest
def rng_digest():return sha_bytes(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5))
def sha_bytes(value):
 import hashlib
 return hashlib.sha256(value).hexdigest()
def validate_manifest(file):
 file=Path(file).resolve();manifest=read(file)
 assert manifest['version']==1 and manifest['kind']=='actual_q0_semantic_preparation'
 assert manifest['requested']==len(manifest['jobs'])and manifest['requested']>0
 assert manifest['scope']in('CPU_developer_contract_bridge','independent_TRAIN_confirmation')
 inputs={str(file):sha(file)};known=set();jobs=[];missing=[];developer_keys=None
 if manifest['scope']=='CPU_developer_contract_bridge':
  p=Path(manifest['old_developer_protocol']);assert sha(p)==manifest['old_developer_protocol_sha256']
  original=read(p);inputs[str(p)]=sha(p)
  developer_keys={(r['head_sha256'],r['language_sha256'])for r in original['jobs']}
 if manifest['scope']=='independent_TRAIN_confirmation':
  f=Path(manifest['candidate_freeze']);frozen=read(f)
  assert sha(f)==manifest['candidate_freeze_sha256']and frozen['frozen']
  assert frozen['full_feedback_candidate_frozen']and frozen['new_confirmation_not_used_to_choose_candidate']
  assert frozen['identity']==manifest['candidate_identity']
  for p,h in frozen['source_sha256'].items():assert sha(p)==h
  inputs[str(f)]=sha(f)
 for row in manifest['jobs']:
  assert row['case_id']not in known;known.add(row['case_id'])
  if not row['available']:
   assert row['reason'];missing.append(dict(case_id=row['case_id'],reason=row['reason'],index=row['index']));continue
  assert type(row['query'])is int and type(row['tick'])is int and row['query']==row['tick']==0
  p=Path(row['file']).resolve();assert sha(p)==row['sha256'];inputs[str(p)]=sha(p)
  with np.load(p,allow_pickle=False)as z:images=initial(z)
  assert row['head_RGB_sha256']==rgb_sha(images[0])and row['instruction_sha256']==language_sha(row['instruction'])
  assert isinstance(row['instruction'],str)and row['instruction']
  if developer_keys is not None:assert (row['head_RGB_sha256'],row['instruction_sha256'])in developer_keys,'New input cannot be labelled an old developer CPU bridge'
  jobs.append(dict(row,file=str(p.resolve())))
 return manifest,jobs,missing,inputs
def run(manifest_file,output):
 manifest_file=Path(manifest_file).resolve();output=Path(output).resolve();assert not output.exists(),'Existing preparation preserved; inspect rather than retry'
 manifest,jobs,missing,inputs=validate_manifest(Path(manifest_file))
 contract=source_check();teacherfile=T/'outputs/native_arrived_mask_v3/protocol.json';teacher=read(teacherfile)
 for rel,h in teacher['source_sha256'].items():assert sha(P/rel)==h
 inputs[str(teacherfile)]=sha(teacherfile)
 modelmeta=T/'outputs/native_arrived_mask_v3/cpu_models.json';inputs[str(modelmeta)]=sha(modelmeta)
 sources={str(p):sha(p)for p in sorted(C.glob('*.py'))}
 for p in(T/'native_cpu_seed_v1/seed.py',T/'native_cpu_seed_v1/probe.py'):sources[str(p)]=sha(p)
 # Real forward for every available job; no case-dependent cache/threshold choice.
 protocol=dict(frozen=True,requested=manifest['requested'],jobs=jobs,missing=missing,scope=manifest['scope'],
  CPU_threads=2,device='cpu',dtype='float32',head_batch=1,all_available_jobs_real_CPU_forward=True,
  source_contract=contract,source_sha256=sources,input_sha256=inputs,manifest_sha256=sha(manifest_file),
  initialization_only=True,online_updates=0,TTA_utility_proven=False)
 output.mkdir(parents=True);write(output/'protocol.json',protocol)
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');cpu=torch.random.get_rng_state().clone();before_rng=rng_digest()
 py_state=random.getstate();np_state=np.random.get_state()
 try:
  with torch.random.fork_rng(devices=[]):detector=semantic.FrozenInstances('cpu')
  startup_rng_changed=dict(Python=random.getstate()!=py_state,
    NumPy=sha_bytes(pickle.dumps(np.random.get_state(),protocol=5))!=sha_bytes(pickle.dumps(np_state,protocol=5)),
    Torch_CPU=not torch.equal(cpu,torch.random.get_rng_state()))
 finally:
  random.setstate(py_state);np.random.set_state(np_state)
 assert torch.equal(cpu,torch.random.get_rng_state())and rng_digest()==before_rng and not torch.cuda.is_initialized()
 write(output/'startup_RNG_scope.json',dict(raw_model_initialization_changed=startup_rng_changed,
  all_three_RNG_states_restored=True,CUDA_initialized=False,inference_RNG_NOT_hidden_or_reset=True))
 print('CPU_MODEL_STARTUP_RNG_RESTORED',startup_rng_changed,flush=True)
 models={'DINO':detector.detector,'SAM':detector.sam};expected=read(modelmeta)['model_state_sha256'];digests={}
 for name,model in models.items():
  assert not model.training and not any(p.requires_grad for p in model.parameters())
  digests[name]=state_digest(model);assert digests[name]==expected[name]
 torch.set_num_threads(2);rows=[]
 for job in jobs:
  assert sha(job['file'])==job['sha256']
  with np.load(job['file'],allow_pickle=False)as z:images=initial(z)
  assert rgb_sha(images[0])==job['head_RGB_sha256']
  rng=rng_digest();cpu=torch.random.get_rng_state().clone();started=time.monotonic()
  value,metadata=infer_pair_cpu(detector,images[0],job['instruction'])
  seconds=time.monotonic()-started
  assert rng_digest()==rng and torch.equal(cpu,torch.random.get_rng_state())and not torch.cuda.is_initialized()
  row=dict(index=job['index'],case_id=job['case_id'],available=bool(value['available']),reason=value['reason'],
    head_RGB_sha256=job['head_RGB_sha256'],instruction_sha256=job['instruction_sha256'],model_state_sha256=digests,
    actual_CPU_inference=True,reused_exact_prior_CPU_result=False,seconds=seconds,cold_CPU_inference_seconds=seconds,
    metadata=metadata,query=0,tick=0,unknown_never_imputed=True)
  if value['available']:
   file=output/'cases'/('%03d.npz'%job['index']);file.parent.mkdir(exist_ok=True)
   with file.open('xb')as f:np.savez_compressed(f,masks=value['masks'])
   row.update(file=str(file),sha256=sha(file))
  write(output/'cases'/('%03d.json'%job['index']),row);rows.append(row)
  write(output/'progress.json',dict(completed=len(rows),requested=manifest['requested'],available=sum(r['available']for r in rows),missing=len(missing),CPU_only=True,current=job['case_id']),mutable=True)
  print('GENERIC_ACTUAL_CPU_Q0_SEED',job['case_id'],row['available'],seconds,flush=True)
 assert all(state_digest(model)==digests[name]for name,model in models.items())
 for group in('source_sha256','input_sha256'):
  for p,h in protocol[group].items():assert sha(p)==h
 write(output/'result.json',dict(complete=True,requested=manifest['requested'],rows=rows,missing=missing,
  available=sum(r['available']for r in rows),actual_new_CPU_inferences=len(rows),reused_CPU_results=0,
  CPU_models_unchanged=True,models_state_sha256=digests,CUDA_initialized=False,scope=manifest['scope'],
  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,protocol_sha256=sha(output/'protocol.json'),
  new_independent_scenes=manifest['requested']if manifest['scope']=='independent_TRAIN_confirmation'else 0,
  online_updates=0,feedback_support_or_TTA_utility_proven=False,TTA_utility_proven=False))
 print('GENERIC_Q0_CPU_PREPARATION_COMPLETE',len(rows),len(missing),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.manifest,a.output)

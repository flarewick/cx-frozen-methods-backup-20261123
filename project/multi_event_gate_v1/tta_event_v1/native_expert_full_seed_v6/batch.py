"""Generic frozen q0 CPU model preparation; no task/outcome/late-frame forward."""
from pathlib import Path
import argparse,random,pickle,time,resource,sys
import numpy as np,torch
from native_expert_full_seed_v6.common import C,T,P,read,sha,write,initial,rgb_sha,language_sha
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
 assert manifest['version']==1 and manifest['kind']=='actual_q0_semantic_preparation'and manifest['scope']=='original_expert_FIT_VAL_cache_part'
 assert manifest['requested']==len(manifest['jobs'])==24 and manifest['worker']in range(4)
 cohortfile=Path(manifest['cohort_manifest']);assert sha(cohortfile)==manifest['cohort_manifest_sha256'];cohort=read(cohortfile)
 assert cohort['requested']==96 and cohort['new_independent_scenes']==0
 assert manifest['jobs']==[r for r in cohort['jobs']if r['index']%4==manifest['worker']]
 specfile=Path(cohort['native_protocol']);assert sha(specfile)==cohort['native_protocol_sha256'];spec=read(specfile)
 completefile=Path(cohort['native_GPU_complete']);assert sha(completefile)==cohort['native_GPU_complete_sha256'];complete=read(completefile)
 assert complete['complete']and complete['protocol_identity']==spec['identity']and len(complete['rows'])==96
 terminalfile=Path(cohort['native_terminal_audit']);assert sha(terminalfile)==cohort['native_terminal_audit_sha256']and read(terminalfile)['passed']
 inputs={str(p):sha(p)for p in(file,cohortfile,specfile,completefile,terminalfile)};jobs=[];missing=[]
 for row in manifest['jobs']:
  source=spec['scenes'][row['index']];actual=complete['rows'][row['index']]
  assert source['case_id']==row['case_id']==actual['case_id']and source['split']==row['split']and source['index']==row['original_index']
  if not row['available']:
   assert not actual['available']and row['reason']==actual['reason'];missing.append(dict(case_id=row['case_id'],reason=row['reason'],index=row['index']));continue
  assert actual['available']and actual['matched_initial_object_robot_command_exact']
  obs=actual['observations'][0];p=Path(row['file']).resolve()
  assert row['query']==row['tick']==obs['query']==obs['global_tick']==0 and str(p)==obs['file']and row['sha256']==obs['sha256']==sha(p)
  assert row['instruction']==source['instruction']and row['instruction_sha256']==language_sha(row['instruction'])
  with np.load(p,allow_pickle=False)as z:images=initial(z)
  assert row['head_RGB_sha256']==rgb_sha(images[0]);inputs[str(p)]=sha(p);jobs.append(row)
 assert len(jobs)+len(missing)==24
 return manifest,jobs,missing,inputs

def infer_or_missing(detector,image,instruction):
 roles,phrases=semantic.prompt(instruction)
 if phrases:
  value,metadata=infer_pair_cpu(detector,image,instruction)
  return value,metadata,True
 return dict(available=False,reason='unsupported_language_retained'),dict(
  roles=roles,phrases=phrases,device='cpu',actual_model_forward=False,
  frozen_parser_unchanged=True,language_rewritten=False,unknown_never_imputed=True),False

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
  CPU_threads=2,device='cpu',dtype='float32',head_batch=1,all_supported_available_jobs_real_CPU_forward=True,unsupported_language_explicit_no_inference=True,
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
  value,metadata,inferred=infer_or_missing(detector,images[0],job['instruction'])
  seconds=time.monotonic()-started
  assert rng_digest()==rng and torch.equal(cpu,torch.random.get_rng_state())and not torch.cuda.is_initialized()
  row=dict(index=job['index'],case_id=job['case_id'],available=bool(value['available']),reason=value['reason'],
    head_RGB_sha256=job['head_RGB_sha256'],instruction_sha256=job['instruction_sha256'],model_state_sha256=digests,
    actual_CPU_inference=inferred,reused_exact_prior_CPU_result=False,seconds=seconds,cold_CPU_inference_seconds=seconds if inferred else None,
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
  available=sum(r['available']for r in rows),actual_new_CPU_inferences=sum(r['actual_CPU_inference']for r in rows),unsupported_language_count=sum(r['reason']=='unsupported_language_retained'for r in rows),reused_CPU_results=0,
  CPU_models_unchanged=True,models_state_sha256=digests,CUDA_initialized=False,scope=manifest['scope'],
  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,protocol_sha256=sha(output/'protocol.json'),
  new_independent_scenes=manifest['requested']if manifest['scope']=='independent_TRAIN_confirmation'else 0,
  online_updates=0,feedback_support_or_TTA_utility_proven=False,TTA_utility_proven=False))
 print('GENERIC_Q0_CPU_PREPARATION_COMPLETE',len(rows),len(missing),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.manifest,a.output)

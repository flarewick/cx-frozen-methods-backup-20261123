"""Language coverage and real CPU detector compatibility; no fitting or GT input."""
from pathlib import Path
import ast,json,hashlib,random,time,itertools
import numpy as np,torch,h5py
from . import semantic
from .inference import infer_or_missing,rng_digest
from native_best100_resume_v7.common import T,P,read,mapped,sha
from adapter_cache_cpu import state_digest
C=Path(__file__).resolve().parent;O=T/'outputs/native_transfer_semantics_v1'
def language_checks():
 runtime=mapped(read(T/'outputs/native_best100_resume_v7/study.json')['runtime_source']);results={};count=0
 for task in('place_empty_cup','place_shoe'):
  templates=read(runtime/'description/task_instruction'/(task+'.json'))['unseen'];strings=[]
  if task=='place_empty_cup':
   targets=read(runtime/'description/objects_description/021_cup/base0.json')['unseen'];refs=read(runtime/'description/objects_description/019_coaster/base0.json')['unseen']
   strings=[p.replace('{A}','the '+a).replace('{B}','the '+b)for p,a,b in itertools.product(templates,targets,refs)]
  else:
   for number in range(10):
    for p,a,arm in itertools.product(templates,read(runtime/'description/objects_description/041_shoe'/('base%d.json'%number))['unseen'],('left','right')):strings.append(p.replace('{A}','the '+a).replace('{a}','the '+arm+' arm'))
  old=[semantic.original.prompt(s)for s in strings];new=[semantic.prompt(s)for s in strings];assert all(p for _,p in new)
  results[task]=dict(official_language_combinations=len(strings),original_supported=sum(bool(p)for _,p in old),new_supported=len(new));count+=len(new)
 for text in('Place the cup on the table.','Look at the cup on the coaster.','Place the cup near the coaster.','Put the cup and shoe on the coaster.','place_shoe','place_empty_cup','Move the mat before the cup.'):
  assert not semantic.prompt(text)[1],text
 parent=ast.parse((T/'language_instances_v2/semantic.py').read_text());new=ast.parse((C/'semantic.py').read_text())
 def infer(tree):return next(n for cls in tree.body if isinstance(cls,ast.ClassDef)and cls.name=='FrozenInstances'for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='infer')
 assert ast.dump(infer(parent))==ast.dump(infer(new));assert semantic.FrozenInstances.__init__ is semantic.original.FrozenInstances.__init__
 return results

def main():
 coverage=language_checks();torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');py=random.getstate();ns=np.random.get_state();rng=rng_digest();before=torch.random.get_rng_state().clone()
 try:
  with torch.random.fork_rng(devices=[]):detector=semantic.FrozenInstances('cpu')
 finally:random.setstate(py);np.random.set_state(ns)
 assert rng_digest()==rng and torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 models=dict(DINO=detector.detector,SAM=detector.sam);digests={k:state_digest(v)for k,v in models.items()};assert digests=={k:read(T/'outputs/native_arrived_mask_v3/cpu_models.json')['model_state_sha256'][k]for k in models};torch.set_num_threads(2)
 oldroot=T/'outputs/native_best100_resume_v5/batches/004/runs/9a45728b2db14d328eea2545357acb76/semantic';old_records=[];inputs={}
 for f in sorted((oldroot/'requests').glob('*.json')):
  request=read(f);previous=read(oldroot/'answers'/f.name)
  with np.load(mapped(request['frame_file']),allow_pickle=False)as z:rgb=z['cam0'].copy()
  value,metadata,inferred=infer_or_missing(detector,rgb,request['instruction']);assert metadata==previous['metadata']and value['available']==previous['available']and value['reason']==previous['reason']and inferred==previous['actual_CPU_inference']
  if value['available']:
   with np.load(mapped(previous['mask_file']),allow_pickle=False)as z:old=z['masks'].copy()
   assert old.dtype==value['masks'].dtype and old.shape==value['masks'].shape and old.tobytes()==value['masks'].tobytes();inputs[str(mapped(previous['mask_file']))]=sha(mapped(previous['mask_file']))
  for p in(f,oldroot/'answers'/f.name,mapped(request['frame_file'])):inputs[str(p)]=sha(p)
  old_records.append(dict(key=f.stem,numeric_metadata_and_mask_arrays_byte_exact=True));print('OLD_NATIVE_SEMANTIC_EXACT',f.stem,flush=True)
 assert len(old_records)==4
 manifestfile=P/'outputs/base_cache_20260926_024808_872087/manifest.json';manifest=read(manifestfile);inputs[str(manifestfile)]=sha(manifestfile);new_records=[]
 from cx_tta.dataset import read_observation
 for task in('place_empty_cup','place_shoe'):
  row=next(r for r in manifest['files']if r['task']==task and mapped(r['path']).exists()and mapped(r['source']).exists());file=mapped(row['source'])
  with h5py.File(file,'r')as f:images,command=read_observation(f,0)
  assert images[0].shape==(240,320,3);before=rng_digest();cpu=torch.random.get_rng_state().clone();start=time.monotonic();value,metadata,inferred=infer_or_missing(detector,images[0],row['instruction']);assert inferred and before==rng_digest()and torch.equal(cpu,torch.random.get_rng_state())
  if value['available']:assert value['masks'].shape==(2,240,320)and value['masks'].dtype==bool
  inputs[str(file)]=sha(file);new_records.append(dict(task=task,source=str(file),source_frame_index=0,instruction=row['instruction'],available=value['available'],reason=value['reason'],seconds=time.monotonic()-start,metadata=metadata));print('NEW_HISTORICAL_CPU_INTERFACE',task,value['available'],value['reason'],flush=True)
 assert all(state_digest(v)==digests[k]for k,v in models.items())and not torch.cuda.is_initialized()
 result=dict(passed=True,official_language_coverage=coverage,original_detector_infer_AST_exact=True,original_detector_constructor_inherited_exact=True,old_four_native_numeric_metadata_and_masks_exact=True,old_records=old_records,new_historical_RGB_only_checks=new_records,models_state_sha256=digests,models_unchanged=True,RNG_preserved=True,CUDA_initialized=False,training_steps=0,new_formal_scenes=0,no_threshold_selection_from_outcomes=True,source_sha256={str(p):sha(p)for p in C.glob('*.py')},input_sha256=inputs)
 O.mkdir(exist_ok=True)
 with(O/'CPU_interface_checks_v2.json').open('x')as f:json.dump(result,f,indent=2)
 print('TRANSFER_SEMANTIC_INTERFACE_CPU_PASS',coverage,flush=True)
if __name__=='__main__':main()

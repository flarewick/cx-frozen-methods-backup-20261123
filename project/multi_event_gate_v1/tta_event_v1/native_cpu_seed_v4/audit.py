"""Compare real generic-route CPU results with old CPU outputs, no fit/reforward."""
from pathlib import Path
import copy,numpy as np
from native_cpu_seed_v4.common import C,T,read,sha,write,initial
from native_cpu_seed_v4.store import PreparedCPUSeedStore
from native_cpu_seed_v4.batch import validate_manifest
def run(root):
 root=Path(root).resolve();store=PreparedCPUSeedStore(root);proto=store.protocol;result=store.result
 assert result['requested']==4 and len(result['rows'])==3 and len(result['missing'])==1
 assert result['missing'][0]['case_id']=='CPU_missing_initial_fixture'
 old=read(T/'outputs/native_cpu_seed_v2/result.json');checks=0;rows=[]
 for job in proto['jobs']:
  row=next(r for r in result['rows']if r['index']==job['index'])
  expected=next(r for r in old['rows']if r['case_id']==job['case_id'])
  assert row['available']==expected['available']and row['reason']==expected['reason'];checks+=2
  with np.load(job['file'],allow_pickle=False)as z:images=initial(z)
  value,masks=store.at_initial_observation(images,job['instruction'],query=0,tick=0)
  assert value['available']==row['available']and value['cold_CPU_inference_seconds']==row['seconds'];checks+=2
  if row['available']:
   with np.load(expected['file'],allow_pickle=False)as z:assert np.array_equal(masks,z['masks']);checks+=1
  else:assert masks is None and not value['initialization_preparation_required'];checks+=1
  changed=[im.copy()for im in images];changed[0][0,0,0]^=1
  miss,mm=store.at_initial_observation(changed,job['instruction'],query=0,tick=0)
  assert not miss['available']and mm is None and miss['initialization_preparation_required'];checks+=1
  try:store.at_initial_observation(images,job['instruction'],query=1,tick=1)
  except ValueError:checks+=1
  else:raise AssertionError('Later query used to initialize seed')
  rows.append(dict(case_id=row['case_id'],available=row['available'],reason=row['reason'],seconds=row['seconds'],original_CPU_output_exact=True))
 # All input/source paths must stay valid from an unrelated deployment cwd.
 import os
 before_cwd=Path.cwd();unrelated=root.parent/'unrelated_runtime_cwd';unrelated.mkdir()
 try:
  os.chdir(unrelated);portable=PreparedCPUSeedStore(root)
  assert len(portable.rows)==3;checks+=1
 finally:os.chdir(before_cwd)
 # Independence admission is checked before model loading/output creation.
 manifest=read(root.parent/'manifest.json');manifest['scope']='independent_TRAIN_confirmation'
 file=root.parent/'missing_freeze_rejection_fixture.json';write(file,manifest)
 try:validate_manifest(file)
 except(KeyError,AssertionError,FileNotFoundError):checks+=1
 else:raise AssertionError('Independent data admitted without complete frozen candidate')
 write(root.parent/'audit.json',dict(passed=True,checks=checks,actual_old_CPU_forward_inputs=3,missing_fixture_retained=1,
  rows=rows,known_and_unknown_CPU_outcomes_exact=True,models_and_RNG_unchanged=result['CPU_models_unchanged'],
  GPU_used=False,arbitrary_requested_count_supported=True,full_candidate_freeze_required_for_independent=True,
  source_sha256={str(p):sha(p)for p in sorted(C.glob('*.py'))},input_sha256={str(root/n):sha(root/n)for n in('protocol.json','result.json')},
  new_independent_scenes=0,model_fit_steps=0,online_updates=0,TTA_utility_proven=False))
 print('GENERIC_Q0_CPU_MODEL_AND_CONTENT_STORE_AUDIT_PASS',checks,flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

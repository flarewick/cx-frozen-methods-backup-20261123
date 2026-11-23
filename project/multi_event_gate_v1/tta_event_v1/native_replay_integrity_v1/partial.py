"""Exact shared-prefix integrity, no success/outcome fields or model execution."""
from pathlib import Path
import json,hashlib
import numpy as np
T=Path(__file__).resolve().parents[1];P=T.parents[1];CANON=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')
A=T/'outputs/native_frozen100_v6/placement/batches/008/runs/63a39dc9e8ad4d44b70d8529fba8fe75'
B=T/'outputs/native_frozen100_v12/placement/batches/008/runs/0fd6c092d0f8463581846e6b720ff46e'
def read(p):return json.loads(p.read_text())
def mapped(p):
 p=Path(p)
 if p.exists():return p
 if p.is_relative_to(CANON):return P/p.relative_to(CANON)
 host=Path('/data/qiufw/cx_migration/cx_new_direction_guide_TTA')
 if p.is_relative_to(host):return P/p.relative_to(host)
 return p
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 # Resolve the actual archived V12 attempt by the known prefix, never its success.
 candidates=list((B.parent).glob('0fd6c092*'));assert len(candidates)==1;newroot=candidates[0]
 inputs={};pairs=0;cases=0;fields=0
 for oldcase in sorted((A/'cases').glob('*')):
  common=[]
  for p in sorted((oldcase/'baseline').glob('policy*.json')):
   q=newroot/'cases'/oldcase.name/'baseline'/p.name
   if not q.exists():continue
   old,fresh=read(p),read(q)
   for field in('query','actor_observation_tick','actor_images','command_state_sha256','noise_witness','encoder','base_infer_calls','actual_internal_noise_calls'):
    assert old[field]==fresh[field],('Saved metadata differs',oldcase.name,p.name,field)
   for field in('policy_file','observation_file'):
    a,b=mapped(old[field]),mapped(fresh[field]);assert sha(a)==old[field.replace('_file','_sha256')]and sha(b)==fresh[field.replace('_file','_sha256')]
    with np.load(a,allow_pickle=False)as z,np.load(b,allow_pickle=False)as w:
     assert set(z.files)==set(w.files)
     for key in z.files:
      assert z[key].dtype==w[key].dtype and np.array_equal(z[key],w[key]),('Saved numeric prefix differs',oldcase.name,p.name,field,key)
      fields+=1
    inputs[str(a)]=sha(a);inputs[str(b)]=sha(b)
   inputs[str(p)]=sha(p);inputs[str(q)]=sha(q);pairs+=1;common.append(old['query'])
  assert common and common==list(range(len(common))),'Every archived case must have a contiguous shared q0 prefix'
  cases+=1
 assert cases==10 and pairs>=10
 out=T/'outputs/native_replay_integrity_v1';out.mkdir(parents=True,exist_ok=True)
 report=dict(passed=True,old_root=str(A),new_root=str(newroot),compared_cases=cases,compared_queries=pairs,compared_array_fields=fields,all_shared_baseline_numeric_prefix_values_exact=True,all_shared_RGB_commands_RNG_noise_tokens_exact=True,no_GPU_or_model_execution=True,no_outcomes_or_labels_read=True,no_failed_job_reclassified=True,input_sha256=inputs)
 with(out/'CPU_partial_case008.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('SAVED_BASELINE_PARTIAL_PREFIX_EXACT',cases,pairs,fields,flush=True)
if __name__=='__main__':main()

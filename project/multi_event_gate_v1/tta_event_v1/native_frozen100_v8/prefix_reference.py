"""CPU-only integrity comparison, never an online input or fitting target."""
from .common import T,read,sha,write,mapped
REFERENCE=T/'outputs/native_frozen100_v6/placement/batches/008/runs/63a39dc9e8ad4d44b70d8529fba8fe75'

def reference_inputs():
 assert REFERENCE.exists()
 files=[REFERENCE/'protocol.json',REFERENCE/'attempt/resources.jsonl',REFERENCE/'failure.json']
 for case in sorted((REFERENCE/'cases').glob('*')):
  files.append(REFERENCE/'accepted'/(case.name+'.json'))
  for p in sorted((case/'baseline').glob('policy*.json')):
   row=read(p);files.extend([p,mapped(row['policy_file']),mapped(row['observation_file'])])
 return {str(p):sha(p)for p in files}

def verify_original_prefix(spec,root):
 if spec['batch_index']!=8:return
 import numpy as np
 checked=0;cases=0;bindings={}
 for oldcase in sorted((REFERENCE/'cases').glob('*')):
  newcase=root/'cases'/oldcase.name
  assert read(root/'accepted'/(oldcase.name+'.json'))==read(REFERENCE/'accepted'/(oldcase.name+'.json'))
  for p in sorted((oldcase/'baseline').glob('policy*.json')):
   old=read(p);fresh=read(newcase/'baseline'/p.name)
   for field in('query','actor_observation_tick','actor_images','command_state_sha256','noise_witness','encoder','base_infer_calls','actual_internal_noise_calls'):
    assert old[field]==fresh[field],('Original scheduling reference differs',oldcase.name,p.name,field)
   for field in('policy_file','observation_file'):
    a,b=mapped(old[field]),mapped(fresh[field])
    assert sha(a)==old[field.replace('_file','_sha256')]and sha(b)==fresh[field.replace('_file','_sha256')]
    with np.load(a,allow_pickle=False)as z,np.load(b,allow_pickle=False)as q:
     assert set(z.files)==set(q.files)
     for key in z.files:assert z[key].dtype==q[key].dtype and np.array_equal(z[key],q[key]),('Original numeric prefix changed',oldcase.name,p.name,field,key)
    bindings[str(a)]=sha(a);bindings[str(b)]=sha(b)
   checked+=1
  cases+=1
 assert cases==10 and checked>=10
 write(root/'CPU_original_prefix_comparison.json',dict(passed=True,original_failed_reference_root=str(REFERENCE),matched_cases=cases,matched_baseline_policy_queries=checked,all_saved_policy_and_observation_arrays_byte_values_equal=True,RGB_commands_sampler_RNG_once_noise_and_tokens_exact=True,no_outcomes_or_labels_read=True,no_reference_forwarded_to_models=True,CPU_only_after_actual_child_exit=True,input_sha256=bindings))

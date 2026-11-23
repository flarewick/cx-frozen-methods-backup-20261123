"""Exact qualified population/preparation, without old semantic/policy results."""
from pathlib import Path
import numpy as np
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify,identity
from native_visible_expand_semantics_v2.store import key
def main():
 o=T/'outputs/native_onpolicy_coverage_v7/pilot_fit';inputs={}
 def bind(f):f=mapped(f);inputs[str(f)]=sha(f);return read(f)
 r=mapped(bind(o/'execution_state.json')['root']);protocol=bind(r/'protocol.json');verify(protocol)
 qualified=bind(r/'atomic_full_recovery_qualification.json');verify(qualified);assert qualified['passed']
 terminal=bind(r/'terminal_audit.json');assert terminal['passed']and terminal['own_GPU_released']
 actual=bind(o/'actual_host_wait_exit.json');assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
 initial=protocol['initial_screen_state'];end=bind(r/'screening_complete.json')['state'];tasks=protocol['tasks'];cache={};keys={};diagnostics={};refs={};requests={};canonical=[]
 for task in tasks:
  state=dict(initial[task]);rows=[];items=[];assert state['accepted']==4 and end[task]['accepted']==8
  while state['accepted']<8:
   f=r/'screening'/task/(str(state['next_seed'])+'.json');row=bind(f);canonical.append(str(f));assert row['task']==task and row['seed']==state['next_seed']and row['episode_index']==state['accepted'];rows.append(row);state['next_seed']+=1
   if not row['eligible']:continue
   cid=task+':'+str(row['seed']);item=bind(r/'accepted'/(cid+'.json'));meta=bind(r/'initial_prefetch'/(cid+'.json'));req=bind(r/'semantic/requests'/(meta['content_key']+'.json'));frame=mapped(req['frame_file']);assert sha(frame)==req['frame_sha256'];inputs[str(frame)]=sha(frame)
   with np.load(frame,allow_pickle=False)as z:assert identity(key([z[f'cam{i}']for i in range(3)],z['command_state'],item['instruction']))==req['content_key']==meta['content_key']
   assert req['instruction']==item['instruction']and item['initial_physical_sha256']==row['initial_physical_sha256']and item['episode_index']==state['accepted']
   ref=r/'expert_references'/(cid+'.json');reference=bind(ref);assert reference['case_id']==cid and reference['expert_plan_success']and reference['expert_task_success']and reference['sampling_stride_physics_steps']==15
   raw=mapped(reference['file']);assert sha(raw)==reference['sha256'];inputs[str(raw)]=sha(raw)
   keys[cid]=meta['content_key'];diagnostics[cid]=meta['diagnostic_identity'];assert set(diagnostics[cid])=={'physical_sha256','rng_sha256'}
   refs[cid]=str(ref);requests[req['content_key']]=req;items.append(item);state['accepted']+=1
  assert state==end[task]and len(items)==4;cache[task]=dict(rows=rows,items=items,end_state=state)
 result=dict(cached_valid_cases=20,task_cache=cache,case_request_keys=keys,case_diagnostics=diagnostics,expert_reference_files=refs,requests=list(requests.values()),input_sha256=inputs,no_policy_outputs_masks_or_labels_imported=True,same_fixed20_TRAIN_development_cases_not_new_independent_data=True,canonical_screen_files=canonical,source_atomic_qualified_root=str(r),historical_C5_preparation_recoveries_used=2,historical_C5_preparation_recoveries_maximum=2,new_automatic_recovery_attempts=0)
 out=T/'outputs/native_onpolicy_preparation_v2';write(out/'actual_preparation_cache.json',result)
 write(out/'CPU_actual_cache_checks.json',dict(passed=True,cached_valid_cases=20,canonical_per_task_prefix_exact=True,earliest_positive_and_negative_decisions_preserved=True,no_policy_masks_actions_or_labels_imported=True,no_CUDA_or_optimizer=True,source_sha256={str(p):sha(p)for p in Path(__file__).resolve().parent.glob('*')if p.is_file()},input_sha256={str(out/'actual_preparation_cache.json'):sha(out/'actual_preparation_cache.json'),**inputs}))
 print('CPU_ATOMIC_V7_FIXED20_PREPARATION_INPUTS_VERIFIED_NO_NEW_INDEPENDENT_DATA')
if __name__=='__main__':main()

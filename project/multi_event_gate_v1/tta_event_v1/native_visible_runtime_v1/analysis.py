"""Complete old full/zero controls plus all new shadow/TTA/wrong whole episodes."""
from pathlib import Path
import argparse,json
import numpy as np
from .paths import T,O,read,sha,write,verify
from .prepare import CONDITIONS
from native_trained_runtime_v1.analysis import arrays,same_query,branch

def audit(spec,root):
 verify(spec);complete=read(root/'GPU_episodes_complete.json')
 assert complete['complete']and complete['protocol_identity']==spec['identity']and complete['frozen_parameter_pointers_and_mutation_counters_unchanged']
 assert len(complete['cases'])==len(spec['scenes'])==4
 oldspec=read(spec['reused_control_protocol']);assert oldspec['identity']==spec['reused_control_protocol_identity']and oldspec['required_gpu_uuid']==spec['required_gpu_uuid']
 records=[];checks=0
 for i,item in enumerate(spec['scenes']):
  folder=root/('case%03d'%i);bundle=read(folder/'bundle.json');control=spec['reused_controls'][i]
  assert bundle['case_id']==control['case_id']==item['case_id']and bundle['all_requested_conditions_retained']
  assert [r['condition']for r in bundle['branches']]==list(CONDITIONS)
  results={};summaries=[]
  for condition in('M_event_full','zero_update',*CONDITIONS):
   if condition in control['conditions']:
    entry=control['conditions'][condition];assert entry['whole_complete_episode']
    for p,h in entry['file_sha256'].items():assert sha(p)==h
    branchfolder=Path(entry['folder']);branchspec=oldspec;role='whole_old_control_original_qualification'
   else:
    witness=next(r for r in bundle['branches']if r['condition']==condition);assert sha(witness['result_file'])==witness['result_sha256']
    branchfolder=folder/condition;branchspec=spec;role='new_candidate_native_episode'
   results[condition],summary=branch(branchfolder,condition,branchspec)
   if condition in CONDITIONS:assert results[condition]['feedback']['action_visible_fixed_parameters_and_buffers_unchanged']
   assert read(branchfolder/'semantic_initial_receipt.json')['initial_diagnostic_identity']==bundle['initial_identity']
   summaries.append(dict(**summary,provenance_role=role,source_folder=str(branchfolder),qualification_sha256=results[condition]['qualification_sha256']))
  full=results['M_event_full']
  for condition in('zero_update','shadow_update'):
   candidate=results[condition];assert candidate['success']==full['success']and candidate['native_action_count']==full['native_action_count']and len(candidate['queries'])==len(full['queries'])
   for a,b in zip(full['queries'],candidate['queries'],strict=True):same_query(a,b);checks+=1
  changes={}
  for condition in('TTA','wrong_feedback'):
   candidate=results[condition];first=None
   for a,b in zip(full['queries'],candidate['queries']):
    same=np.array_equal(arrays(a['actions_file'])['actions'],arrays(b['actions_file'])['actions'])
    same_query(a,b,actions=same,trace=same,actual_actions_only=True);checks+=1
    if not same:
     first=b['query'];assert b['event']['adaptation_applied']and any(u['accepted']and u['query']<=first for u in candidate['feedback']['updates']);break
   if first is None:assert len(full['queries'])==len(candidate['queries'])and full['success']==candidate['success']
   changes[condition]=dict(first_changed_query=first,success=candidate['success'],rescue=bool(not full['success']and candidate['success']),harm=bool(full['success']and not candidate['success']))
  records.append(dict(case_id=item['case_id'],developer_only=True,branches=summaries,matched_changes=changes))
 result=dict(passed=True,requested_developer_cases=4,requested_branches=12,comparison_total_branches=20,reused_whole_control_episodes=8,all_requested_complete=True,matched_query_checks=checks,records=records,zero_shadow_actions_observations_physics_RNG_exact=True,first_change_common_prefix_and_once_noise_exact=True,source_phase_issuing_hidden_horizon_checks=True,new_independent_scenes=0,TTA_utility_proven=False,scope='Four old training/developer scenes; no independent utility claim')
 dest=root/'CPU_episode_audit.json'
 if dest.exists():assert read(dest)==result
 else:write(dest,result)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();print(json.dumps(audit(read(O/'protocol.json'),a.dataset)),flush=True)

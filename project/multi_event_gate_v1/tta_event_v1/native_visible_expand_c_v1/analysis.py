"""All requested developer episodes, actual receipts and matched controls; no utility claim."""
from pathlib import Path
import argparse,json
import numpy as np
from .paths import T,O,read,sha,write,verify
from native_visible_episode_v1.episode import CONDITIONS

def arrays(file,digest=None):
 if digest is not None:assert sha(file)==digest
 with np.load(file,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def equal(a,b):
 return a.keys()==b.keys()and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes()for k in a)
def same_query(a,b,*,actions=True,trace=True,actual_actions_only=False):
 assert a['query']==b['query']and a['tick']==b['tick']
 assert equal(arrays(a['observation']['file'],a['observation']['sha256']),arrays(b['observation']['file'],b['observation']['sha256']))
 assert equal(arrays(a['policy']['policy_file'],a['policy']['policy_sha256']),arrays(b['policy']['policy_file'],b['policy']['policy_sha256']))
 assert a['policy']['noise_witness']==b['policy']['noise_witness']
 ea,eb=a['policy']['environment_audit'],b['policy']['environment_audit']
 assert ea==eb
 if actions:
  left=arrays(a['actions_file'],a['actions_sha256']);right=arrays(b['actions_file'],b['actions_sha256'])
  if actual_actions_only:assert left['actions'].dtype==right['actions'].dtype and left['actions'].tobytes()==right['actions'].tobytes()
  else:assert equal(left,right)
 if trace:
  aa,bb=a['execution'],b['execution']
  assert equal(arrays(aa['actual_trace_file'],aa['actual_trace_sha256']),arrays(bb['actual_trace_file'],bb['actual_trace_sha256']))

def branch(folder,condition,spec):
 result=read(folder/'result.json');assert result['complete']and not result['CPU_fixture_only']
 assert result['condition']==condition and result['qualification_sha256']==sha(spec['trained_qualification'])
 assert result['native_episode_complete']and result['feedback']['fast_reset_on_close']and result['remaining_live_sources']==0
 assert result['reason']in('native_step_limit','native_success_terminal')
 assert result['success']or result['native_action_count']==result['original_native_step_limit']
 assert len(result['queries'])>0 and result['native_action_count']<=result['original_native_step_limit']
 total_ticks=0;actions_started=0;actual_applied=[];source_used=set();query_seconds=[0.]
 for q,row in enumerate(result['queries']):
  assert row['query']==row['policy']['query']==q and row['tick']==total_ticks
  policy=row['policy'];assert policy['base_infer_calls']==policy['actual_internal_noise_calls']==1
  assert policy['environment_audit']['native_environment_unchanged']and policy['environment_audit']['diagnostic_only']
  assert policy['noise_witness']['environment_unchanged']and policy['noise_witness']['before']!=policy['noise_witness']['after']
  proposal=arrays(policy['policy_file'],policy['policy_sha256']);action=arrays(row['actions_file'],row['actions_sha256'])
  assert action['actions'].shape==(32,14)and action['actions'].dtype==np.float32
  assert np.array_equal(action['actions'][:,[6,13]],proposal['base'][:,[6,13]])
  assert np.array_equal(action['base'],proposal['base'])and np.array_equal(action['reference'],proposal['raw_delta'])
  trace=arrays(row['execution']['actual_trace_file'],row['execution']['actual_trace_sha256']);assert np.array_equal(trace['source_commands'],action['actions'])
  receipt=read(folder/('execution%03d.json'%q));assert receipt['query']==q and receipt['global_start_tick']==total_ticks
  assert receipt['committed_ticks']==len(trace['dt_s'])and np.all(trace['dt_s']>0)
  assert [r['offset']for r in receipt['receipts']]==list(range(len(receipt['receipts'])))
  started=sum(r['count_after']-r['count_before']for r in receipt['receipts']);assert 0<=started<=32
  assert sum(r['status']=='returned'for r in receipt['receipts'])==row['execution']['commands_returned']
  if q<len(result['queries'])-1:assert receipt['complete_source_chunk']and row['execution']['commands_returned']==32
  actions_started+=started;total_ticks+=len(trace['dt_s']);query_seconds.append(query_seconds[-1]+float(np.sum(trace['dt_s'])))
  assert row['execution']['end_tick']==total_ticks
  if row['event']['adaptation_applied']:actual_applied.append(q)
  response=row['response']
  if 'source_query'in response:
   source=response['source_query'];assert source not in source_used and q in(source+1,source+2);source_used.add(source)
 assert total_ticks==result['physical_ticks']and actions_started==result['native_action_count']
 assert len(source_used)<=3
 readout=read(folder/'readout_bridge.json');assert readout['native_calls']==total_ticks and readout['all_sampled_fields_byte_exact']and not readout['poisoned']
 release=read(folder/'completed_task_release.json');assert release['completed_task_released']and release['completed_current_scene_released']and release['model_parameter_pointers_and_RNG_unchanged']
 semantic=read(folder/'semantic_initial_receipt.json');assert semantic['all_three_RGB_command_language_exact']
 source_reports=sorted(folder.glob('source*/source.json'));assert len(source_reports)==result['completed_sources']<=3
 forwards=0;packets=[]
 for file in source_reports:
  meta=read(file);source=meta['source_query'];forwards+=meta['owner']['actual_tracking_calls']
  if meta['local_file']is not None:arrays(meta['local_file'],meta['local_sha256'])
  if meta['mask_file']is not None:arrays(meta['mask_file'],meta['mask_sha256'])
  updates=[r for r in result['queries']if r['response'].get('source_query')==source]
  for update in updates:
   q=update['query'];boundary=[b for b in meta['boundaries']if b['query']==q];assert len(boundary)==1
   ids=boundary[0]['eligible_rows'];assert len(ids)==update['response']['rows']and ids
   origin=read(folder/('execution%03d.json'%source));assert origin['complete_source_chunk']
   assert len(origin['receipts'])==32 and all(r['status']=='returned'for r in origin['receipts'])
   phase=origin['completion'][0]['requested95_tick'];assert phase is not None
   z=arrays(origin['actual_trace_file'],origin['actual_trace_sha256']);phase_seconds=query_seconds[source]+float(np.sum(z['dt_s'][:phase]))
   for i in ids:
    local=meta['rows'][i];owner=local['command_query'];assert owner in(source,source+1)and owner<q
    assert 25<=local['end_tick']-local['start_tick']<=125
    assert result['queries'][source]['tick']+phase<=local['start_tick']<local['end_tick']<=update['tick']
    source_masks={r['global_tick']:r['elapsed_episode_s']for r in meta['mask_readouts']}
    assert source_masks[local['start_tick']]>=phase_seconds and source_masks[local['end_tick']]-phase_seconds>=.25
   packets.append(dict(source_query=source,query=q,rows=len(ids),phase95_tick=result['queries'][source]['tick']+phase,actual_issued32=True))
 return result,dict(condition=condition,success=result['success'],native_actions=actions_started,queries=len(result['queries']),physical_ticks=total_ticks,SAM2_forwards=forwards,applied_queries=actual_applied,accepted_updates=sum(r['accepted']for r in result['feedback']['updates']),consumed_packets=len(result['feedback']['updates']),packets=packets,wall_seconds=result['seconds'])

def audit(spec,root):
 verify(spec);complete=read(root/'GPU_episodes_complete.json');assert complete['complete']and complete['protocol_identity']==spec['identity']
 assert complete['frozen_parameter_pointers_and_mutation_counters_unchanged']and len(complete['cases'])==len(spec['scenes'])==4
 records=[];checks=0
 for i,item in enumerate(spec['scenes']):
  folder=root/('case%03d'%i);bundle=read(folder/'bundle.json');assert bundle['case_id']==item['case_id']and bundle['all_requested_conditions_retained']
  assert [r['condition']for r in bundle['branches']]==list(CONDITIONS)
  if not bundle['initialized']:
   assert bundle['initialization_missing']['reason']=='native_initialization_missing'and not bundle['initialization_missing']['frame_available']
   assert all(b['success']is None and b['status']=='not_started_native_initialization_missing'for b in bundle['branches'])
   assert not any((folder/b).exists()for b in CONDITIONS)
   records.append(dict(case_id=item['case_id'],developer_only=True,initialized=False,initialization_missing=bundle['initialization_missing'],branches=bundle['branches'],matched_changes=None));continue
  results={};summaries=[]
  for condition in CONDITIONS:
   witness=next(r for r in bundle['branches']if r['condition']==condition);assert sha(witness['result_file'])==witness['result_sha256']
   results[condition],summary=branch(folder/condition,condition,spec);summaries.append(summary)
   if condition in ('shadow_update','TTA','wrong_feedback'):assert results[condition]['feedback']['action_visible_fixed_parameters_and_buffers_unchanged']
   assert read(folder/condition/'semantic_initial_receipt.json')['initial_diagnostic_identity']==bundle['initial_identity']
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
     first=b['query'];assert b['event']['adaptation_applied']
     assert any(u['accepted']and u['query']<=first for u in candidate['feedback']['updates'])
     break
   if first is None:assert len(full['queries'])==len(candidate['queries'])and full['success']==candidate['success']
   changes[condition]=dict(first_changed_query=first,success=candidate['success'],rescue=bool(not full['success']and candidate['success']),harm=bool(full['success']and not candidate['success']))
  records.append(dict(case_id=item['case_id'],developer_only=True,initialized=True,branches=summaries,matched_changes=changes))
 result=dict(passed=True,requested_developer_cases=4,requested_branches=20,actual_native_episodes=5*sum(r['initialized']for r in records),initialized_cases=sum(r['initialized']for r in records),initialization_missing_cases=[r['case_id']for r in records if not r['initialized']],all_requested_complete=True,matched_query_checks=checks,records=records,zero_shadow_actions_observations_physics_RNG_exact=True,first_change_common_prefix_and_once_noise_exact=True,source_phase_issuing_hidden_horizon_checks=True,new_independent_scenes=0,TTA_utility_proven=False,scope='Four additional fixed developer scenes; part of expanded16, not independent task improvement')
 dest=root/'CPU_episode_audit.json'
 if dest.exists():assert read(dest)==result
 else:write(dest,result)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();print(json.dumps(audit(read(O/'protocol.json'),a.dataset)),flush=True)

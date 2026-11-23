"""NumPy-only audited offline inputs from real full trajectories and recovery."""
from pathlib import Path
import importlib.util
import numpy as np
from native_policy_execution_v1.common import T,read,sha
from native_onpolicy_recovery_v3.common import mapped,verify,identity
from native_onpolicy_arm_teacher_v1.stream import packet,ARM

def arrays(path):
 with np.load(mapped(path),allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}

class Reader:
 def __init__(self):self.inputs={};self.cache={};self.kernel=None
 def file(self,path,digest):
  path=mapped(path);assert sha(path)==digest
  self.inputs[str(path)]=digest
  if str(path)not in self.cache:self.cache[str(path)]=arrays(path)
  return self.cache[str(path)]
 def qualified(self,root):
  root=Path(root).resolve();assert root.is_relative_to((T/'outputs').resolve())
  qual=read(root/'atomic_full_recovery_qualification.json');assert qual['passed']and qual['parent_terminal_passed']and qual['recovery_audit_passed']and qual['no_optimizer'];verify(qual)
  terminal=read(root/'terminal_audit.json');assert terminal['passed']and terminal['actual_host_exit_code']==terminal['actual_child_exit_code']==0 and terminal['own_GPU_released'];verify(terminal)
  out=root.parents[3];actual=read(out/'actual_host_wait_exit.json');assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  assert mapped(read(out/'execution_state.json')['root']).resolve()==root
  spec=read(root/'protocol.json');verify(spec);body=dict(spec);digest=body.pop('identity');assert identity(body)==digest==terminal['batch_protocol_identity']
  if spec.get('governor_extra_margin_required'):
   proof=root/'CPU_governor_margin_audit.json';value=read(proof)
   assert value['passed']and value['additional_cumulative35_and_process60_180_device60_36']and value['original34_and_all_original_guards_preserved']and value['steady_samples']>0
   verify(value);self.inputs[str(proof)]=sha(proof);self.inputs.update(value['input_sha256'])
   source=T/'native_governor_margin_v2/margin.py'
   assert any(mapped(name).resolve()==source.resolve()and digest==sha(source)for name,digest in spec['source_sha256'].items())
  if spec.get('initial_ARM_path_feedback'):
   sources=[mapped(name)if Path(name).is_absolute()else T.parents[1]/name for name in spec['source_sha256']if 'native_onpolicy_arm_collection_v'in str(name)and str(name).endswith('/runtime.py')]
   assert len(sources)==1,'One explicitly source-bound operational ARM runtime required'
   expected=(T/'native_onpolicy_collection_v38/runtime.py').read_text().replace('native_onpolicy_mixed_teacher_v3','native_onpolicy_arm_teacher_v1')
   runtime=sources[0].read_text()
   if spec.get('independent_required_next_population_preparation'):
    assert spec['minimum_actual_current_queries_before_early_preparation']==9
    assert spec['early_fixed_next_population_preparation']==bool(spec.get('future_preparation_population'))
    helper=T/'native_early_next_preparation_v1/scope.py'
    assert any(mapped(name).resolve()==helper.resolve()and digest==sha(helper)for name,digest in spec['source_sha256'].items())
    if spec.get('early_preparation_wait_seconds') is not None:
     assert spec['early_preparation_wait_seconds']==.05
     wait='    future_ready.wait(.05)\n';assert runtime.count(wait)==1
     runtime=runtime.replace(wait,'')
    from native_early_next_preparation_v1.scope import restore as restore_next_schedule
    runtime=restore_next_schedule(runtime)
   if spec.get('future_preparation_uses_unfinished_whole_current_cases'):
    assert spec['future_preparation_minimum_unfinished_current_cases']==4
    assert spec['scheduling_only_revision']['maximum_foreground_before_required_future']==1
    assert runtime.count('with primary_lock:return case_count-len(whole_done)>=4')==1
    runtime=runtime.replace('with primary_lock:return case_count-len(whole_done)>=4','with primary_lock:return case_count-len(primary_done)>=4').replace('primary_done=set();whole_done=set();reference_tasks=set()','primary_done=set();reference_tasks=set()').replace("   with primary_lock:whole_done.add(item['case_id'])\n",'')
   if spec.get('actual_query_gated_startup_lanes'):
    assert spec['actual_query_gated_startup_lanes']==[[0],[1,2,3],[4,5,6,7,8]]
    old_log='dict(lanes=[0],remaining_wait_for_actual_first_query=True'
    assert runtime.count(old_log)==1
    runtime=runtime.replace(old_log,'dict(lanes=[0,1,2,3],remaining_wait_for_actual_first_query=True')
    startup=sources[0].with_name('startup.py')
    assert startup.read_text()==(T/'native_onpolicy_arm_collection_v22/startup.py').read_text()
    assert any(mapped(name).resolve()==startup.resolve()for name in spec['source_sha256'])
   assert runtime==expected,'Scientific runtime change requires new dataset qualification'
   assert (sources[0].parent/'episode.py').read_text()==(T/'native_onpolicy_arm_collection_v5/episode.py').read_text()
  lifecycle=T/'native_expert_case_release_v1/release.py'
  if str(lifecycle.relative_to(T.parents[1]))in spec['source_sha256']:
   proof=root/'CPU_closed_expert_task_release_audit.json';released=read(proof);assert released['passed']and released['all_closed_scene_and_task_and_proxy_weakrefs_dead']and released['one_original_shared_pair_preserved']and released['original_model_and_RNG_preserved'];verify(released)
   self.inputs[str(proof)]=sha(proof);self.inputs.update(released['input_sha256'])
   if spec.get('q0_scene_handoff_with_whole_native_reference_controls'):
    cached=spec.get('cached_preparation')or{}
    expected=spec['cases_per_batch']-cached.get('cached_valid_cases',0)
    tasks={t for t in spec['tasks']if len(cached.get('task_cache',{}).get(t,{}).get('items',[]))<5}
    assert released['closed_after_full_not_after_screen']and released['transferred_q0_tasks']==expected
    if spec.get('cached_q0_semantics_lazy_revalidation'):
     assert released['cached_revalidated_cases']==cached.get('cached_valid_cases',0)
     assert released['all_cached_policy_entries_byte_exact']
     regenerated=released['regenerated_cached_cases'];assert len(regenerated)==len(set(regenerated))
     assert released['cached_regeneration_whole_controls']==len(regenerated)
     assert released['whole_native_reference_controls']==len(tasks)+len(regenerated)
    elif spec.get('cached_q0_regenerated_from_actual_current_native_scene'):
     assert released['current_cached_q0_tasks']==cached.get('cached_valid_cases',0)
     assert released['all_actual_current_q0_tasks']==spec['cases_per_batch']
     assert released['old_cache_physics_RNG_commands_exact']
     assert released['whole_native_reference_controls']==len(spec['tasks'])
    else:assert released['whole_native_reference_controls']==len(tasks)
    if spec.get('obsolete_MPLIB_world_detached_after_expert_q0'):
     assert released['all_obsolete_MPLIB_scenes_released']
  CPU_admission_sources=[mapped(name)if Path(name).is_absolute()else T.parents[1]/name for name in spec['source_sha256']if str(name).endswith('/cpu_admission.py')]
  if CPU_admission_sources:
   assert all(sha(f)==sha(T/'native_onpolicy_collection_v26/cpu_admission.py')for f in CPU_admission_sources),'New CPU admission body requires new reader qualification'
   previous=[r for r in root.parent.iterdir()if r.is_dir()and r!=root];assert len(previous)<=2
   for prior in previous:
    proof=read(prior/'CPU_only_replan_qualified.json');assert proof['passed']and proof['no_GPU_or_scientific_work']and proof['prepared_and_semantic_processes_actually_waited']and proof['no_GPU_clock_or_budget_reset']and proof['not_native_data_qualification']
    assert not(prior/'attempt/barrier/permit.json').exists()
    f=prior/'attempt/resources.jsonl';assert not f.exists()or f.stat().st_size==0
    assert not(prior/'GPU_episodes_complete.json').exists()and not(prior/'terminal_audit.json').exists()
    verify(read(prior/'protocol.json'))
    for name in('CPU_admission_race.json','CPU_only_replan_qualified.json','CPU_admission_prepared_exit.json','semantic_exit.json','source_bound_CPU_preload_proof.json','protocol.json'):
     f=prior/name;self.inputs[str(f)]=sha(f)
    for name in('CPU_admission_prepared_exit.json','semantic_exit.json'):
     actual=read(prior/name);assert actual['actual_Popen_wait_return']and actual.get('proc_absent_after_wait',True)
  if spec.get('CPU_semantic_adaptive_total8_threads'):
   proof=root/'CPU_semantic_thread_budget_audit.json';bounded=read(proof);assert bounded['passed']and bounded['limit8_preserved']and bounded['maximum_active_threads']<=8 and bounded['workers_actually_waited0']
   assert bounded['actual_requests']==read(root/'semantic/result.json')['requests']and bounded['actual_requests']>0;verify(bounded)
   assert all(mapped(name).resolve().is_relative_to((root/'semantic').resolve())for name in bounded['input_sha256'])
   self.inputs[str(proof)]=sha(proof);self.inputs.update(bounded['input_sha256'])
  self.admission_proof(root,spec)
  missing_guard=T/'native_process_missing_guard_v2/guard.py'
  if str(missing_guard.relative_to(T.parents[1]))in spec['source_sha256']:
   assert spec['fresh_valid_NOT_FOUND_unknown_interval_zero_bound_v2']and spec['resource_guard_revision']=='fresh_validated_NOT_FOUND_dense_clock_entire_unknown_interval_zero_lower_bound_v2'
  idle=T/'native_process_idle_guard_v1/guard.py'
  if str(idle.relative_to(T.parents[1]))in spec['source_sha256']:
   assert spec['conservative_idle_missing_lower_bound_guard']
   proof=root/'CPU_idle_missing_lower_bound_audit.json';bounded=read(proof);assert bounded['passed']and bounded['actual_samples']>0 and bounded['steady_samples']>0 and bounded['all_process_windows_recomputed_exact']and bounded['unknown_interval_integrated_at_zero_not_prior_positive']and bounded['instantaneous_missing_values_preserved']and bounded['all_original_average_floors_34']and bounded['no_device_to_process_substitution']
   assert any(mapped(name).resolve()==(root/'protocol.json').resolve()and digest==sha(root/'protocol.json')for name,digest in bounded['input_sha256'].items())
   assert any(mapped(name).resolve()==(root/'attempt/resources.jsonl').resolve()and digest==sha(root/'attempt/resources.jsonl')for name,digest in bounded['input_sha256'].items())
   verify(bounded);self.inputs[str(proof)]=sha(proof);self.inputs.update(bounded['input_sha256'])
  rule4=[T/package/'history.py'for package in('native_server_rule4_v2','native_server_rule4_v3','native_server_rule4_v4','native_server_rule4_v5','native_server_rule4_v6','native_server_rule4_v7','native_server_rule4_v8','native_server_rule4_v9','native_server_rule4_v10')]
  if any(str(source.relative_to(T.parents[1]))in spec['source_sha256']for source in rule4):
   proof=root/'CPU_cautious_rule4_terminal.json';resource=read(proof);assert resource['passed']and resource['actual_active_samples']>0 and resource['original_scientific_resource_thresholds_unchanged'];verify(resource)
   self.inputs[str(proof)]=sha(proof);self.inputs.update(resource['input_sha256'])
  account=T/'native_account_stream_v1/transport.py'
  if str(account.relative_to(T.parents[1]))in spec['source_sha256']:
   proof=root/'CPU_account_transport_terminal.json';stream=read(proof);assert stream['passed']and stream['actual_normal_CPU_workers']==3 and stream['actual_fresh_account_scans']>0 and stream['original_collection_bodies_and_three_second_deadline_unchanged']and stream['no_GPU_metrics_or_accounting_clock_changed'];verify(stream)
   self.inputs[str(proof)]=sha(proof);self.inputs.update(stream['input_sha256'])
  report=read(root/'CPU_recovery_stream_audit.json');assert report['passed']and report['optimizer_steps']==0
  cases=read(root/'GPU_episodes_complete.json');assert cases['complete']and cases['frozen_parameter_pointers_and_mutation_counters_unchanged']and cases['protocol_identity']==digest
  if spec.get('q0_once_policy_overlaps_initial_CPU_semantics_before_any_action'):
   for case in cases['cases']:self.overlap_proof(root/'cases'/case['case_id'])
  assert len(report['records'])==len(cases['cases'])and {x['case_id']for x in report['records']}=={x['case_id']for x in cases['cases']}
  for p in(root/'atomic_full_recovery_qualification.json',root/'terminal_audit.json',root/'CPU_recovery_stream_audit.json',root/'protocol.json',out/'actual_host_wait_exit.json'):self.inputs[str(p)]=sha(p)
  return spec,cases['cases'],report
 def overlap_proof(self,folder):
  result=folder/'M_event_full/result.json';record=read(result);self.inputs[str(result)]=sha(result)
  proof=folder/'M_event_full_q0_policy_semantic_overlap.json'
  if not record['queries']:
   assert not proof.exists(),'No query may claim a policy/semantic overlap'
   return
  first=record['queries'][0];policy=first['policy'];assert first['query']==first['tick']==policy['query']==policy['actor_observation_tick']==0
  assert policy['base_infer_calls']==policy['actual_internal_noise_calls']==1 and policy['environment_audit']['native_environment_unchanged']
  row=read(proof);assert row['query']==row['physical_tick']==0
  for key in('actual_once_q0_before_semantic_resolution','no_actor_execution_before_semantic_resolution','no_additional_query_or_native_step'):assert row[key] is True
  assert type(row['forward_file_precedes_CPU_answer'])is bool  # Cached answers may predate the policy; no latency selection.
  receipt=folder/'M_event_full/semantic_initial_receipt.json';semantic=read(receipt)
  assert semantic['query']==semantic['tick']==0 and semantic['all_three_RGB_command_language_exact']
  for pathkey,hashkey,expectedpath,expectedhash in[
   ('actual_once_policy_file','actual_once_policy_sha256',policy['policy_file'],policy['policy_sha256']),
   ('CPU_semantic_file','CPU_semantic_sha256',semantic['CPU_result_file'],semantic['CPU_result_sha256'])]:
   path=mapped(row[pathkey]);assert path.resolve()==mapped(expectedpath).resolve()and row[hashkey]==expectedhash==sha(path)
   self.inputs[str(path)]=expectedhash
  for path in(proof,receipt):self.inputs[str(path)]=sha(path)
 def admission_proof(self,root,spec):
  if spec.get('verified_native_q0_releases_remaining_real_lanes_before_CPU_semantic_wait'):
   lanes=spec['simulation_lanes'];assert lanes in(8,9,10)
   admission=root/('startup_release_'+str(lanes-4)+'.json');row=read(admission)
   assert row['released_lanes']==list(range(4,lanes))and row['continuous_resource_clock']
   if row['reason']=='actual_verified_native_q0_RGB_command_prefetch':
    assert row['native_action_count']==0 and row['no_original_actor_query_fabricated']
    cid=row['case_id'];assert read(root/'accepted'/(cid+'.json'))['case_id']==cid
    initial=root/'initial_prefetch'/(cid+'.json');meta=read(initial);assert meta['content_key']==row['content_key']
    req=root/'semantic/requests'/(row['content_key']+'.json');request=read(req);frame=mapped(request['frame_file']);assert sha(frame)==request['frame_sha256']
    for f in(initial,req,frame):self.inputs[str(f)]=sha(f)
   else:
    assert (spec.get('cached_preparation')or{}).get('cached_valid_cases',0)>0
    assert row['reason']in('actual_once_policy_query_complete','actual_case_terminal')and 0<=row['lane']<4
    if row['reason']=='actual_once_policy_query_complete':assert row['query']==row['tick']==0
   self.inputs[str(admission)]=sha(admission)
 def predictor(self):
  if self.kernel is None:
   root=T/'outputs/native_phase_feedback_v1/frozen_auxiliary';protocol=read(root/'protocol.json');assert protocol['auxiliary_candidate_frozen']and protocol['primary_model']=='correct'and protocol['ridge']==1.
   file=root/'predictor.npz';assert sha(file)==protocol['checkpoint_sha256']=='86069427fa4a4e2f9386227e35bdaebbcd18a70773dff4b7686ef83b854fb0c1'
   source=T/'native_local_response_v2/model.py';assert any(mapped(p).resolve()==source.resolve()and h==sha(source)for p,h in protocol['source_sha256'].items())
   spec=importlib.util.spec_from_file_location('actual_frozen_onpolicy_coordinates',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   state=arrays(file)
   for a in state.values():a.setflags(write=False)
   self.kernel=(module,state)
   for p in(root/'protocol.json',file,source):self.inputs[str(p)]=sha(p)
  return self.kernel
 def example(self,root,spec,case,row):
  # The caller must have verified this exact atomic parent, not a short probe.
  assert str(Path(root)/'atomic_full_recovery_qualification.json')in self.inputs
  assert row['case_id']==case['case_id']and row['label_available']and row['provisional_until_entire_parent_batch_qualifies']
  folder=Path(root)/'cases'/case['case_id']/'M_event_full';p=packet(spec,folder,case);assert p is not None and p['query']==row['query']and p['choice']['application_mode']==row['application_mode']
  record=read(folder/'result.json');q=p['choice']['packet']['query'];source=p['choice']['packet']['source_query'];packetrow=p['choice']['packet'];byq={r['query']:r for r in record['queries']}
  assert len(byq)==len(record['queries'])and sorted(byq)==list(range(len(byq)))
  local=self.file(packetrow['source_local_file'],packetrow['source_local_sha256']);ids=np.asarray(packetrow['row_ids'],np.int64);assert len(ids)>0 and len(set(ids.tolist()))==len(ids)
  issuing=local['command_query'][ids];assert set(issuing.tolist())<=set((source,source+1))and np.all(issuing<q)
  hidden=[]
  for commandq in issuing:
   policy=byq[int(commandq)]['policy'];a=self.file(policy['policy_file'],policy['policy_sha256']);assert a['hidden'].shape==(32,64);hidden.append(a['hidden'])
  selected=byq[p['query']];current=byq[q]
  def policy(row):
   v=row['policy'];obs=self.file(row['observation']['file'],row['observation']['sha256']);actual=self.file(v['observation_file'],v['observation_sha256']);assert v['actor_observation_tick']==row['tick']and actual.keys()==obs.keys()and all(np.array_equal(actual[k],obs[k])for k in obs)
   return self.file(v['policy_file'],v['policy_sha256'])
  current=policy(current);action=policy(selected);label=self.file(row['target_file'],row['target_sha256']);valid=label['valid_offsets'];assert valid.dtype==np.int64 and len(valid)>0 and np.all((valid>=0)&(valid<32))
  assert np.array_equal(label['once_base_ARM'],action['base'][valid][:,ARM])and np.array_equal(label['direction_target_ARM'],np.clip(label['teacher_ARM']-label['once_base_ARM'],-.1,.1).astype(np.float32))
  module,state=self.predictor();xx,u,kind,_=module.coordinates(local['context'][ids],local['command'][ids],local['start_points'][ids],'correct');prediction=module.predict(state,xx,u,kind)
  result=dict(hidden=np.stack(hidden),context=xx,command=u,kind=kind,kernel_prediction=prediction,target=local['target'][ids],next_hidden=current['hidden'][None],next_logits=current['logits'][:,ARM][None],next_reference=current['raw_delta'][:,ARM][None],action_hidden=action['hidden'][None],action_logits=action['logits'][:,ARM][None],action_reference=action['raw_delta'][:,ARM][None],action_valid_offsets=valid.copy(),action_teacher_target=label['direction_target_ARM'])
  assert all(np.isfinite(a).all()for a in result.values())and result['hidden'].shape==(len(ids),32,64)and result['command'].shape==(len(ids),194)and result['action_teacher_target'].shape==(len(valid),12)
  result.update(application_mode=row['application_mode'],actual_feedback_query=q,offline_action_query=p['query'])
  return result

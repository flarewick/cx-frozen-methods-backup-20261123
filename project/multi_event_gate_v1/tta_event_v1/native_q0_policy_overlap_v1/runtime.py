"""Enumerated reversible edits of the current full collector runtime only."""
def edits():
 old="     expected,masks,seed_report=prepared['seed_store'].at_initial(initial,item['instruction'],query=0,tick=0)"
 new="""     deferred_semantics=branch=='M_event_full'
     expected=masks=seed_report=None
     def resolve_semantic(record,policy):
      nonlocal expected,masks,seed_report
      expected,masks,seed_report=prepared['seed_store'].at_initial(initial,item['instruction'],query=0,tick=0)
      if policy is not None:
       from pathlib import Path
       assert record['query']==policy['query']==0 and record['global_tick']==0
       assert policy['base_infer_calls']==policy['actual_internal_noise_calls']==1
       assert policy['environment_audit']['native_environment_unchanged']
       answer=Path(seed_report['CPU_result_file']);forward=Path(policy['policy_file'])
       write(folder/(branch+'_q0_policy_semantic_overlap.json'),dict(query=0,physical_tick=0,actual_once_policy_file=str(forward),actual_once_policy_sha256=policy['policy_sha256'],CPU_semantic_file=str(answer),CPU_semantic_sha256=seed_report['CPU_result_sha256'],actual_once_q0_before_semantic_resolution=True,forward_file_precedes_CPU_answer=forward.stat().st_mtime_ns<answer.stat().st_mtime_ns,no_actor_execution_before_semantic_resolution=True,no_additional_query_or_native_step=True))
      return expected,bool(seed_report['available'])
     if not deferred_semantics:resolve_semantic(None,None)"""
 old_call="     result=episode(task,actor,raw,condition='M_event_full',event_config=EventConfig(**spec['event_config']),response_model=prepared['response_model'],qualification=spec['trained_qualification'],predictor=prepared['predictor'],source_factory=factory,seed_available=seed_report['available'],expected_initial=expected,instruction=item['instruction'],folder=branchfolder,retire=retire,check_cancel=c.check_cancel)"
 new_call="""     runner=episode if deferred_semantics else episode._q0_original
     overlap_kwargs=dict(semantic_bootstrap=resolve_semantic,bootstrap_initial=initial)if deferred_semantics else{}
     result=runner(task,actor,raw,condition='M_event_full',event_config=EventConfig(**spec['event_config']),response_model=prepared['response_model'],qualification=spec['trained_qualification'],predictor=prepared['predictor'],source_factory=factory,seed_available=None if deferred_semantics else seed_report['available'],expected_initial=expected,instruction=item['instruction'],folder=branchfolder,retire=retire,check_cancel=c.check_cancel,**overlap_kwargs)"""
 old_cleanup='    task_ref=weakref.ref(task);scene_ref=weakref.ref(task.scene);task=None;raw=None;actor=None;factory=None;native_diagnostic=None'
 new_cleanup=old_cleanup+';resolve_semantic=None;overlap_kwargs=None;runner=None'
 return[(old,new),(old_call,new_call),(old_cleanup,new_cleanup)]

def transform(text):
 before=text
 for old,new in edits():
  assert text.count(old)==1,('Unexpected collector source',old[:70]);text=text.replace(old,new)
 assert restore(text)==before
 return text

def restore(text):
 for old,new in reversed(edits()):
  assert text.count(new)==1,('Unexpected overlapped collector source',new[:70]);text=text.replace(new,old)
 return text

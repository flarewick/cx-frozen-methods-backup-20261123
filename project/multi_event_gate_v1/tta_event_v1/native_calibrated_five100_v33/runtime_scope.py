"""Exact production edits, reversible relative to frozen C17 runtime."""
def transform(old):
 text=old.replace('native_onpolicy_collection_v17','native_calibrated_five100_v33')
 edits=[
 ("c=Coordinator(cuda=True,native=load_native(),native_step=load_step());c.rng.install()", "c=Coordinator(cuda=True,native=load_native(),native_step=load_step());c.rng.install()\n from .setup_profile import install as profile_setup\n original_prepare_task=c.prepare_task\n def profiled_task(task,lane):return original_prepare_task(profile_setup(task,root/'native_setup_profile.jsonl'),lane)\n c.prepare_task=profiled_task"),
 ("cleanup=[];actor=None;factory=None;native_diagnostic=None;handoff_integrity=None", "cleanup=[];actor=None;factory=None;native_diagnostic=None;handoff_integrity=None;cached_revalidated=False"),
 ("meta=read(root/'initial_prefetch'/(item['case_id']+'.json'));diagnostic=meta.get('diagnostic_identity')", "meta_file=root/'initial_prefetch'/(item['case_id']+'.json');cached_file=root/'cached_initial_prefetch'/(item['case_id']+'.json')\n     meta=read(meta_file if meta_file.exists()else cached_file);diagnostic=meta.get('diagnostic_identity')"),
 ("if captured is None:\n      initial=prepare_initial", "if captured is None and branch=='M_event_full'and cached_file.exists():\n      from .cached_revalidation import prepare as revalidate_cached\n      initial,changed=revalidate_cached(task,item,cfg,raw,routed_seed,spec=spec,meta=meta,prepared=prepared,root=root,unstable=UnStableError,replay=replay_prefix)\n      meta=read(meta_file);cached_revalidated=True\n      if changed:\n       reference=True;branches.append('handoff_reference')\n     elif captured is None:\n      initial=prepare_initial"),
 ("if handoff_integrity is not None:\n      assert result", "if cached_revalidated:\n      assert result['queries']and result['queries'][0]['query']==0\n      write(folder/(branch+'_initialization_integrity.json'),dict(actual_current_cached_q0=True,actual_policy_entry_independent_RGB_command_check_completed=True,old_physics_RNG_commands_exact=True,requires_whole_fresh_full_control=changed))\n     if handoff_integrity is not None:\n      assert result")]
 edits.extend([(" install_forward_cache(c,torch,models,base,cache,root/'forward_allocator_reclamation.jsonl')"," check_allocator=install_forward_cache(c,torch,models,base,cache,root/'forward_allocator_reclamation.jsonl')"),("      with sam2_profile():return fn(*args,**kwargs)","      check_allocator()\n       with sam2_profile():return fn(*args,**kwargs)")])
 for before,after in edits:
  assert text.count(before)==1,before
  text=text.replace(before,after)
 return text

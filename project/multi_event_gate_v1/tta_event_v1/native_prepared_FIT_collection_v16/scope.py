from native_prepared_FIT_collection_v13.scope import transform as previous_transform,restore_runtime as previous_restore
M={'native_onpolicy_arm_collection_v58': 'native_onpolicy_arm_collection_v61', 'native_onpolicy_calibration_data_v41': 'native_onpolicy_calibration_data_v44', 'native_prepared_FIT_collection_v13': 'native_prepared_FIT_collection_v16'}
PLAN="  newest=T/'outputs/native_onpolicy_full_ops_v2/ARM59_actual_resource_result_v1.json';nv=read(newest);verify(nv);assert nv['release']['own_gpu_released']and nv['cumulative_stage_GPU_seconds']==3641.891\n  if split=='fit'and number==0:v['inherited_stage_GPU_seconds']=nv['cumulative_stage_GPU_seconds']\n  proofs=[T/'outputs/native_guide_reuse_v1/CPU_component_checks_v1.json',T/'outputs/native_guide_reuse_v1/component_CPU_actual_exit_v1.json',T/'outputs/native_guide_reuse_v1/CPU_full_model_checks_v1.json',T/'outputs/native_guide_reuse_v1/full_CPU_actual_exit_v1.json']\n  from native_direct_frames_v2.verify import verify_proof\n  for file in proofs:\n   x=read(file);assert verify_proof(x)\n   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_Popen_wait_return']and x['proc_absent_after_wait']\n   else:assert x['passed']\n  v['existing_guide_reuse_100us_typecheck_static_heap_batch3_prefix']=True\n  v['guide_prefix_initialization_warm_forward_count']=2\n  v['guide_current_card_eager_feature_checks_per_task']=3\n  v['action_sampler_numerical_path_unchanged_Deferred_AOT_fix_not_applicable']=True\n  v['input_sha256'].update({str(p):sha(p)for p in[newest,*proofs]})\n  v['source_sha256'].update({str(p):sha(p)for folder in['native_guide_reuse_v1','native_prepared_FIT_collection_v16','native_batched_prefix_graph_v1']for p in(T/folder).glob('*')if p.is_file()})\n  v['source_sha256'].update({str(p):sha(p)for p in[T.parents[1]/'deployment/benefit_stage_typecheck.py',T.parents[1]/'deployment/benefit_prefix_graph.py',T.parents[1]/'deployment/benefit_prefix_graph_lifetime_v10.py']})\n"
INSTALL=" from native_guide_reuse_v1.prefix import Prefix\n guide_prefix=Prefix(c,torch,base,denoise_graph,render_overlap,manifest['gpu_uuid'])\n"
REPORT=" write(root/'guide_prefix_runtime.json',guide_prefix.report())\n"
CLEANUP='  guide_prefix.actor.batched_encode=guide_prefix.original\n'
HEAP=' from native_guide_reuse_v1.heap import freeze_prepared\n freeze_prepared(prepared,root)\n'
TYPECHECK='    from native_guide_reuse_v1.typecheck import install as install_guide_typecheck\n    install_guide_typecheck()\n'
HOST_ANCHOR="        write(root / 'atomic_full_recovery_qualification.json',"
HOST_EXTRA="        from native_guide_reuse_v1.audit import audit as audit_guide\n        write(root/'CPU_guide_reuse_audit.json',audit_guide(root,spec))\n"
def rename(text):
 for a,b in M.items():text=text.replace(a,b)
 return text

def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='runtime.py':
  anchor=' c.native=render_overlap\n';assert text.count(anchor)==1;text=text.replace(anchor,anchor+INSTALL)
  anchor=" write(root/'render_overlap_runtime.json',render_overlap.report())\n";assert text.count(anchor)==1;text=text.replace(anchor,REPORT+anchor)
  anchor='  render_overlap.close()\n';assert text.count(anchor)==1;text=text.replace(anchor,CLEANUP+anchor)
 if name=='preload.py':
  anchor=' prepared=prepare(spec,root)\n';assert text.count(anchor)==1;text=text.replace(anchor,anchor+HEAP)
  anchor='    install_direct_frames()\n';assert text.count(anchor)==1;text=text.replace(anchor,anchor+TYPECHECK)
 if name=='python_scheduling.py':
  assert text.count('INTERVAL_SECONDS=.001')==1;text=text.replace('INTERVAL_SECONDS=.001','INTERVAL_SECONDS=.0001')
 if name=='collection_plan.py':
  anchor='  from native_unread_capture_full_v1.budget import require_planned_full_batch\n';assert text.count(anchor)==1;text=text.replace(anchor,PLAN+anchor)
 if name=='host.py':
  assert text.count(HOST_ANCHOR)==1;text=text.replace(HOST_ANCHOR,HOST_EXTRA+HOST_ANCHOR)
  text=text.replace("for p in (root/'CPU_bounded_state_audit.json',", "for p in (root/'CPU_guide_reuse_audit.json',root/'CPU_bounded_state_audit.json',")
 return text.replace('outputs/native_onpolicy_arm_collection_v61/CPU_contract_checks_v1.json','outputs/native_onpolicy_arm_collection_v61/CPU_contract_checks_v2.json').replace('outputs/native_onpolicy_arm_collection_v61/CPU_actual_host_exit_v1.json','outputs/native_onpolicy_arm_collection_v61/CPU_actual_host_exit_v2.json')

def restore_runtime(text):
 for value in[INSTALL,REPORT,CLEANUP]:assert text.count(value)==1;text=text.replace(value,'')
 return previous_restore(text)

from native_policy_execution_v1.common import read,sha

def validate(v,tasks,gpu):
 assert v['passed']and v['GPU_uuid']==gpu and v['actual_constant_owner_retained']and v['no_extra_policy_queries_or_episodes']
 assert set(v['task_conformance'])==set(tasks)and all(n==3 for n in v['task_conformance'].values())
 assert len(v['conformance_rows'])==v['original_eager_verification_calls']==3*len(tasks)
 assert all(r['features_metadata_RNG_exact']and r['actual_GPU_uuid']==gpu for r in v['conformance_rows'])
 assert v['original_prefix_warm_forwards']==2 and v['capture_recordings']==1
 return True

def audit(root,spec):
 file=root/'guide_prefix_runtime.json';v=read(file);validate(v,spec['tasks'],spec['required_gpu_uuid'])
 heap=root/'CPU_static_heap_freeze.json';h=read(heap);assert h['passed']and h['CPU_only']and not h['CUDA_initialized']and h['original_all_gc_collect_calls_preserved']and h['automatic_GC_unchanged']
 raw=root/'CPU_storage_whole25_audit.json';r=read(raw);assert r['passed']and r['whole_pairs']==25 and r['all_raw_feedback_and_original_trajectory_arrays_exact']
 return dict(passed=True,current_card_original_eager_feature_conformance=True,static_CPU_heap_freeze_and_original_GC_retained=True,whole25_original_raw_controls_required=True,input_sha256={str(p):sha(p)for p in[file,heap,raw]})

from native_policy_execution_v1.common import read,sha

def validate(value):
 assert value['passed']and value['worker_pool_closed']and value['no_extra_capture_or_inference']
 assert value['original_native_calls']>0 and value['original_owner_requests']==len(value['requests'])>0
 assert len(value['worker_threads'])==1 and value['owner_thread']not in value['worker_threads']
 for row in value['requests']:
  assert row['owner_thread']==value['owner_thread']and row['method']in('encode','sample','seed','check','diagnostic_owner','operation')
  if row['method']in('encode','sample'):assert row['lane'].endswith('_encoder'if row['method']=='encode'else'_sampler')
  if row['method']=='seed':assert row['lane'].endswith(('_encoder','_sampler'))
  if row['method']=='sample':assert row['graph_initialized']
 return True

def audit(root,spec):
 assert spec['native_original_render_overlap_safe_owner_requests']
 file=root/'render_overlap_runtime.json';value=read(file);validate(value)
 return dict(passed=True,original_native_calls=value['original_native_calls'],original_owner_requests=value['original_owner_requests'],original_native_worker_and_single_Torch_owner=True,original_whole25_controls_separately_required=True,input_sha256={str(file):sha(file)})

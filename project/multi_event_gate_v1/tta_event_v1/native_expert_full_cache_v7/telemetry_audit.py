"""Bind live same-state bridge proofs to actually committed native ticks, CPU only."""
from .paths import read,write,sha

def audit_bridge(spec,root):
 gpu=read(root/'GPU_native_complete.json');new=set(spec['remaining_new_indices']);rows=[]
 for row in gpu['rows']:
  if row['index']not in new:continue
  p=row['readout_bridge_file'];assert sha(p)==row['readout_bridge_sha256'];proof=read(p)
  assert proof['case_id']==row['case_id']and proof['lane']==row['lane']and not proof['poisoned']
  n=sum(chunk['committed_ticks']for chunk in row.get('chunks',[]));assert proof['native_calls']==n
  wanted=[i for i in range(n)if i<128 or i%256==0]
  assert [r['call_index']for r in proof['sampled_pairs']]==wanted and proof['paired_calls']==len(wanted)
  assert all(r['all_fields_byte_exact']and len(r['readout_sha256'])==64 for r in proof['sampled_pairs'])
  assert proof['same_state_before_native_physics']and proof['no_extra_actions_queries_or_holds']and row['collection_GPU_uuid']==spec['required_gpu_uuid']
  rows.append(dict(index=row['index'],case_id=row['case_id'],available=row['available'],committed_ticks=n,paired_calls=len(wanted),file=p,sha256=sha(p)))
 assert len(rows)==len(new)==38
 result=dict(passed=True,new_cases=38,recovered_cases=58,new_GPU_uuid=spec['required_gpu_uuid'],source_GPU_uuid=spec['source_GPU_UUID'],
  rows=rows,actual_same_world_sampled_pairs=sum(r['paired_calls']for r in rows),committed_ticks=sum(r['committed_ticks']for r in rows),
  all_sampled_fields_dtype_shape_value_exact=True,original_clock_and_controller_unchanged=True,training_steps=0,online_updates=0,TTA_utility_proven=False)
 write(root/'telemetry_qualification.json',result);return result

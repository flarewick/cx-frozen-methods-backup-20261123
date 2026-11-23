import json,hashlib
from pathlib import Path
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped

def prefix(provider):
 path=mapped(provider.get('observation_file',provider.get('file'))).resolve()
 assert any(path.is_relative_to((T/'outputs'/p).resolve())for p in ['native_server_rule4_v2','native_server_rule4_v3','native_server_rule4_v4','native_server_rule4_v5','native_server_rule4_v6','native_server_rule4_v7','native_server_rule4_v8','native_server_rule4_v9','native_server_rule4_v10','native_server_rule4_v12','native_server_rule4_v13'])
 size=provider.get('complete_byte_prefix',provider.get('complete_prefix_bytes'))
 if size is None:
  assert path.stat().st_size<=64*2**20 and sha(path)==provider['sha256'];size=path.stat().st_size;expected=provider['sha256']
 else:expected=provider.get('byte_prefix_sha256')
 assert 0<size<=64*2**20
 with path.open('rb')as f:data=f.read(size)
 assert len(data)==size and data.endswith(b'\n')and hashlib.sha256(data).hexdigest()==expected
 for key,hashkey in [('parent_protocol','parent_protocol_sha256'),('immutable_admission_proof','immutable_admission_proof_sha256'),('admission_proof','admission_proof_sha256'),('actual_parent_exit','actual_parent_exit_sha256'),('CPU_only_protocol','protocol_sha256')]:
  if key in provider:assert sha(mapped(provider[key]))==provider[hashkey]
 for line in data.splitlines():
  row=json.loads(line);assert row['no_CUDA_or_GPU_work']and row['scientific_GPU_clock_not_reset']and row['fleet']['read_only']
 return dict(provider,file=str(path),complete_prefix_bytes=size,byte_prefix_sha256=expected)

def validate(value):
 assert value['passed']and value['all_original_and_recent_provider_prefixes_exact']and value['no_unknown_prefix_or_gap_credit']and value['only_resource_admission_history_not_scientific_metrics']
 for provider in value['bound_providers']:prefix(provider)
 return True

def audit(root,output=None):
 root=Path(root);spec=read(root/'protocol.json');assert spec['authenticated_recent_CPU_rule4_history_required']
 admission=root/'attempt/rule4_admission.json';a=read(admission);assert a['passed']and a['no_CUDA_or_GPU_work']
 file=mapped(a['observation_file']).resolve();monitor=file.parent;assert any(monitor.is_relative_to((T/('outputs/'+name)).resolve())for name in ['native_server_rule4_v12','native_server_rule4_v13'])
 base=monitor/'CPU_history_seed.json';recent=monitor/'CPU_history_recent_extension.json';old=read(base);new=read(recent)
 assert old['passed']and new['passed']and old['current_boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert sha(base)==new['original_seed_sha256']and new['no_unknown_prefix_or_gap_credit']and new['no_scientific_GPU_metrics_or_clocks_imported']
 providers=old['providers']+new['recent_providers'];bound=[prefix(p)for p in providers]
 # Bind the exact current-monitor complete prefix through its actual admitted
 # observation. Later metadata cannot change this decision or its prior history.
 data=file.read_bytes();assert len(data)<=64*2**20
 lines=[];found=False
 for line in data.splitlines(keepends=True):
  if not line.endswith(b'\n'):break
  lines.append(line);row=json.loads(line)
  if row['monotonic']==a['latest_observation']['monotonic']:
   assert row==a['latest_observation'];found=True;break
 assert found;raw=b''.join(lines);current=dict(file=str(file),complete_prefix_bytes=len(raw),byte_prefix_sha256=hashlib.sha256(raw).hexdigest());prefix(current);bound.append(current)
 for e in a['estimates'].values():assert e['unobserved_prefix_credit']==0
 # Original V11 audit separately validates original fresh75/50,120/55 and
 # all ongoing decisions; immutable constructor uses original History methods.
 value=dict(passed=True,all_original_and_recent_provider_prefixes_exact=True,no_unknown_prefix_or_gap_credit=True,only_resource_admission_history_not_scientific_metrics=True,bound_providers=bound,source_bound_original_threshold_methods_preserved=True,original_provider_count=len(old['providers']),recent_provider_count=len(new['recent_providers']),input_sha256={str(p):sha(p)for p in[admission,base,recent]})
 validate(value);write(output or root/'CPU_authenticated_history_chain_audit.json',value);return value

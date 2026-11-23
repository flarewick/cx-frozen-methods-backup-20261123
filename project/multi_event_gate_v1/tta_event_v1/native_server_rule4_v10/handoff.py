"""Reuse actual CPU eligibility observations, never GPU scientific metrics."""
from pathlib import Path
import hashlib,json,time,os
T=Path(__file__).resolve().parents[1];P=T.parents[1]
def sha_bytes(v):return hashlib.sha256(v).hexdigest()
def read(p):return json.loads(p.read_text())
def seed(history,output):
 providers={};frames=[];now=time.monotonic();offset=time.time()-now;rejected=[]
 for version in ('native_onpolicy_collection_v7','native_onpolicy_collection_v8','native_onpolicy_collection_v9','native_onpolicy_collection_v10','native_onpolicy_collection_v11','native_onpolicy_collection_v12','native_onpolicy_collection_v13','native_onpolicy_collection_v14','native_onpolicy_collection_v15','native_onpolicy_collection_v16','native_initial_arm_mechanism_v1','native_onpolicy_arm_collection_v1','native_onpolicy_collection_v17','native_onpolicy_collection_v18','native_onpolicy_collection_v19','native_onpolicy_collection_v20','native_onpolicy_collection_v21','native_onpolicy_collection_v22','native_onpolicy_collection_v23','native_onpolicy_collection_v24','native_initial_arm_mechanism_v2','native_initial_arm_mechanism_v3','native_initial_arm_mechanism_v4','native_onpolicy_arm_collection_v2','native_onpolicy_collection_v25','native_onpolicy_collection_v26','native_onpolicy_collection_v27','native_calibrated_validation_v1','native_calibrated_five100_v1'):
  for state in (T/'outputs'/version).glob('*/execution_state.json'):
   root=Path(read(state)['root']);permit=root/'attempt/rule4_admission.json'
   if not permit.exists():continue
   proof=read(permit);assert proof['passed']and proof['no_CUDA_or_GPU_work'];file=Path(proof['observation_file']).resolve()
   package=file.parent.parent.name
   if package not in ('native_server_rule4_v2','native_server_rule4_v3','native_server_rule4_v4','native_server_rule4_v5','native_server_rule4_v6','native_server_rule4_v7','native_server_rule4_v8','native_server_rule4_v9','native_server_rule4_v10')or not file.is_relative_to((T/'outputs'/package).resolve()):raise ValueError('Invalid CPU history provider path')
   if file in providers:continue
   protocol_file=root/'protocol.json';protocol=read(protocol_file)
   for source in (T/package).glob('*'):
    if source.is_file()and source.suffix in('.py','.md'):
     assert protocol['source_sha256'].get(str(source.relative_to(P)))==sha_bytes(source.read_bytes()),('CPU history producer source changed',source)
   data=file.read_bytes()
   if len(data)>64*2**20:rejected.append(dict(file=str(file),reason='bounded64MiB provider limit'));continue
   prefix=data if data.endswith(b'\n')else data.rsplit(b'\n',1)[0]+b'\n'
   if not prefix.strip():continue
   count=0
   for line in prefix.splitlines():
    row=json.loads(line)
    assert row['no_CUDA_or_GPU_work']and row['scientific_GPU_clock_not_reset']and row['fleet']['read_only']
    # Same machine/boot clock relation; no invented intervals across reboots.
    if row['monotonic']>now or abs(row['observed_unix_ns']/1e9-row['monotonic']-offset)>5:continue
    ages={int(pid):(int(value[0]),float(value[1]))for pid,value in row['process_ages'].items()}
    frames.append((row['monotonic'],row['fleet'],ages));count+=1
   providers[file]=dict(observation_file=str(file),complete_byte_prefix=len(prefix),byte_prefix_sha256=sha_bytes(prefix),parent_protocol=str(protocol_file),parent_protocol_sha256=sha_bytes(protocol_file.read_bytes()),immutable_admission_proof=str(permit),immutable_admission_proof_sha256=sha_bytes(permit.read_bytes()),valid_frames=count)
 # Standalone CPU-only observation persists across stage failures; no GPU work is qualified.
 observer=T/'outputs/native_onpolicy_full_ops_v2/CPU_only_rule4_observer_v1/protocol.json'
 if observer.exists():
  protocol=read(observer);assert protocol['CPU_only']and protocol['no_GPU_work_or_scientific_clock_change']
  for name,digest in protocol['source_sha256'].items():assert sha_bytes(Path(name).read_bytes())==digest
  root=Path(protocol['root']).resolve();assert root.is_relative_to((T/'outputs/native_server_rule4_v9').resolve())
  file=root/'observations.jsonl'
  if file.exists()and file not in providers:
   data=file.read_bytes();assert len(data)<=64*2**20
   prefix=data if data.endswith(b'\n')else data.rsplit(b'\n',1)[0]+b'\n';count=0
   for line in prefix.splitlines():
    row=json.loads(line);assert row['no_CUDA_or_GPU_work']and row['scientific_GPU_clock_not_reset']and row['fleet']['read_only']
    if row['monotonic']>now or abs(row['observed_unix_ns']/1e9-row['monotonic']-offset)>5:continue
    ages={int(pid):(int(value[0]),float(value[1]))for pid,value in row['process_ages'].items()}
    frames.append((row['monotonic'],row['fleet'],ages));count+=1
   providers[file]=dict(observation_file=str(file),complete_byte_prefix=len(prefix),byte_prefix_sha256=sha_bytes(prefix),parent_protocol=str(observer),parent_protocol_sha256=sha_bytes(observer.read_bytes()),standalone_CPU_only_observer=True,valid_frames=count)
 # Multiple monitors may overlap. Sorting does not duplicate elapsed time;
 # History integrates only positive gaps<=8 and retains original PID age.
 unique={}
 for frame in frames:
  key=(frame[0],json.dumps(frame[1],sort_keys=True));unique[key]=frame
 ordered=sorted(unique.values(),key=lambda f:f[0])
 for observed,fleet,ages in ordered:history.update(fleet,observed,ages)
 report=dict(passed=True,imported_frames=len(ordered),providers=list(providers.values()),rejected=rejected,current_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unobserved_prefix_credit=0,only_resource_admission_history=True,no_scientific_GPU_metrics_or_clocks_imported=True,no_policy_data_masks_labels_or_weights_imported=True)
 with(Path(output)/'CPU_history_seed.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 return report

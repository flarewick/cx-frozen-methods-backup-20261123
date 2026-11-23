"""Keep the SAME ready CPU process while the selected GPU is occupied."""
from pathlib import Path
import json,os,time
from native_policy_execution_v1.common import read,write,sha

def observe(root,gpu,fleet,process):
 assert not(root/'attempt/barrier/permit.json').exists()
 resources=root/'attempt/resources.jsonl'
 assert not resources.exists()or resources.stat().st_size==0
 assert process.poll()is None,'Prepared CPU process exited while waiting'
 launch=read(root/'attempt/launch.json')
 assert launch['pid']==process.pid
 p=Path('/proc',str(process.pid));assert p.stat().st_uid==os.getuid()
 ticks=int((p/'stat').read_text().split(') ',1)[1].split()[19])
 assert ticks==launch['started_ticks']
 ready=read(root/'source_bound_CPU_preload_proof.json');spec=read(root/'protocol.json')
 assert ready['passed']and not ready['cuda_initialized']and ready['os_threads']==1
 assert ready['source_sha256']==spec['source_sha256']
 row=dict(observed_unix_ns=time.time_ns(),gpu=gpu,prepared_pid=process.pid,prepared_started_ticks=ticks,
  no_GPU_permit=True,no_GPU_resource_samples=True,same_ready_CPU_process_retained=True,
  no_GPU_reservation_or_account_lock_held=True,fresh_fleet=fleet,
  original_CPU_preload_proof_sha256=sha(root/'source_bound_CPU_preload_proof.json'))
 with(root/'CPU_retained_preload_wait.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 return row

def audit(root):
 file=root/'CPU_retained_preload_wait.jsonl';permit=root/'attempt/barrier/permit.json'
 assert permit.exists()
 rows=[json.loads(l)for l in file.read_text().splitlines()]if file.exists()else[]
 preload=root/'source_bound_CPU_preload_proof.json';launch=read(root/'attempt/launch.json')
 for row in rows:
  assert row['no_GPU_permit']and row['no_GPU_resource_samples']and row['same_ready_CPU_process_retained']
  assert row['no_GPU_reservation_or_account_lock_held']
  assert row['observed_unix_ns']<=permit.stat().st_mtime_ns
  assert row['prepared_pid']==launch['pid']and row['prepared_started_ticks']==launch['started_ticks']
  assert row['original_CPU_preload_proof_sha256']==sha(preload)
 files=[preload,permit,root/'attempt/launch.json']+([file]if file.exists()else[])
 result=dict(passed=True,retained_CPU_wait_observations=len(rows),all_wait_before_actual_GPU_permit=True,
  same_source_bound_CPU_process_preserved=True,not_failed_mask_import_or_GPU_retry=True,
  no_GPU_clock_or_budget_reset=True,input_sha256={str(p):sha(p)for p in files})
 write(root/'CPU_retained_preload_wait_audit.json',result);return result

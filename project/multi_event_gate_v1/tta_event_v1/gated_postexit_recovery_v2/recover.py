"""Only finish CPU audits after the actual complete normal GPU exit; no rollout."""
import inspect,json,time,types
from pathlib import Path
from .common import T,P,O,B,ROOT,read,mapped,sha,verify,write

def prior():
 a=read(B/'actual_host_wait_exit.json');assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 failure=read(B/'campaign_failure.json');assert failure['traceback'].endswith('RuntimeError: GPU occupied before our verified arrival or arrival order unverified; stopping only our job\n')
 spec=read(ROOT/'protocol.json');verify(spec);assert spec['dataset_split']=='TRAIN'and spec['cases_per_batch']==8
 normal=read(ROOT/'attempt/barrier/normal_child_exit.json');assert normal['normal_exit']and normal['exit_code']==0 and normal['wait_status']==0 and normal['proc_absent_after_wait']and not normal['cpu_only_fixture']and normal['protocol_identity']==spec['identity']
 complete=read(ROOT/'GPU_episodes_complete.json');assert complete['complete']and complete['protocol_identity']==spec['identity']and len(complete['cases'])==8
 rows=[json.loads(s)for s in(ROOT/'attempt/resources.jsonl').read_text().splitlines()];stopped=[x['stopped']for x in rows if'stopped'in x];assert stopped==['GPU occupied before our verified arrival or arrival order unverified; stopping only our job']
 assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids']and rows[-1]['cleanup_seconds_after_start']==1644.661
 telemetry=[x for x in rows if'seconds'in x];late=telemetry[-1];note=late['normal_child_exit_proof'];assert note['accepted']and note['exit_code']==0 and note['normal_child_reaped']and note['proc_absent_rechecked']and note['parent_alive_same_identity']
 assert late['gpu_pids']==[]and late['raw_foreign_pids']==[1033427]and late['normal_child_retired_pids']==[910680]
 foreign_time=late['gpu_arrival_order']['foreign_first_seen_seconds']['1033427']
 assert foreign_time>1600 and note['waited_monotonic_ns']==normal['waited_monotonic_ns']
 # Every late row after a verified retirement has no GPU owner. Before that,
 # original fresh-arrival and current server guards must have actually passed.
 drained=False
 for row in telemetry:
  if row.get('normal_child_exit_proof',{}).get('accepted'):drained=True
  if drained:assert not row['gpu_pids']
  else:assert not row['gpu_arrival_order']['stop_for_prior_or_unverified_occupant']and not row['raw_foreign_pids']
 assert drained and len(list(ROOT.glob('cases/*/*/result.json')))==36
 assert read(ROOT/'CPU_TOPP_workers_exit.json')['all_CPU_workers_normal_exit0']and not read(ROOT/'CPU_TOPP_workers_exit.json')['parent_CUDA_initialized']
 sem=read(ROOT/'semantic_exit.json');assert sem['exit_code']==0 and sem['actual_Popen_wait_return']
 return spec,rows,normal

def main():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 spec,rows,normal=prior()
 from .analysis import audit
 scientific=audit(spec,ROOT);assert scientific['passed']and scientific['actual_native_episodes']==32
 from native_render_overlap_v4.audit import validate
 assert validate(read(ROOT/'render_overlap_runtime.json'))
 for f in('guide_prefix_runtime.json','native_image_readback_runtime.json','CPU_static_heap_freeze.json'):assert read(ROOT/f)['passed']
 p=read(ROOT/'owner_profile_runtime.json');assert p['passed']and p['closed']and p['open_call_depth']==0
 assert 'OIDN Error:'not in(ROOT/'GPU_host.log').read_text(errors='replace')
 # Exact original host3.14 sum/observer resource audit, performed before CPU3.11 model replay.
 resource=read(O/'audits/CPU_governor180_resource_audit.json');verify(resource);assert resource['passed']
 raw=O/'audits/attempt/governor180_raw.jsonl'
 from native_semantic_host_audit_v1.audit import compute
 semantic=compute(ROOT);assert semantic['passed'];write(O/'audits/CPU_semantic_thread_budget_audit.json',semantic)
 session=next(ROOT.glob('sessions/*/native_storage.json'));storage=read(session)
 for name,expected in storage['static_source_metadata'].items():
  f=mapped(name);s=f.stat();assert s.st_size==expected['bytes']and s.st_mtime_ns==expected['mtime_ns']and s.st_mode==expected['mode']and sha(f)==expected['sha256']
 assert not torch.cuda.is_initialized();verify(spec)
 files=[B/'actual_host_wait_exit.json',B/'campaign_failure.json',ROOT/'protocol.json',ROOT/'GPU_episodes_complete.json',ROOT/'attempt/barrier/normal_child_exit.json',ROOT/'attempt/resources.jsonl',ROOT/'attempt/governor180.jsonl',ROOT/'CPU_closed_expert_task_release_audit.json',ROOT/'CPU_TOPP_workers_exit.json',ROOT/'semantic_exit.json',session]
 files+=list((O/'audits').rglob('*.json'))+[raw]
 value=dict(passed=True,complete=True,CPU_only_no_GPU_or_new_episodes=True,additional_optimizer_updates=0,original_host_exit_code=1,original_failure_preserved=True,actual_GPU_child_exit0_and_absent=True,actual_GPU_child=normal,actual_CPU_recovery_exit_pending=True,exact_36_whole_branches_replayed=True,original_current180_raw_and_server_decisions_replayed=True,stop_occurs_only_after_own_GPU_has_normally_exited=True,late_foreign_PID=1033427,old_GPU_time_including_all_attempts=4225.221,stage_GPU_budget=7200,groups=None,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')},input_sha256={str(f):sha(f)for f in files})
 write(O/'result.json',value);print('COMPLETE_NORMAL_GPU_POSTEXIT_CPU_AUDITS_PASS',flush=True)
if __name__=='__main__':main()

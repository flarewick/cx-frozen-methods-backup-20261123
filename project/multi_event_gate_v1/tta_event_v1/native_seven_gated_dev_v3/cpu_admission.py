"""Bounded CPU replanning only; never retry a process that received a GPU permit."""
from pathlib import Path
from .common import read,write

class CPUAdmissionRace(RuntimeError):pass

def occupied_before_permit(root,gpu,fleet):
 card=next(c for c in fleet['cards']if c['index']==gpu)
 if not card['pids']:return
 assert not (root/'attempt/barrier/permit.json').exists()
 resource=root/'attempt/resources.jsonl'
 assert not resource.exists()or resource.stat().st_size==0
 ready=read(root/'source_bound_CPU_preload_proof.json')
 assert ready['passed']and not ready['cuda_initialized']and ready['os_threads']==1
 assert ready['source_sha256']==read(root/'protocol.json')['source_sha256']
 write(root/'CPU_admission_race.json',dict(reason='selected_GPU_became_occupied_during_CPU_preparation',selected_gpu=gpu,fresh_inventory=fleet,no_GPU_permit=True,no_GPU_resource_samples=True,no_scientific_execution=True,CPU_preload_proof=ready))
 raise CPUAdmissionRace('Selected card changed before any GPU permit; CPU-only replan')

def assert_replan(root):
 row=read(root/'CPU_admission_race.json');assert row['no_GPU_permit']and row['no_GPU_resource_samples']and row['no_scientific_execution']
 assert not(root/'attempt/barrier/permit.json').exists()
 f=root/'attempt/resources.jsonl';assert no_real_GPU_samples(root)
 for name in ['CPU_admission_prepared_exit.json','semantic_exit.json']:
  exit=read(root/name);assert exit['actual_Popen_wait_return']and not Path('/proc',str(exit['pid'])).exists()
 assert not (root/'GPU_episodes_complete.json').exists()and not(root/'terminal_audit.json').exists()
 write(root/'CPU_only_replan_qualified.json',dict(passed=True,no_GPU_or_scientific_work=True,prepared_and_semantic_processes_actually_waited=True,no_GPU_clock_or_budget_reset=True,not_native_data_qualification=True))


def replan_if_other_empty(root,gpu,fleet):
 card=next(c for c in fleet['cards']if c['index']==gpu)
 if fleet['own_gpu_count']<3 and card['pids']and any(c['available']and c['index']!=gpu for c in fleet['cards']):
  occupied_before_permit(root,gpu,fleet)


def no_real_GPU_samples(root):
 import json
 p=root/'attempt/resources.jsonl'
 if not p.exists()or p.stat().st_size==0:return True
 rows=[json.loads(s)for s in p.read_text().splitlines()]
 return len(rows)==1 and set(rows[0])=={'cleanup_seconds_after_start','own_gpu_released','remaining_own_gpu_pids'}and rows[0]['own_gpu_released']and not rows[0]['remaining_own_gpu_pids']

def final_rule4_replan(root,gpu,error):
 assert not(root/'attempt/barrier/permit.json').exists()and no_real_GPU_samples(root)
 ready=read(root/'source_bound_CPU_preload_proof.json')
 assert ready['passed']and not ready['cuda_initialized']and ready['os_threads']==1
 assert ready['source_sha256']==read(root/'protocol.json')['source_sha256']
 assert transient_rule4_reason(error),'Unknown failure must not become a CPU admission retry'
 write(root/'CPU_admission_race.json',dict(reason='final_rule4_changed_before_GPU_permit',selected_gpu=gpu,actual_error=str(error),no_GPU_permit=True,no_GPU_resource_samples=True,no_scientific_execution=True,CPU_preload_proof=ready))
 raise CPUAdmissionRace('Final rule4 changed before GPU permit; original bounded CPU-only replan')from error

def historical_final_admission_failure():
 from .common import T,sha,mapped,verify
 base=T/'outputs/native_calibrated_five100_v22/official_007';actual=base/'actual_host_wait_exit.json';a=read(actual)
 assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 root=mapped(read(base/'execution_state.json')['root']);spec=read(root/'protocol.json');verify(spec)
 assert not(root/'attempt/barrier/permit.json').exists()and no_real_GPU_samples(root)
 ready=read(root/'source_bound_CPU_preload_proof.json');assert ready['passed']and not ready['cuda_initialized']and ready['source_sha256']==spec['source_sha256']
 assert 'Server rule4: existing own GPU4 below75% utilization and50% own memory'in read(root/'failure.json')['traceback']
 assert not list((root/'cases').glob('*/result.json'))and not(root/'GPU_episodes_complete.json').exists()
 import pathlib
 prepared=read(root/'attempt/launch.json')['pid'];assert not pathlib.Path('/proc',str(prepared)).exists()
 assert read(root/'semantic_exit.json')['actual_Popen_wait_return']and not pathlib.Path('/proc',str(read(root/'semantic_exit.json')['pid'])).exists()
 return{str(p):sha(p)for p in(actual,base/'execution_state.json',root/'protocol.json',root/'source_bound_CPU_preload_proof.json',root/'attempt/resources.jsonl',root/'failure.json',root/'semantic_exit.json',root/'attempt/launch.json')}


def transient_rule4_reason(error):
 text=str(error)
 return text in{'Selected GPU is not genuinely empty; preserve existing jobs','Selected GPU is not present in fresh inventory','Inconsistent account GPU count','All-hours account GPU limit is3; refusing another GPU','Selected GPU is not idle','All-hours user cap of3 GPUs','Rule8: existing GPUs need180s paired utilization AND VRAM thresholds','Fleet changed since180s observation; wait for fresh observer'}or text.startswith('Server rule4:')or text in{'No fresh conservative rule4 history','Selected GPU absent','Account GPU limit','Selected GPU is occupied','First GPU requires30s account-idle observation','Prior GPUs require120s history and lifetime own-memory>=55%; utilization is diagnostic only','Own GPU population changed since historical eligibility; await a fresh scheduled observation','Prior own GPU process retired during admission','Prior own GPU process identity changed','Selected GPU is absent','Account already uses three GPUs'}


def historical_three_CPU_races():
 from .common import T,sha,mapped,verify
 base=T/'outputs/native_calibrated_five100_v23/official_007';a=read(base/'actual_host_wait_exit.json')
 assert a['exit_code']==1 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 roots=list((base/'batches/000/runs').iterdir());assert len(roots)==3
 inputs={str(base/'actual_host_wait_exit.json'):sha(base/'actual_host_wait_exit.json')}
 for root in roots:
  spec=read(root/'protocol.json');verify(spec)
  assert read(root/'CPU_only_replan_qualified.json')['passed']and not(root/'attempt/barrier/permit.json').exists()and no_real_GPU_samples(root)
  for name in('CPU_admission_prepared_exit.json','semantic_exit.json'):
   a=read(root/name);assert a['actual_Popen_wait_return']and not Path('/proc',str(a['pid'])).exists()
  for name in('protocol.json','CPU_admission_race.json','CPU_only_replan_qualified.json','CPU_admission_prepared_exit.json','semantic_exit.json'):inputs[str(root/name)]=sha(root/name)
 return inputs

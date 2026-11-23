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
 f=root/'attempt/resources.jsonl';assert not f.exists()or f.stat().st_size==0
 for name in ['CPU_admission_prepared_exit.json','semantic_exit.json']:
  exit=read(root/name);assert exit['actual_Popen_wait_return']and not Path('/proc',str(exit['pid'])).exists()
 assert not (root/'GPU_episodes_complete.json').exists()and not(root/'terminal_audit.json').exists()
 write(root/'CPU_only_replan_qualified.json',dict(passed=True,no_GPU_or_scientific_work=True,prepared_and_semantic_processes_actually_waited=True,no_GPU_clock_or_budget_reset=True,not_native_data_qualification=True))

import json,sys,copy
from pathlib import Path
import torch
assert not torch.cuda.is_initialized()and not list(Path('/dev').glob('nvidia*'));torch.set_num_threads(1)
import benefit_followup_protocol_v2 as p,benefit_growth_runtime as runtime
runtime.install_sources()
from bridge241 import read,sha,verify_origin,comparable
from concurrent_queue import EpisodeQueue
reg=read(p.ROOT/'migration241/origin_G_registry.json');plan=read(p.ROOT/'validate_plan.json');total=0
for job in plan['jobs']:
 assert job['manifest']['deployment_shared6']['gpu_uuid']==reg['target_GPU_UUID']
 queue=EpisodeQueue(Path(job['run']),job['manifest'])
 if job['task']not in reg['jobs']:assert not queue.state['completed'];continue
 oldrun,old,q=verify_origin(reg['jobs'][job['task']]);assert comparable(old)==comparable(job['manifest']);receipt=read(Path(job['run'])/'resume_import241_receipt.json');total+=receipt['imported_whole_episodes']
 for seed,e in queue.state['completed'].items():
  if not e['process_session'].startswith('import241_'):continue
  original=q['completed'][seed];v=copy.deepcopy(e);v['process_session']=original['process_session'];v['evidence_sha256']=original['evidence_sha256'];assert v==original
 for mutation in [lambda m:m.update(task_config='demo_clean'),lambda m:m['pilot'].update(alpha=.6),lambda m:m.update(checkpoint_sha256='0'*64),lambda m:m['deployment_shared6'].update(compile_mode='default')]:
  bad=copy.deepcopy(job['manifest']);mutation(bad);assert comparable(old)!=comparable(bad)
assert total==reg['origin_whole_G']
p.atomic(p.ROOT/'migration241/CPU_verified.json',dict(passed=True,no_GPU=True,actual_imported_G=total,all_original_queues_validated=True,whole_results_only_session_and_evidence_paths_rebased=True,source_GPU0_not_relabelled=True,negative_scientific_changes_rejected=True,original_GPU3_2fc_metadata_and_bytes_retained=True,full_new_G_action_prefix_and_score_replay_still_required=True,helper_sha256=p.sha(__file__)))
print('241_CPU_IMPORT_VERIFIED',total,flush=True)

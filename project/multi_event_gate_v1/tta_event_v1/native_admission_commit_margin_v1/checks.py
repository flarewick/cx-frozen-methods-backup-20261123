"""Replay original recorded admission estimates; CPU fixtures never authorize CUDA."""
from pathlib import Path
import copy,json

def run():
 from .margin import require
 from native_policy_execution_v1.common import T,read
 o=T/'outputs/native_onpolicy_arm_collection_v16/val_000';identity=read(o/'rule4_monitor_identity.json');root=Path(identity['root'])
 if not root.exists():
  from native_onpolicy_recovery_v3.common import mapped
  root=mapped(root)
 observed=[json.loads(x)for x in(root/'observations.jsonl').read_text().splitlines()]
 boundary=[]
 for row in observed:
  prior=[c for c in row['fleet']['cards']if c['own_pids']and c['index']!=5]
  if prior and all(row['estimates'][c['uuid']]['eligible']for c in prior):boundary.append(row)
 assert boundary,'Recorded original admission threshold was not reached'
 first=boundary[0];selected=next(c['uuid']for c in first['fleet']['cards']if c['index']==5)
 proof=dict(passed=True,latest_observation=first,estimates=first['estimates'])
 rejected=False
 try:require(proof,selected)
 except RuntimeError:rejected=True
 assert rejected,'Actual borderline case must keep waiting on CPU'
 # A later credible observation with the same real memory rate eventually
 # passes; deliberately synthetic extension is only a CPU arithmetic fixture.
 future=copy.deepcopy(proof)
 for v in future['estimates'].values():
  v['observed_memory_time_integral']+=.75*120
  v['process_age_seconds']+=120
 assert require(future,selected)is future
 absent=copy.deepcopy(proof);absent['latest_observation']['fleet']['cards']=[dict(c,own_pids=[])for c in absent['latest_observation']['fleet']['cards']]
 assert require(absent,selected)is absent
 # Final original guard call remains byte exact; a population change still
 # must be refused there, and the added margin grants no admission itself.
 source=(T/'native_onpolicy_arm_collection_v17/host.py').read_text()
 original=(T/'native_onpolicy_arm_collection_v16/host.py').read_text().replace('native_onpolicy_arm_collection_v16','native_onpolicy_arm_collection_v17')
 restored=source.replace("  from native_admission_commit_margin_v1.margin import require as require_commit_margin\n",'').replace(";first_admission=require_rule4(info['uuid']);require_commit_margin(first_admission,info['uuid']);admission.check_admission",";require_rule4(info['uuid']);admission.check_admission")
 assert restored==original
 return dict(passed=True,CPU_only=True,actual_ARM16_borderline_history_refused=True,
             synthetic_stable_estimate_can_pass=True,first_card_unchanged=True,
             final_original_strict_require_and_guard_byte_exact=True,
             original_observation_file=str(root/'observations.jsonl'),GPU_work=False)

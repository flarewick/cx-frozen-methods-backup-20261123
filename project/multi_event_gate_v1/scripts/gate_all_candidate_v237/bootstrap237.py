"""Import a certified historical expert prefix under an explicitly new protocol."""
import json,copy,sys
from pathlib import Path
import benefit_followup_protocol_v2 as p
p.prepare();cohort=int(__import__('os').environ.get('CX_GATE_TRANSFER_COHORT','0'));old=p.HOME/('candidate_veto_cup_bowls_v200_'+str(cohort))/'dev0'
reg=json.loads((p.ROOT.parent/'preregistration237.json').read_text())
for f,h in reg['controls_origin_sha256'].items():
 file=Path(f)
 if not file.exists():file=p.R/file.relative_to('/data/qiufw/cx_migration/cx_new_direction_guide_TTA')
 assert p.sha(file)==h
src=old/'stream_prepare/parallel_screen_completed.json';v=json.loads(src.read_text());original_protocol=v['protocol_sha256'];v['protocol_sha256']=p.sha(p.ROOT/'protocol.json')
assert all(2<=r['qualified']<20 for r in v['tasks'].values())
target=p.ROOT/'stream_prepare/parallel_screen_completed.json';target.parent.mkdir(exist_ok=True)
if target.exists():assert json.loads(target.read_text())==v
else:p.atomic(target,v)
r=json.loads((old/'finite_bootstrap_receipt_v6.json').read_text());r.update(snapshot_sha256=p.sha(target),imported_certified_original_expert_prefix=True,actual_qualified={t:x['qualified']for t,x in v['tasks'].items()},source_receipt=str(old/'finite_bootstrap_receipt_v6.json'),source_receipt_sha256=p.sha(old/'finite_bootstrap_receipt_v6.json'),source_snapshot=str(src),source_snapshot_sha256=p.sha(src),original_protocol_sha256=original_protocol,new_protocol_sha256=v['protocol_sha256'],no_new_GPU_expert_collection_claim=True)
p.atomic(p.ROOT/'finite_bootstrap_receipt_v6.json',r)
print('237_EXPERT_PREFIX_IMPORTED_WITH_ORIGIN',flush=True)

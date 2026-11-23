"""Actual saved once-base queries: original controller and zero-update equality."""
from pathlib import Path
import copy,json,hashlib
import numpy as np
from .compose import EventComposer,ARM
from cx_tta.multi_event import EventConfig,EventDetector,LocalController,apply_arm_candidate
from cx_tta.empirical_onset_gate import candidate_packet
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 spec=read(T/'outputs/native_expert_full_cache_v8/protocol.json');config=EventConfig(**spec['event_config'])
 index=read(T/'outputs/native_expert_full_feedback_v1/index.json');cases=[c for c in index['cases']if c['old_pilot_reused']];assert len(cases)==4
 queries=active=0;records=[]
 for case in cases:
  detector=EventDetector(config);controller=LocalController(config);full=EventComposer(config);zero=EventComposer(config);changed=EventComposer(config)
  for row in case['queries']:
   p=row['policy'];assert sha(p['policy_file'])==p['policy_sha256']
   with np.load(p['policy_file'],allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
   base=data['base'];raw=data['raw_delta'];legal={k[len('legal_'):]:v for k,v in data.items()if k.startswith('legal_')}
   packet=candidate_packet(legal,raw);event,evidence=detector.step(row['query'],packet)
   decision=controller.step(row['query'],event,packet,lambda p,c:dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.),enabled=True)
   expected,delta=apply_arm_candidate(base,raw,decision['active'])
   a,d,m=full.step(base,raw,legal);z,zd,zm=zero.step(base,raw,legal,adapted_ARM=raw[:,ARM].copy())
   assert a.tobytes()==expected.tobytes()==z.tobytes()and d.tobytes()==delta.tobytes()==zd.tobytes()
   assert m['decision']==zm['decision']==decision and m['event']==event and m['evidence']==evidence
   fake=-raw[:,ARM].copy();b,bd,bm=changed.step(base,raw,legal,adapted_ARM=fake)
   assert bm['decision']==decision and np.array_equal(b[:,[6,13]],base[:,[6,13]])
   if not decision['active']:assert b.tobytes()==expected.tobytes()
   active+=bool(decision['active']);queries+=1
  records.append(dict(case_id=case['case_id'],queries=len(case['queries']),original_full_and_zero_arrays_byte_exact=True))
  full.reset();assert full.query==0
 assert queries==36 and active>0
 report=dict(passed=True,real_once_base_queries=queries,original_active_queries=active,records=records,original_detector_controller_and_zero_update_byte_exact=True,adaptation_never_changes_selector=True,inactive_and_gripper_actions_exact=True,SAM2_forward_calls=0,VLA_forward_calls=0,policy_rollouts=0,qualified_response_session_still_required=True,TTA_utility_proven=False,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')})
 out=T/'outputs/native_response_action_v1';out.mkdir(exist_ok=True)
 with(out/'CPU_composition_checks.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('FROZEN_EVENT_RESPONSE_COMPOSITION_PASS',queries,active,flush=True)
if __name__=='__main__':main()

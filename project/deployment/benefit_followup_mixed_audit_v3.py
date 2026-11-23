"""Strict original collection audit plus sealed streaming membership provenance."""
import hashlib,json
from pathlib import Path
import benefit_followup_protocol_v2 as p
import benefit_growth_runtime as runtime
import benefit_followup_mixed_core_v3 as core
from benefit_followup_stream_v3 import atomic,canonical,StreamingEligibility

def main():
 runtime.install_sources()
 from shared6_eval_backend import sources
 q=json.loads((p.ROOT/'qualified.json').read_text());ledger=core.state()
 assert q['complete']and q['policy_episodes']==0 and ledger.ledger.complete()and core.validate_snapshot(core.current_qualification())
 assert q['tasks']==ledger.ledger.tasks and q['eligibility_event_chain_head']==ledger.head and q['source_sha256']==sources()and q['qualification_stream_scope_sha256']==p.sha(core.scope_path())
 plan=json.loads((p.ROOT/'collect_plan.json').read_text())
 for j in plan['jobs']:
  m=j['manifest'];assert m['pilot']['qualification_stream_scope_sha256']==p.sha(core.scope_path())and m['source_sha256'][str(core.scope_path())]==p.sha(core.scope_path())
  assert str(p.ROOT/'qualified.json')not in m['source_sha256']and str(p.ROOT/core.EVENTS)not in m['source_sha256']
 checkpoints={}
 for file in p.ROOT.glob('control_*/cpu_finalize_intent.json'):
  data=json.loads(file.read_text())
  if data.get('source_sha256')!=sources():continue
  cp=file.parent/'mixed_expert_checkpoint.json';v=json.loads(cp.read_text())
  assert data['mixed_expert_checkpoint_sha256']==p.sha(cp)and data['qualification_stream_scope_sha256']==p.sha(core.scope_path())
  # Earlier checkpoints bind earlier journal prefixes; the full final journal
  # may have legitimately grown. Their exact tips must match a strict prefix.
  matches=[]
  for i in range(len(ledger.events)+1):
   prefix=ledger.events[:i];head=prefix[-1]['event_hash']if prefix else ledger.snapshot_sha256
   saved=dict(schema='append_only_expert_eligibility_v3',starts=ledger.starts,target=ledger.target,maximum=ledger.maximum,protocol_sha256=ledger.protocol_sha256,initial_snapshot_sha256=ledger.snapshot_sha256,head=head,events=prefix)
   if hashlib.sha256(canonical(saved)+bytes([10])).hexdigest()==v['event_journal_sha256']:matches.append(i)
  assert len(matches)==1,'Checkpoint is not an exact prefix of original expert journal'
  replay=StreamingEligibility(p.TASKS,ledger.snapshot,ledger.protocol_sha256)
  for event in ledger.events[:matches[0]]:replay.submit(event['task'],event['row'],origin=event['origin'])
  assert v['expert_counts']=={t:r['qualified']for t,r in replay.ledger.tasks.items()}and v['complete']==replay.ledger.complete()
  assert v['all_expert_threads_finished']and v['source_sha256']==sources()and v['qualification_stream_scope_sha256']==p.sha(core.scope_path())
  for name,h in v['expert_runtime_audits_sha256'].items():assert p.sha(Path(name))==h
  receipt=json.loads((file.parent/'cpu_finalize_receipt.json').read_text());assert receipt['passed']and receipt['intent_sha256']==p.sha(file)
  checkpoints[str(cp)]=p.sha(cp)
 assert checkpoints
 from benefit_followup_audit_v2 import collect
 report=collect();assert report['passed']and report['complete']
 proof=dict(passed=True,complete=True,source_sha256=sources(),qualification_stream_scope_sha256=p.sha(core.scope_path()),eligibility_event_chain_head=ledger.head,initial_expert_snapshot_sha256=ledger.snapshot_sha256,new_expert_decisions=len(ledger.events),original_expert_prefix_unchanged=True,membership_independent_of_policy_outcomes=True,all88_original_episode_and_zero_tolerance_pair_checks_passed=True,collection_audit_sha256=p.sha(p.ROOT/'collection_audit.json'),checkpoint_sha256=checkpoints,auditor_sha256=p.sha(__file__))
 atomic(p.ROOT/'mixed_membership_final_audit.json',proof);print('MIXED_MEMBERSHIP_AND_ORIGINAL_88_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()

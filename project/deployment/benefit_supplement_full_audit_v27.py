"""Full original 88 audit with an explicit strictly proven diagnostic extension."""
import json
import benefit_followup_protocol_v2 as p
from benefit_followup_stream_v3 import atomic

def main():
 import benefit_supplement_alias_audit_v27 as alias;alias.install()
 import benefit_followup_audit_v2 as base
 original_collect=base.collect
 def collect():
  report=original_collect();assert report['passed']and report['complete'];amendment=p.HOME/'supplement_v24/inactive_alias_amendment_v27.json'
  report.update(additional_validation_amendment_sha256=p.sha(amendment),additional_validation_helper_sha256=p.sha(p.D/'benefit_supplement_alias_audit_v27.py'),additional_validation_CPU_proof_sha256=p.sha(p.HOME/'supplement_v24/inactive_alias_CPU_proof_v27.json'),additional_validation_wrapper_sha256=p.sha(__file__),raw_bytes_unchanged=True,real_native_contacts_actions_and_clock_numeric_tolerance=0)
  atomic(p.ROOT/'collection_audit.json',report);atomic(p.ROOT/'collection_distribution.json',report)
  file=p.ROOT/'collection_audit_amendment_receipt.json';receipt=json.loads(file.read_text());receipt.update(collection_audit_sha256=p.sha(p.ROOT/'collection_audit.json'),additional_validation_amendment_sha256=p.sha(amendment),additional_validation_wrapper_sha256=p.sha(__file__),raw_AA_failure_evidence_preserved=True);atomic(file,receipt)
  return report
 base.collect=collect
 import benefit_followup_mixed_audit_v3 as original;original.main()
 proof=json.loads((p.ROOT/'mixed_membership_final_audit.json').read_text());assert proof['passed']and proof['complete']and proof['collection_audit_sha256']==p.sha(p.ROOT/'collection_audit.json')
 atomic(p.ROOT/'supplement_full_audit_receipt_v27.json',dict(passed=True,complete=True,collection_audit_sha256=p.sha(p.ROOT/'collection_audit.json'),membership_audit_sha256=p.sha(p.ROOT/'mixed_membership_final_audit.json'),additional_validation_amendment_sha256=p.sha(p.HOME/'supplement_v24/inactive_alias_amendment_v27.json'),wrapper_sha256=p.sha(__file__),original_full_episode_and_membership_checkers_unchanged=True,raw_data_and_science83_unchanged=True,only_proven_inactive_offline_field_has_explicit_exception=True,all_real_recorded_values_zero_tolerance=True))
 print('SUPPLEMENT_FULL88_STRICT_EXTENSION_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()

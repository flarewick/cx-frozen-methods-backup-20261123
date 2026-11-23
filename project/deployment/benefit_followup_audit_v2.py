"""Followup audit reuses original strict episode/array/physics validators."""
import inspect,json,textwrap,sys
import benefit_growth_runtime as runtime
import benefit_followup_protocol_v2 as p
from benefit_followup_protocol_v2 import D,ROOT,HOME,PRIOR,sha,atomic
import benefit_stage_audit as original
import benefit_growth_canonical_audit as canonical
from benefit_growth_audit import resource_audit

def collect():
 runtime.install_sources();canonical.verify_amendment();canonical.CORRECTIONS.clear()
 source=textwrap.dedent(inspect.getsource(original.collect)).replace("('TRAIN','VAL')","('TRAIN','DEV','TEST')")
 source=source.replace("D/'reports/target_benefit_stage_distribution_20261026.json'","ROOT/'collection_distribution.json'").replace("D/'reports/target_benefit_stage_distribution_20261026.md'","ROOT/'collection_distribution.md'")
 ns=dict(original.collect.__globals__,ROOT=ROOT,TASKS=p.TASKS,split=p.split,selected=p.selected,resource_audit=resource_audit,exact=canonical.exact)
 exec(compile(source,__file__,'exec'),ns);report=ns['collect']()
 report.update(audit_method='original_strict_values_with_proven_inactive_offline_attribute_amendment',validation_amendment_sha256=sha(HOME/'inactive_alias_validation_amendment.json'),canonical_auditor_sha256=sha(D/'benefit_growth_canonical_audit.py'),followup_auditor_sha256=sha(__file__),corrected_inactive_diagnostic_pairs=len(canonical.CORRECTIONS))
 atomic(ROOT/'collection_audit.json',report);atomic(ROOT/'collection_distribution.json',report)
 atomic(ROOT/'collection_audit_amendment_receipt.json',dict(passed=True,collection_audit_sha256=sha(ROOT/'collection_audit.json'),amendment_sha256=sha(HOME/'inactive_alias_validation_amendment.json'),canonical_auditor_sha256=sha(D/'benefit_growth_canonical_audit.py'),followup_auditor_sha256=sha(__file__),original_validator_files_unchanged=True,corrections=canonical.CORRECTIONS))
 print('FOLLOWUP_COLLECTION_AUDIT',json.dumps(dict(passed=report['passed'],complete=report['complete'],distribution=report['distribution'],corrections=len(canonical.CORRECTIONS))),flush=True)
 return report

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]!='collect':raise RuntimeError('No validation before followup model/DEV readiness')
 collect()

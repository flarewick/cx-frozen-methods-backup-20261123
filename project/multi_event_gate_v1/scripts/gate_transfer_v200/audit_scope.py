"""Same full scientific auditor and actual-attempt resource scope, for new tasks."""
import importlib.util,json,sys
from pathlib import Path
import torch
import benefit_followup_protocol_v2 as p
HERE=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized()
 spec=importlib.util.spec_from_file_location('transfer_resource_scope',HERE.parent/'gate_extension_recovery_v192/scope_v196.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
 if sys.argv[1:] == ['verify']:
  fixture=p.HOME/'candidate_veto_event_full_GPU3_v154/dev0';result=s.resource_for(fixture);assert result['passed']and len(result['sessions'])==11
  p.atomic(p.ROOT/'transfer_resource_scope_CPU.json',dict(passed=True,CPU_only=True,fixture_resource_only=True,fixture_root=str(fixture),helper_sha256=p.sha(HERE/'audit_scope.py'),resource_helper_sha256=p.sha(Path(s.__file__)),actual_all11_historical_attempts_verified=True));return
 assert sys.argv[1:]in (['controls'],['validate'])
 proof=json.loads((p.ROOT/'transfer_resource_scope_CPU.json').read_text());assert proof['passed']and proof['helper_sha256']==p.sha(HERE/'audit_scope.py')and proof['resource_helper_sha256']==p.sha(Path(s.__file__))
 import benefit_growth_audit as g,benefit_followup_audit_v2 as f
 g.resource_audit=f.resource_audit=lambda:s.resource_for(p.ROOT)
 import benefit_event_full_audit_v134 as a
 getattr(a,sys.argv[1])()
 target=p.ROOT/'collection_audit.json'if sys.argv[1]=='controls'else p.ROOT.parent/'final.json';v=json.loads(target.read_text());assert v['passed']and v['complete']and v['resource_audit']['actual_cohort_root']==str(p.ROOT)
 p.atomic(p.ROOT/('transfer_'+sys.argv[1]+'_scope_receipt.json'),dict(passed=True,report_sha256=p.sha(target),actual_attempts=len(v['resource_audit']['sessions']),frozen_gate_and_event_full_unchanged=True,helper_sha256=p.sha(__file__)))
if __name__=='__main__':main()

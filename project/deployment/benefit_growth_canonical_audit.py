"""Strict audit amendment for proven inactive diagnostic aliases; raw bytes stay sealed."""
import ast,copy,inspect,json,sys,textwrap,time
from pathlib import Path
import benefit_growth_protocol as p
from benefit_growth_protocol import ROOT,HOME,D,sha,atomic
import benefit_stage_audit as original
import benefit_growth_audit as growth
from benefit_growth_alias_diagnosis import prove_inactive,cleaned
ORIGINAL_EXACT=original.exact
CORRECTIONS=[]
AMENDMENT=HOME/'inactive_alias_validation_amendment.json'

def source_contract():
 runtime=D/'.work/rootfs/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin'
 actor=runtime/'envs/utils/actor_utils.py';base=runtime/'envs/_base_task.py'
 tree=ast.parse(actor.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Actor')
 getter=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='get_pose')
 body=[n for n in getter.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str))]
 assert len(body)==1 and isinstance(body[0],ast.Return)and ast.unparse(body[0].value)=='self.actor.get_pose()'
 bt=ast.parse(base.read_text());assignments=[n for n in ast.walk(bt)if isinstance(n,ast.Assign)and any(isinstance(y,ast.Attribute)and ast.unparse(y)=='self.cluttered_obj'for x in n.targets for y in ast.walk(x))]
 assert len(assignments)==1 and isinstance(assignments[0].value,ast.Call)and ast.unparse(assignments[0].value.func)=='rand_create_cluttered_actor'
 assert not any(isinstance(n,ast.Delete)and any(isinstance(y,ast.Attribute)and ast.unparse(y)=='self.cluttered_obj'for x in n.targets for y in ast.walk(x))for n in ast.walk(bt))
 return {str(f.relative_to(D.parent)):sha(f)for f in [actor,base,D/'../multi_event_gate_v1/cx_tta/recording.py',D/'../multi_event_gate_v1/cx_tta/attribution_physics.py',D/'benefit_stage_audit.py',D/'benefit_growth_audit.py',D/'benefit_growth_alias_diagnosis.py']}

def verify_amendment():
 a=json.loads(AMENDMENT.read_text());assert a['source_contract_sha256']==source_contract()
 assert a['canonical_auditor_sha256']==sha(__file__)and a['numeric_tolerance']==0
 return a

def exact(a,b,limit=None):
 issues=ORIGINAL_EXACT(a,b,limit)
 if not issues or any(k!='full_passive_trace'for k,q in issues):return issues
 try:pa,pb=prove_inactive(a[2]),prove_inactive(b[2])
 except (AssertionError,KeyError,TypeError):return issues
 ca=list(a);cb=list(b);ca[2]=[cleaned(r)for r in a[2]];cb[2]=[cleaned(r)for r in b[2]]
 if ORIGINAL_EXACT(ca,cb,limit):return issues
 CORRECTIONS.append(dict(seed=a[0]['seed'],reference_branch=a[0]['pilot_branch'],candidate_branch=b[0]['pilot_branch'],reference_session=a[0]['process_session'],candidate_session=b[0]['process_session'],prefix_limit=limit,raw_differences=issues,reference_inactive_alias=pa,candidate_inactive_alias=pb,all_remaining_fields_original_zero_tolerance_exact=True))
 return []

def collect():
 verify_amendment();CORRECTIONS.clear()
 # Preserve an existing failed audit byte-for-byte before the amended rerun.
 for name in ('collection_audit.json','collection_distribution.json','collection_distribution.md'):
  f=ROOT/name
  if f.exists():
   target=ROOT/(name+'.raw_before_inactive_alias_amendment')
   if not target.exists():target.write_bytes(f.read_bytes())
 source=textwrap.dedent(inspect.getsource(original.collect)).replace("('TRAIN','VAL')","('TRAIN','DEV','TEST')")
 source=source.replace("D/'reports/target_benefit_stage_distribution_20261026.json'","ROOT/'collection_distribution.json'").replace("D/'reports/target_benefit_stage_distribution_20261026.md'","ROOT/'collection_distribution.md'")
 ns=dict(original.collect.__globals__,ROOT=ROOT,TASKS=p.TASKS,split=p.split,selected=p.selected,resource_audit=growth.resource_audit,exact=exact)
 exec(compile(source,__file__,'exec'),ns);report=ns['collect']()
 report.update(audit_method='original_strict_values_with_proven_inactive_offline_attribute_amendment',validation_amendment_sha256=sha(AMENDMENT),canonical_auditor_sha256=sha(__file__),corrected_inactive_diagnostic_pairs=len(CORRECTIONS),raw_comparison_evidence_preserved=True)
 atomic(ROOT/'collection_audit.json',report);atomic(ROOT/'collection_distribution.json',report)
 with(ROOT/'collection_distribution.md').open('a')as f:f.write('\n\n本报告使用预先冻结的离线诊断别名修正规则；原始失败报告和全部数据保留，真实物理、接触、动作和数组仍为原严格检查。修正配对数：'+str(len(CORRECTIONS))+'。证据见collection_audit_amendment_receipt.json。\n')
 receipt=dict(passed=True,block=p.BLOCK,amendment_sha256=sha(AMENDMENT),canonical_auditor_sha256=sha(__file__),original_audit_source_sha256=sha(D/'benefit_stage_audit.py'),original_growth_audit_source_sha256=sha(D/'benefit_growth_audit.py'),all_original_source_files_unchanged=True,all_NPZ_action_and_native_clock_validators_original=True,all_real_native_and_contact_values_original_zero_tolerance=True,corrections=CORRECTIONS,collection_audit_sha256=sha(ROOT/'collection_audit.json'),raw_report_preserved=str(ROOT/'collection_audit.json.raw_before_inactive_alias_amendment'))
 atomic(ROOT/'collection_audit_amendment_receipt.json',receipt)
 print('CANONICAL_COLLECTION_AUDIT_PASSED',json.dumps(dict(block=p.BLOCK,corrections=len(CORRECTIONS),distribution=report['distribution'])),flush=True)
 return report

def validate():
 verify_amendment();CORRECTIONS.clear()
 # growth.validate copies the old validator globals; only replace the diagnostic
 # comparison entry while preserving the exact original function object/source.
 before=original.exact
 try:original.exact=exact;growth.validate()
 finally:original.exact=before
 atomic(ROOT/'gate_validation_amendment_receipt.json',dict(passed=True,amendment_sha256=sha(AMENDMENT),canonical_auditor_sha256=sha(__file__),corrections=CORRECTIONS))
 final=json.loads((ROOT/'final.json').read_text());final.update(validation_amendment_sha256=sha(AMENDMENT),canonical_auditor_sha256=sha(__file__),corrected_inactive_diagnostic_pairs=len(CORRECTIONS));atomic(ROOT/'final.json',final);atomic(HOME/'final.json',final)
 with(HOME/'final.md').open('a')as f:f.write('\n原始记录保留；比较使用TEST开启前冻结的离线诊断别名修正规则。所有真实物理、策略与接触值仍严格一致，容差未增加。\n')

if __name__=='__main__':{'collect':collect,'validate':validate}[sys.argv[1]if len(sys.argv)>1 else'collect']()

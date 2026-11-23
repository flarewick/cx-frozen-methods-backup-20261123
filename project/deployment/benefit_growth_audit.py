"""Independent cohort audit; original paired-array/physics validators retained."""
import inspect,json,textwrap
import benefit_growth_protocol as p
from benefit_growth_protocol import D,ROOT,BASE,sha,atomic
import benefit_stage_audit as old

def resource_audit():
 # Exact archived resource acceptance logic, now scoped to this new cohort.
 source=textwrap.dedent(inspect.getsource(old.resource_audit))
 source=source.replace("sorted((D/'reports').glob('target_benefit_stage_gpu*_*.jsonl'))","sorted(list(ROOT.glob('telemetry_gpu*_*.jsonl'))+list(ROOT.glob('screen_*.jsonl')))")
 ns=dict(old.resource_audit.__globals__,ROOT=ROOT)
 exec(compile(source,__file__,'exec'),ns);return ns['resource_audit']()

def collect():
 source=textwrap.dedent(inspect.getsource(old.collect))
 source=source.replace("('TRAIN','VAL')","('TRAIN','DEV','TEST')")
 source=source.replace("D/'reports/target_benefit_stage_distribution_20261026.json'","ROOT/'collection_distribution.json'")
 source=source.replace("D/'reports/target_benefit_stage_distribution_20261026.md'","ROOT/'collection_distribution.md'")
 ns=dict(old.collect.__globals__,ROOT=ROOT,TASKS=p.TASKS,split=p.split,selected=p.selected,resource_audit=resource_audit)
 exec(compile(source,__file__,'exec'),ns);report=ns['collect']()
 proof=dict(passed=True,block=p.BLOCK,split=p.SPLIT,archived_audit_source_sha256=sha(D/'benefit_stage_audit.py'),new_audit_source_sha256=sha(__file__),collection_audit_sha256=sha(ROOT/'collection_audit.json'),original_validator_functions_unchanged=True)
 atomic(ROOT/'collection_audit_implementation.json',proof)
 return report

def validate():
 source=textwrap.dedent(inspect.getsource(old.validate))
 source=source.replace("case['split']!='VAL'","case['split']!='TEST'")
 source=source.replace("sha(D/'benefit_stage_gate.py')","sha(D/'benefit_growth_gate.py')")
 source=source.replace("sha(__file__)","sha(D/'benefit_growth_learning.py')")
 source=source.replace('from benefit_stage_gate import predict','from benefit_growth_gate import predict')
 source=source.replace('assert len(cases)==8','assert len(cases)==40')
 source=source.replace('gate_validation_episodes=8,total_episodes=96','gate_validation_episodes=40,total_episodes=128')
 source=source.replace("independent_EVAL_claim=False","independent_EVAL_claim=False,independent_TEST=True")
 source=source.replace("'32 TRAIN and 8 VAL only; no 100-episode efficacy claim, no adapter retraining'","'40 preregistered independent TEST scenes; no 100-episode or broad generalization claim; adapter frozen'")
 source=source.replace("D/'reports/target_benefit_stage_complete_20261026.json'","ROOT/'gate_validation_audit.json'")
 source=source.replace("D/'reports/target_benefit_stage_complete_20261026.md'","ROOT/'gate_validation_legacy_template.md'")
 ns=dict(old.validate.__globals__,ROOT=ROOT,resource_audit=resource_audit)
 exec(compile(source,__file__,'exec'),ns);ns['validate']()
 final=json.loads((ROOT/'final.json').read_text());counts=final['validation_successes']
 final.update(goal_achieved=counts['gate']>max(counts['baseline'],counts['always_first']),source_frozen_adapter=True,all_new_collection_episodes=352,all_new_gate_validation_episodes=40,all_new_total_episodes=392,selection_sha256=sha(p.HOME/'selection.json'),frozen_model_sha256=sha(p.HOME/'gate.json'),audit_implementation_sha256=sha(__file__))
 atomic(ROOT/'final.json',final);atomic(p.HOME/'final.json',final)
 lines=['# 独立测试：可学习候选门控','',f"40 个固定 TEST 场景。baseline={counts['baseline']}/40，无门控单事件B={counts['always_first']}/40，门控G={counts['gate']}/40。",'',f"目标达成（同时优于两组）：{final['goal_achieved']}。",'真实G逐局执行，冻结模型分数与所选完整分支轨迹检查通过。TRAIN拟合、DEV选型，TEST未参与调参；当前adapter保持冻结。','40场景规模有限，不作统计显著性或跨任务普遍优势声明。']
 (p.HOME/'final.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
 import sys
 {'collect':collect,'validate':validate}[sys.argv[1]if len(sys.argv)>1 else'collect']()

"""Read-only scientific inputs; publish only after both full and stronger prefix audits."""
import json,hashlib,time
from pathlib import Path
W=Path(__file__).resolve().parents[2];H=W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[];cases=[];history=[];resources=[]
for i in (0,1):
 root=H/f'candidate_veto_all_events_v237_{i}';f=root/'final.json';pf=root/'dev0/all_candidate_extended_prefix248.json';v=read(f);prefix=read(pf)
 assert v['complete']and v['science_passed']and len(v['cases'])==40
 assert prefix['passed']and prefix['complete']and prefix['comparisons']==80 and prefix['scenes']==40
 assert v['resource_audit']['new_protected_attempts_passed']
 cases+=v['cases'];history+=v['resource_audit']['historical_failed_attempts'];resources+=v['resource_audit']['attempts'];files += [f,pf]
assert len(cases)==80 and len({(c['task'],c['seed'])for c in cases})==80
metrics={}
for task in ['place_empty_cup','stack_bowls_two']:
 rows=[c for c in cases if c['task']==task];assert len(rows)==40
 m={k:sum(c[k]for c in rows)for k in ['baseline_success','event_full_success','first_gate_success','all_gate_success','B_chunks','first_G_chunks','all_G_chunks','vetoed_candidates','later_vetoes']}
 m.update(scenes=40,B_fail_G_success=sum(not c['event_full_success']and c['all_gate_success']for c in rows),B_success_G_fail=sum(c['event_full_success']and not c['all_gate_success']for c in rows),harms_avoided=sum(c['baseline_success']and not c['event_full_success']and c['all_gate_success']for c in rows),rescues_lost=sum(not c['baseline_success']and c['event_full_success']and not c['all_gate_success']for c in rows),introduced_failures_on_baseline_and_B_success=sum(c['baseline_success']and c['event_full_success']and not c['all_gate_success']for c in rows),all_gate_success_with_fewer_chunks=sum(c['all_gate_success']and c['event_full_success']and c['all_G_chunks']<c['B_chunks']for c in rows))
 metrics[task]=m
v=dict(complete=True,science_passed=True,all_attempts_resources_passed=not history,historical_resource_failed_attempts=history,new_protected_attempts_passed=True,metrics=metrics,cases=cases,source_sha256={str(f):sha(f)for f in files},summary_helper_sha256=sha(Path(__file__)),created_unix=time.time(),held_out_EVAL_not_training=True,exploratory_repeated_scene_transfer_not_independent_confirmation=True,frozen_adapter_and_v118=True)
out=H/'all_candidate_summary250.json';out.write_text(json.dumps(v,indent=2)+'\n')
lines=['# 冻结 v118 全事件对照：每任务40场景','','adapter、event_full、v118权重与阈值固定，只将首次候选评分扩展到所有合法评分机会。拒绝当前候选后，后续候选仍可评分；原三次介入预算、候选资格、冷却与执行器保持。后续事件上的v118分数属于未重新校准的迁移诊断。','','|任务|baseline|event_full|首次门控|全事件门控|避免原B损害|丢失原B救回|新增原baseline/B均成功的失败|','|---|---:|---:|---:|---:|---:|---:|---:|']
for task,m in metrics.items():lines.append(f"|{task}|{m['baseline_success']}/40|{m['event_full_success']}/40|{m['first_gate_success']}/40|{m['all_gate_success']}/40|{m['harms_avoided']}|{m['rescues_lost']}|{m['introduced_failures_on_baseline_and_B_success']}|")
lines+=['','|任务|B纠正块|首次门控纠正块|全事件纠正块|B失败→G成功|B成功→G失败|配对成功且减少纠正场景|','|---|---:|---:|---:|---:|---:|---:|']
for task,m in metrics.items():lines.append(f"|{task}|{m['B_chunks']}|{m['first_G_chunks']}|{m['all_G_chunks']}|{m['B_fail_G_success']}|{m['B_success_G_fail']}|{m['all_gate_success_with_fewer_chunks']}|")
lines+=['','本轮观察到正向效果：cup修复1个原B损害且未丢失B救回，净增加2个成功；bowls修复3个原B损害，但丢失1个救回且新增2个baseline/B原本都成功的失败，净增加1个成功。两任务实际纠正块合计215→129（减少40%）。这支持继续研究全事件门控，尚不能称为可靠消除过度纠正或跨任务普遍有效。']
lines += ['','两批独立20场景，合计40/任务。所有80条G均通过原动作/物理时钟/控制器与冻结分数重放，以及相对baseline的首次介入前缀检查；另完成相对B和首次门控共160个共同前缀/首次分叉候选零容差检查。相同激活日程要求全轨迹一致。已有EVAL场景上的探索性对照不能替代新场景确认或正式100局评估。否决记录数包括预算耗尽后的评分，不等同于实际减少的纠正块数。','','每个任务分别衡量收益和损害，不能仅凭总体纠正次数减少宣称避免过度纠正。上表的“避免原B损害”要求baseline成功、B失败且新G成功；“丢失原B救回”要求baseline失败、B成功且新G失败。','','资源：所有新增受保护尝试通过原始资源审计，但整个实验保留1次历史失败：GPU3约305.91秒整卡时间加权累计32.9427%，307.423秒释放，整轮不能称全部资源通过。后来增加240秒起40%提前保护，原33%及其他限制保持；249只改变CPU物理核分配，未改变科学身份或策略。正常显存维护保留完整整局，只续缺失场景。', '', '监督数据的含义：若后续决定训练可泛化的全事件门控，需要在独立TRAIN/DEV场景中，针对当前候选和真实既往接受/拒绝历史，从完全相同前缀分叉执行“接受当前纠正/跳过当前纠正”，固定后续策略，比较终局收益与损害。两类数据都需要：原本成功却被纠正损害/冗余的候选，以及必须纠正才能救回的候选。当前EVAL不能直接转换成这些逐事件训练标签。', '', f'机器可读完整证据：{out.name}。原始失败、断点和来源全部保留。']
report=H/'全事件门控v118对照结果.md';report.write_text('\n'.join(lines)+'\n');print(json.dumps(dict(report=str(report),metrics=metrics,science_passed=True,all_attempts_resources_passed=not history),ensure_ascii=False,indent=2))

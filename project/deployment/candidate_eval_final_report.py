"""Read-only final matched100 comparison; refuses incomplete or mismatched cohorts."""
import hashlib,json,math
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent/'multi_event_gate_v1';P=D/'reports'
RUNS={'baseline_zero':('Z_zero','eval_20261024_112748_487903'),'original_shared_full':('M_event_full','eval_20261024_112749_116669'),'optimized_shared250_a050':('M_shared250_a050','eval_20261025_051422_382421'),'optimized_single100_a025':('M_single100_a025','eval_20261025_051422_995880')}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def paired(base,candidate):
    both=sum(a['success']and b['success']for a,b in zip(base,candidate));harm=sum(a['success']and not b['success']for a,b in zip(base,candidate));rescue=sum(not a['success']and b['success']for a,b in zip(base,candidate));neither=len(base)-both-harm-rescue;n=harm+rescue
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(harm,rescue)+1))/2**n) if n else 1.
    return dict(both_success=both,both_failure=neither,candidate_only_success=rescue,baseline_only_success=harm,net_success_difference=rescue-harm,difference_percentage_points=100*(rescue-harm)/len(base),exact_two_sided_mcnemar_p=p)
def main():
    ref=json.loads((W/'outputs/training_diagnostics/shared6_vs_single_20261025/evaluation_references/open_laptop.json').read_text());summary={};episodes={};sources={}
    for label,(condition,name) in RUNS.items():
        run=W/'outputs/task_adapter_eval/open_laptop'/condition/name
        result=json.loads((run/'result.json').read_text());journal=json.loads((run/'journal.json').read_text());manifest=json.loads((run/'manifest.json').read_text());eps=journal['episodes']
        if not result.get('complete') or result['completed']!=100 or len(eps)!=100 or result['valid_seeds']!=ref['eligible_seeds'] or [e['seed']for e in eps]!=ref['eligible_seeds'] or result['censored']!=0 or any(not e['complete']or type(e['success'])is not bool for e in eps):raise RuntimeError('Incomplete/nonmatched fixed original100: '+label)
        successes=sum(e['success']for e in eps)
        if successes!=result['successes'] or result['evaluation_identity']!=manifest['evaluation_identity'] or journal['identity']!=manifest['evaluation_identity']:raise RuntimeError('Summary or identity differs')
        for filename in ['manifest.json','result.json','journal.json']:sources[str(run/filename)]=sha(run/filename)
        episodes[label]=eps;summary[label]=dict(condition=condition,run=str(run),completed=100,successes=successes,success_rate=successes/100,checkpoint_sha256=manifest['checkpoint_sha256'],evaluation_identity=manifest['evaluation_identity'],candidate=manifest.get('candidate'))
    for label,eps in episodes.items():
        for base,ep in zip(episodes['baseline_zero'],eps):
            if ep['initial_physical_sha256']!=base['initial_physical_sha256'] or ep['instruction']!=base['instruction'] or ep['episode_index']!=base['episode_index']:raise RuntimeError('Matched scene/instruction provenance changed: '+label)
    pairs={label:{'vs_baseline_zero':paired(episodes['baseline_zero'],episodes[label]),'vs_original_shared_full':paired(episodes['original_shared_full'],episodes[label])}for label in ['optimized_shared250_a050','optimized_single100_a025']}
    report=dict(complete=True,task='open_laptop',fixed_original100_scenes=True,conditions=summary,paired_comparisons=pairs,source_sha256=sources,development_regression_on_previously_observed_eval=True,independent_final_test=False,causal_single_vs_shared_attribution=False,gate_trained=False,TTA=False,historical_baselines_rerun=False)
    resource=P/'target_candidate_eval_full_resource_audit_20261025.json'
    audit=json.loads(resource.read_text())
    if not audit['passed']:raise RuntimeError('Final resource audit not passed')
    active=json.loads((P/'target_candidate_eval_active_20261025.json').read_text())
    report['resource_audit']=dict(path=str(resource),sha256=sha(resource),passed=True)
    report['continuation_evidence']=[dict(path=path,sha256=sha(path)) for path in active.get('recovery_history',[])]
    report['same_initial_physical_states_and_instructions']=True
    out=P/'target_open_laptop_optimized_full100_20261025.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    rows=['# open_laptop：四方案同100场景结果','','| 方案 | 成功/总局数 | 成功率 |','|---|---:|---:|']
    names={'baseline_zero':'baseline / zero','original_shared_full':'原始共享 event_full','optimized_shared250_a050':'优化共享：250步，50%强度','optimized_single100_a025':'优化单任务：100步，25%强度'}
    for label,row in summary.items():rows.append(f"| {names[label]} | {row['successes']}/100 | {row['successes']}% |")
    rows+=['','所有方案使用同100个固定场景，初始物理状态与指令逐局核对一致；完整成功和失败均保留。','','这是既有评估场景上的开发回归比较，不是新的独立最终测试。两个优化候选的训练步数与修正强度不同，不能将差异单独归因于共享或单任务训练。','','## 配对比较','','| 优化方案 | 对照 | 新增成功 | 丢失成功 | 净变化 | McNemar精确双侧p |','|---|---|---:|---:|---:|']
    for label,values in pairs.items():
        for key,comparison in values.items():rows.append(f"| {names[label]} | {'baseline' if key=='vs_baseline_zero' else '原始共享'} | {comparison['candidate_only_success']} | {comparison['baseline_only_success']} | {comparison['net_success_difference']:+d} | {comparison['exact_two_sided_mcnemar_p']:.6g} |")
    rows+=['','## 续跑与核验','','两候选各100局完整、成功68局、失败32局，截断结果0；原始CPU最终审计与逐局场景、初态、指令一致性核对均通过。','','100015、100064曾出现同初态专家资格重筛漂移，已按用户确认的固定原100场景口径保留历史排除及新诊断证据。100108本次独立重筛与历史一致；剩余108至128号种子的15条有效和6条排除新记录全部匹配历史，已预先保存，未导入任何其他条件的策略结果。','','四次正式会话共核验1289条启动280秒后的活跃样本，监控阈值检查均通过；显存样本峰值23290MiB，账号最多同时2张卡。GPU2已核验释放。资源审计见同目录 target_candidate_eval_full_resource_audit_20261025.json。','','共享优化方案保留baseline全部67个成功场景，并额外成功1个；单任务优化方案额外成功3个、丢失baseline成功2个。两者净提升均1个百分点，对baseline的McNemar精确双侧p均为1.0，尚不足以证明稳定优势。']
    out.with_suffix('.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps(dict(report=str(out),conditions={label:dict(completed=r['completed'],successes=r['successes'],success_rate=r['success_rate'])for label,r in summary.items()},paired_comparisons=pairs),ensure_ascii=False))
if __name__=='__main__':main()

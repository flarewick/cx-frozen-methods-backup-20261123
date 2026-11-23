# 优化共享 adapter：四任务100局完整评估

本轮四个任务各100局，共400局，原始CPU最终审计、场景配对和资源审计均已通过。冻结第250步共享权重、ARM修正强度0.5；既有baseline及原始event_full未重跑。

| 任务 | baseline | 原始event_full | 优化共享 | 相对baseline |
|---|---:|---:|---:|---:|
| grab_roller | 86/100 | 83/100 | 87/100 | +1个百分点 |
| adjust_bottle | 90/100 | 86/100 | 91/100 | +1个百分点 |
| lift_pot | 24/100 | 20/100 | 20/100 | -4个百分点 |
| dump_bin_bigbin | 66/100 | 64/100 | 63/100 | -3个百分点 |

四任务等权平均：baseline 66.50%，原始event_full 63.25%，优化共享 65.25%。优化共享相对baseline为-1.25个百分点，相对原始event_full为+2.00个百分点。

grab_roller、adjust_bottle各比baseline高1个百分点；lift_pot低4个百分点，dump_bin_bigbin低3个百分点。结果不支持优化共享方案已经全面或整体优于baseline。逐任务配对检验均未显示对baseline的显著差异；这里使用之前已观察的固定评估场景，属于开发回归，非新的独立最终测试。

此前open_laptop四方案100局对照已完成：baseline67%、原始共享61%、优化共享68%、优化单任务68%。handover_mic缺乏可核验的完整旧对照，不在本轮四任务范围内。

资源与续跑核验：六个正式GPU会话均核验释放；两次早期守卫停止分别在283.488秒、283.626秒释放，完整局保留。运行超过5分钟的1847条活跃样本全部满足原守卫阈值，启动样本计入累计均值；账号最多同时3张卡，显存采样峰值23126MiB。dump续跑进程累计均值74.67%，CPU约250%，原8局内容SHA与续跑前完全一致。四条首次PID出现时的显存字段缺口在资源审计中显式记录，均位于启动前5秒且随后恢复，未影响五分钟后的逐PID检查。

报告与证据：

- [四任务逐场景配对结果](target_cross_task_shared250_a050_results_20261025.json)
- [400局结果及六会话资源审计](target_cross_task_shared250_a050_full400_audit_20261025.json)
- [原8局保留核验](target_cross_dump_original8_retention_verified_20261025.json)
- [dump恢复后的持续验收](target_cross_dump_gpu2_resume_sustained_20261025.json)

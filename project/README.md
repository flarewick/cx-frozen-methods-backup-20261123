# cx_new_direction_guide_TTA

## 已部署服务器接续工作（2026-10-24）

新增 [同步代码、实验数据与恢复3条Codex历史会话](deployment/CONTINUE_ON_OTHER_SERVER.md)。目标机已经跑通RoboTwin时无需重装环境；脱敏聊天分卷随Git保存，较大的当前实验数据通过同仓库Release恢复。

最新TTA方向诊断已完成，见 [完整逐景结果与归因](multi_event_gate_v1/tta_event_v1/docs/direction_review_20261024.md)。现配方尚未通过扩评门槛。

## 新增入口（2026-10-23）

- [按任务名采集、缓存、训练和评估共享 adapter](multi_event_gate_v1/task_adapter_v1/README.md)：默认 pot/laptop/bottle/roller/bin/mic 六任务，支持中断续跑，不保存视频；真实六任务训练尚待外部服务器执行。
- [TTA 四条件方向/微扰诊断](multi_event_gate_v1/tta_event_v1/direction_probe_v1/README.md)：原 full、教师重表达、原 TTA、反向增量；原八景及数值约束冻结。不是已验证的新部署配方。
- 已完成原 Docker 部署的服务器见 [pull 后启动方式](deployment/DEPLOY.md#2026-10-23增量更新)。新增代码无需重建镜像；原资产和基础模型仍需已完成传输。

## Linux / RTX 4090 跨服务器迁移（2026-10-22）

最新实验入口见 [多事件方案](multi_event_gate_v1/README.md) 与 [TTA 开发方案](multi_event_gate_v1/tta_event_v1/README.md)；下方 2026-10-05 状态保留作历史记录。

当前 multi_event_gate_v1 与 tta_event_v1 代码已纳入备份。安装环境、必要权重/资产传输、GPU 检查及可续跑评估命令见 [deployment/DEPLOY.md](deployment/DEPLOY.md)。大文件采用 SSH/rsync 白名单传输，不包含在 Git 仓库中。

跨任务方向残差与小参数适配的独立实验工程。所有新代码、缓存和结果在本目录；旧工程及 qfw 仅作只读输入。

日常入口：[活动计划](docs/plan.md)、[文档索引](docs/README.md)、[实验资产索引](outputs/README.md)、[GitHub备份](docs/github_backup.md)。以活动计划末尾最新记录为准。

## 当前状态（2026-10-05）

- 已完成四任务 240 条专家示范、冻结动作缓存、B2 训练与开发评估，以及局部干预/短窗/末端脉冲数据验证。
- 当前候选是冻结VLA上的共享限幅残差adapter，42,934参数；联合特征PCA到32维，前向使用9项合法可观测字段及当前基础动作。S0仅动作监督，SP增加位置辅助；611行固定训练、4行开发留出、同初始化/1400步。不把教师几何、任务ID或成功标签输入部署前向。
- 四任务20-seed闭环开发筛查整体没有改善，跨进程基础动作差异使原严格配对不成立；数值和结局归因有局限，不能把少量switch收益直接作为泛化证据。
- 后续同进程、共用compiled sampler的四任务完整零残差对照已通过重算。冻结S0组件重跑的40个回合现已全部验收；8个预选开发景中L0/S0/quarter/no_gripper/no_history成功5/4/5/3/2条，没有全局开关净收益。夹爪、姿态与历史作用已有数值线索，不能称全部根因已隔离。
- 固定611行/PCA/初始化/1400步的任务权重配对和四任务完整重复评估已完成。48回合完整、32首查询配对、24同条件完整轨迹重复通过；本轮同进程L0/原S0/任务归一化头成功8/8/4条，各16记录，只有8个不同开发景。新头救2/伤6，两重复各净减2，停止扩大该分支。见[独立CPU审阅](outputs/reports/attribution/task_balanced_accepted_20261004_135836_691479/review.md)。
- 专家保存帧历史S2m的[严格配对训练](outputs/effect_learning/expert_history_action_20261004_143206_614773/training_result.json)和四任务完整评估均完成。48回合、32首查询配对、24完整重复通过；本轮L0/原S0/专家历史头成功10/6/8条，各16记录、仅8个不同开发景。新头救0/伤2，丢失原S0的switch/cup恢复，未过晋级门槛，停止扩大该分支。见[最新独立审阅](outputs/reports/attribution/expert_history_accepted_20261005_021422_896290/review.md)。33项原源缺失继续登记。
- [S2n原生缓存](outputs/feature_caches/native_history_20261005_030329_014198/audit.json)已在本机H100完成并通过CPU验收：59当前行、63观测特征，复用15份/补编码48份；九合法字段、真实past、实际源base及尾块有效掩码通过，新增约0.84MiB。[11个目标时序/原生域审计](outputs/reports/audits/native_targets_20261005_030625_875605/result.json)通过，物理恢复收益尚未合格，不训练新恢复行。旧投影负结果直接复用。新的[两任务原限幅目标协议](outputs/reports/protocols/native_entry_bounded_20261005_033511_170403/protocol.json)只验shoe421004/n96、switch421001/n128：一次固定教师前缀后实时VLA续接，L0/未限幅/坐标clip/臂联合方向投影各两次反序重复；本机四条件单块截断及[独立CPU读回](outputs/diagnostics/native_entry_targets/turn_switch/entry_smoke_20261005_033634_380706/CPU_audit.json)已通过，H100和私有缓存已释放；[两个完整续接指令](outputs/reports/protocols/native_entry_bounded_20261005_033511_170403/evaluation_commands.md)就绪。目标收益通过后才按[原S2n设计](outputs/reports/attribution/expert_history_accepted_20261005_021422_896290/next_experiments.json)冻结共同增广数据/PCA/初始化/预算、只改变新增真实past训练两头。不重复四任务旧筛查或原生数采。
- shoe新入口漏用了此前已验收的陈旧杂物别名诊断处理，原指令已在本机复现。新增[独立v2协议](outputs/reports/protocols/native_entry_bounded_v2_20261005_045526_497512/protocol.json)及`scripts/probe_native_entry_v2.sh`；原条件、RGB/命令精确匹配和1e-5物理阈值不变，旧代码及switch外部运行保持不动。本机shoe完整四条件×两重复已结束并通过独立CPU验收，56重放检查真实物理误差0，四条件各0/2；switch已有完整结果复用验收，L0/raw/clip/方向投影为0/2、2/2、0/2、0/2。[合并结果](outputs/reports/protocols/native_entry_bounded_v2_20261005_045526_497512/accepted_results.json)显示有界恢复合格0，暂不启动新历史对照训练，先复用数值检查单块接续和候选构造。GPU/私有缓存已释放；[修复记录](outputs/reports/protocols/native_entry_bounded_v2_20261005_045526_497512/repair_record.json)。
- [末段命令CPU分析](outputs/reports/attribution/native_entry_commands_20261005_070748_546319/result.json)：switch仅末5/29步的右臂joint4/5超限，gripper cap未激活；两投影前24步保留教师（2e-6容差），命令FK工具端最大位置偏差26.5/31.0mm，原教师29新动作成功而两投影失败。下一步最小末5步机制测量与功能接触路径候选，继续冻结训练/router/TTA；FK不作为实际运动或接触证据。
- 最新[S2o四条件设计](outputs/reports/protocols/output_label_support_20261005_075650_144185/design.json)：复用原611/PCA/初始化/1400预算，拆开输出tanh约束和监督clip；不把0.1残差范围当机械硬定律。早期S1v/S1y已有无界学习提议离线实验，当前43k部署头尚无该严格闭环对照。[CPU预激活检查](outputs/reports/attribution/output_range_audit_20261005_075034_258694/result.json)已完成；四头训练及[CPU验收](outputs/effect_learning/output_support_20261005_082307_793921/training_audit.json)、[本机截断接口复算](outputs/effect_learning/output_support_20261005_082307_793921/local_smoke_audit.json)已完成。A与旧S0逐张量相同；原生约束/夹爪时序保留，B/D取消额外残差上界。[四任务完整指令](outputs/effect_learning/output_support_20261005_082307_793921/evaluation_commands.md)已就绪，各20分支、GPU0；本机未启动正式80分支，无新成功率结论。
- 尚无证据证明adapter提升多数任务成功率；当前不是已训练LoRA或在线TTA系统。暂不训练router、启动TTA或扩大同类数采。

## 工程入口

| 路径 | 用途 |
| --- | --- |
| `cx_tta/` | 独立采集、数据契约、缓存、训练、评估及审计模块 |
| `scripts/` | CPU 环境入口和各阶段命令封装 |
| `configs/` | 固定数据分组、采集和实验配置 |
| `tests/` | 数据隔离、时序和计算逻辑验证 |
| `docs/plan.md` | 活动研究计划及结果记录 |
| `docs/archive/` | 历史阶段报告与命令，不能直接当成当前待办 |
| `outputs/` | 原位保留的数据、训练资产、评估与可追溯结果 |
| `backup/` | Git跟踪的选定小模型、固定缓存、协议、真实数值证据及SHA清单 |

CPU 命令统一通过 `bash scripts/cpu.sh ...` 执行，复用 `qfw/robotwin/venv_pi05_rt/bin/python`，禁写旧目录的字节码，缓存留在本工程。GPU 仅按已授权的具体实验范围使用。

正式数据位于 `outputs/collections/<任务>/<配置>/collect_<UTC时间>/`，保留 HDF5 内训练所需的三路图像、状态、动作，不额外生成 MP4。历史绝对路径已写入清单，未经依赖审计不要移动数据或模型目录。

GitHub备份只包含代码/文档和选定小资产，原始数据与骨干等外部依赖需要另行保存；不能把轻量Git仓库当成完整实验镜像。协议原始路径和SHA不因备份修改。具体范围与恢复限制见[backup/README.md](backup/README.md)。

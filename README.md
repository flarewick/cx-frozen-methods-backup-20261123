# RoboTwin 冻结方法备份

本备份保存 2026-11-23 整理的三个研究方案及其原始版本身份。

| 方案 | 进入实际执行的代码 | 权重与验证依据 |
| --- | --- | --- |
| event_full | `deployment/run.sh --entry full` → `multi_event_gate_v1/scripts/eval_event_full_worker.sh` → `cx_tta.evaluate_frozen_event`。后续共享 step250 四任务用 `deployment/candidate_eval_host.py` 和 `candidate_eval_runtime.py`。 | 保留原 adapter 及其初始化配置、step250 checkpoint、投影、机器人配置、四任务 400 回合验收报告和绑定的证据。 |
| 全事件可学习候选门控 | `multi_event_gate_v1/scripts/gate_all_candidate_v237/allgate237.py` 和 `entry237.py`，经原事件控制器和受限 ARM 修正执行。 | `M_shared250_a050` 与固定 v118 权重。归档原 `all_candidate_gate_v1_20261101` manifest、333 项文件及 6 份参考登记。 |
| 原冻结 TTA093 | `native_cross_task100_v19`，连接原 actor、实际执行反馈、对象跟踪、快参数更新和原生整回合审计。 | `TTA093_20261101_v1`、`0937511a…` 响应权重、原四任务结果及冻结清单。 |
| 后续 TTA78 与门控组合 | `native_seven_official_v9`。普通组复用冻结 stacking adapter。组合组通过 `native_seven_task_gate_expansion_v1`、`gated_arm_context_v1` 保留 step250 门控头和原 TTA 辅助读出。 | `78f1849…` 响应权重及训练资格、普通与组合两套 seed 0 study、已验收结果与中断证据。七任务阶段结果以对应日志为准。 |

TTA 子目录在 `multi_event_gate_v1/tta_event_v1/` 下。算法及运行身份以原始清单为准，旧恢复入口须按说明另立运行目录。备份文件保留源 SHA，方便追溯已验收轨迹使用的模型及代码。

## π₀.₅ shared6 GT 补充备份（2026-11-25）

`place_empty_cup`、`stack_bowls_two` 和 `place_shoe` 共用这套冻结版本。G 由第 250 步共享 adapter（α=0.5）和 v118 全事件门控组成。GT 在同一套 G 上加载 shared6 响应网络及反馈预测器。在线快参数在每局开始时初始化。

- [GT 版本清单与 SHA-256](pi05-shared6-gt-20261125.json)
- [正式 GT 响应权重](project/multi_event_gate_v1/robotwin_comparison_v1/pi05_gt_shared6_v1/response_training_v2/response_step1000.pt)
- [固定 ARM 投影与归一化](project/multi_event_gate_v1/robotwin_comparison_v1/pi05_gt_shared6_v1/response_dataset_v2/frozen_basis.pt)
- [GT 加载资格](project/multi_event_gate_v1/robotwin_comparison_v1/pi05_gt_shared6_v1/shared6_response_qualification_v2.json)

用下节的 `snapshot.py restore` 恢复 `project/` 与 `evidence/` 中的完整依赖。原路径结构和资格文件保留，反馈预测器通过证据分卷恢复。RoboTwin π₀.₅ 基础 checkpoint 和感知模型按外部资产记录准备。真机接入还需对齐相机、关节动作和执行反馈接口，并验证输出。

## 文件清单

- `CODE_DEPENDENCIES.tsv` 单独列出源码与执行脚本。
- `inventory.json` 逐项列出原相对路径、SHA-256、大小、存储位置和纳入原因。它记录外部资产、训练数据出处和历史文件的剩余缺口。共享 adapter 原始训练的整套 HDF5 只记录来源与摘要。
- `project/` 是保留相对路径的可读源码、小型检查点和轻量材料。原全事件冻结快照包含有时间版本差异的参考代码。
- `evidence/` 是来自绑定清单的较大文件分卷。恢复时与 `project/` 合并。每卷和解包后的每个文件都独立核对 SHA-256。
- `tools/collect.py` 和 `tools/assemble.py` 记录依赖扫描及打包方法。静态补集包含实际入口可能导入的兼容模块。原始执行清单中的绑定与静态补集可从纳入原因区分。

## 验证与恢复

用 Python 3.11 或更新版本，在本仓库根目录运行：

```bash
python3 -B tools/snapshot.py verify
python3 -B tools/snapshot.py restore /path/to/new/cx_new_direction_guide_TTA
python3 -B tools/verify_frozen.py /path/to/new/cx_new_direction_guide_TTA
```

恢复目标必须是新目录。复用原程序时，需要准备同版本的 CUDA 用户容器或 Docker 环境、RoboTwin 官方资产、冻结 VLA checkpoint、tokenizer 和感知模型。按部署资产清单恢复外部文件，再复核路径挂载、GPU/驱动资格和科学预检。部分原清单含历史主机或容器绝对路径，需要用既有部署映射复原。服务器专属编译缓存按原来源核查或重新生成，避免移用其他 GPU UUID 的缓存。备份完整性检查只使用 CPU，迁移后的评估需要走正常的实际运行验收。

原项目的 Git 历史、Codex 聊天、账户密钥和系统环境由各自渠道保存。外部部署资产见 `project/deployment/artifact_manifest.json` 与原冻结清单。

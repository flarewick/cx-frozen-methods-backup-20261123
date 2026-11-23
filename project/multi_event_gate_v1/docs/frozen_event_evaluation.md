# 冻结 M_event_full：通用完整评估

入口：`multi_event_gate_v1/scripts/eval_event_full.sh`。旧评估入口、事件策略和权重保持不变；本入口复制已有完整评估/调度生命周期，仅扩展任务、环境、权重选择和续跑身份。

## 默认协议

- `--mode both`：同一任务评估 baseline（内部名 `Z_zero`）与 `M_event_full`，各100个有效场景。可单独选择 `--mode baseline` 或 `--mode M_event_full`。
- `--seed 0` 是 RoboTwin 官方的种子组参数，实际第一个场景是 `100000*(1+seed)`，即100000；不是把实际场景seed设为0。官方来源 `qfw/robotwin/RoboTwin/script/eval_policy_client.py:327`。
- `--task-config demo_randomized`，也支持 `demo_clean`；默认 `--instruction-type unseen`。使用选定原生配置、官方有效场景筛选、语言生成和完整动作预算。异常场景跳过后继续凑满100个有效场景，因此通常不等于连续100个seed。缺失抓取目标的已知教师规划错误按无效场景处理；策略/基础设施异常中断并保留进度，不当作失败场景或随意换seed。
- 冻结 compiled VLA、模型内部采样噪声、每个原生动作后更新观测。baseline计算同样的候选辅助路径，但实际残差为零；正式开始前运行同实例 L0/zero 一致性预热检查。
- `M_event_full` 使用现有即时多事件控制器，**不加载可学习门控、不训练、不做TTA**。事件检测、去重、重新就绪、冷却和局部退出均保持原样；合法机会全部放行，每局最多3块纠正，每块最多32个动作，ARM逐坐标残差限幅0.1rad、GRIP残差0，至少1块纯基础动作冷却；progress机会默认关闭。不是全程持续加入残差，也不是统一延后一块。

## 一条命令运行一个任务的两条件

```bash
bash /data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA/multi_event_gate_v1/scripts/eval_event_full.sh \
  --task stack_bowls_two --mode both --task-config demo_randomized \
  --seed 0 --episodes 100 --gpu 0 --max-workers 2
```

把任务名替换成 `place_empty_cup`、`beat_block_hammer`、`place_shoe`、`turn_switch` 即可；也已支持 `stack_blocks_two`。`--adapter auto` 默认按下表选择，不改变任何任务的事件规则。对新任务应显式指定 `--adapter legacy4` 或 `--adapter stacking2`，记录为冻结权重迁移；入口检查官方任务、指令、预算和共享双臂配置，未逐一做所有 RoboTwin 任务的真实GPU烟测。

同一GPU只启动一个调度器；若五任务都交给一张卡，使用一个任务列表，避免开五个互相竞争的池：

```bash
bash /data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA/multi_event_gate_v1/scripts/eval_event_full.sh \
  --tasks stack_bowls_two,place_empty_cup,beat_block_hammer,place_shoe,turn_switch \
  --mode both --task-config demo_randomized --seed 0 --episodes 100 --gpu 0 --max-workers 2
```

这共10个条件任务、1000局。调度器默认最多2个独立进程，先等待第一个进程完成一局（含编译预热），再检查CPU、RAM、GPU余量启动第二个。初始每任务预留18GiB显存/32GiB主存，显存另留6GiB与1GiB启动余量，并根据已观测峰值增加预留；这是资源估计，不能保证任意机器都恰好双并发。显存监控超时只暂停放行，不杀健康进程。每局成功率沿用官方彩色格式，池结束打印两条件总成功率/差值。可加 `--dry-run` 只做CPU预检。

## 续跑与存储

中断后**重复同一条命令**。任务、环境、权重、条件、seed、数量和源码指纹共同定义评估身份；GPU编号和并发数不影响身份，可以换卡续跑。完整成功局和失败局均保留；未完成局用原seed/有效场景序号重新开始，不拼接半条轨迹。满100局后复核并复用，不重复评估。修改环境/权重/episode总数/源码将形成另一个身份，不能混入旧结果；身份锁阻止同一评估被同时续跑。

数据位于：

```text
multi_event_gate_v1/outputs/frozen_event_eval/
  <task>/<demo_randomized或demo_clean>/<legacy4或stacking2>/<Z_zero或M_event_full>/eval_<时间>/
    manifest.json       固定参数、权重/源码哈希
    journal.json        原子提交的筛选记录及完整episode（续跑必需）
    episodes.json       便于阅读的episode索引
    result.json         最终成功数、成功率、实际有效seed和编译会话
    sessions/process_*/ 运行配置、私有源码、终端日志、压缩查询记录
```

调度日志在 `outputs/frozen_event_pools/pool_<时间>/`；其中 `comparison.json` 比较成功率并检查有效seed、初始物理/观测和指令。未完成的池保留已启动worker结果；不静默重试失败进程，修复原因后重跑同命令续接。

不保存视频、RGB、训练特征或事件NPZ；保留结果、压缩事件/动作哈希记录和复现信息。大型静态资产使用受保护的只读引用，不为每局复制；编译缓存在私有 `/tmp`，正常退出和可处理的中断后清理。容器强杀不能保证执行清理，不能删除仍在运行进程的缓存。不要删除 `journal.json` 或其引用的完整episode证据。

两个条件为独立模型/仿真/编译进程，同seed并不保证动作级完全一致。`comparison.json` 的成败翻转仅为描述统计，不能直接用作同前缀因果救回/误伤训练标签。对正式收益仍应逐任务报告结果，并保留独立编译复现；不能只看跨任务合计。

## 可用 adapter 与后续训练

| 名称 | 已覆盖任务 | 固定checkpoint |
|---|---|---|
| `legacy4` | cup / hammer / shoe / switch | 父目录 `outputs/effect_learning/output_support_20261005_082307_793921/A_bounded_clipped_seed0.pt` |
| `stacking2` | bowls / blocks | 本目录 `outputs/stacking_adapter/shared_20261014_031147_697383/adapter_seed0.pt` |

两者均为约4.3万参数的共享方向残差头，不是每任务一个模型。legacy4保留真实前一块历史；stacking2在训练/推理均为current-only，使用两任务各40个训练专家场景和10个验证场景，尚属专家初始化阶段，不包含恢复/成功保持监督。本次请求的五个任务已有适用权重，无须先补训才能开始比较；`demo_clean`仍用同一固定权重，作为环境变化测试。

当前不存在已经统一训练并验证的“六任务通用adapter”。此轮按两套权重评估只能说明对应权重下的跨任务表现，不能声称单个adapter覆盖所有任务。后续推荐保留本轮对照，先在更多任务评估冻结迁移，再用各任务独立TRAIN、按任务/场景平衡的专家及恢复/成功保持数据训练**一个共享adapter**；原四任务和堆叠任务同时回测，防止遗忘。单任务adapter可作容量/监督上限诊断，暂不作为通用方案默认实现。若候选权重更新，学习门控的收益/误伤监督和校准必须与新候选重新匹配；本轮full不依赖门控训练。

验收记录在 `outputs/reports/frozen_event_setup_20261021/`：17项CPU测试（含真实权重加载、续跑、真实CPU子进程并发、资源监控异常、场景比较）、五任务×两环境×两条件预检。新通用入口本轮未启动GPU仿真，不将上述测试称为真实GPU吞吐或成功率验收。

当前状态更新（2026-11-10 11:58）：根据用户要求，撤销C8额外“日间账号必须全空”的限制，C8纯CPU队列已确认无GPU并operatorstop。未来用 `native_seven_official_v9` / `native_seven_campaign_v8` / unit `cx-tta-seven-official100-v9.service`，invocation `13e837237e7441a3bc817d3bffb9ac56`，MainPID1551738。模型/数学/seed0/全部whole对照保持不变，仍从前三任务官方索引6–11续跑，C5normal前缀各6保留。GPU1是当前空卡，GPU2设备/PID真实均值已过70、显存68.48%；新观察器满原180窗口并通过全部规则才准入。当前只确认队列启动，actual GPU PID待fresh核对。

修复的是账号聚合顺序：旧逐样本min(device,own)再平均使异步记录偏低；新observer5对only-own PID分量先按原180分别求均值，再以100/device分量均值保守封顶，最后MAX；单PID/device/VRAM/CPU/RSS/raw采样/None0/完整窗口/规则阈值全不变、不借foreign。CPU原39规则、原始AST/实际窗口、foreign-only0/None0及交错示例均通过。guard4/整体C9 CPU分别实际0/0.918秒、0/46.403秒。旧原始失败不重标，旧normal仍原审计有效；该修正不是GPU吞吐达到80%的证据。

C9两组study已freeze0，plain001=5182003c55124ca268aaebe95760e218b0ffb3ebe08a623ae8d1626cf6ed15f3，gated000=6d55befac61bea46742a1b37a13da3bf437a0663759bffb3fe58b0eeccdb831d；source全冻结。firstlogic cap28800/继承17144.223/余11655.777，其他logical14400共享两半，不新增GPU耗时预算。新ownedobserver5从约11:55有35h窗口，原observer4/sharedLIBERO不动。主要前台watch12341/watch_v3每45秒至约13:57，watch退出不代表GPU结束。最终只使用 report_v7/resource_readout_v6/parentcampaign_v8，不能用旧混合汇总。

当前状态更新（2026-11-10 11:27）：C8新unit `cx-tta-seven-official100-v8.service` / invocation `1f0d6dac1f72416aba476604daec95a9` / MainPID3158267已启动 `native_seven_campaign_v7`，后端 `native_seven_official_v8`。目前CPU等待账号协调，没有本方GPU进程；物理空卡不等于多卡准入。日间新准入先要求原observer4实际最新180秒中账号持续无其他GPU，完整窗口/50样本/174秒覆盖/9秒gap/8秒新鲜度均保留；运行中仍guard3原严格Rule8，不停止别的对话任务。夜间按原规则准入/max3。该协调仅减少临界多卡反复退出，不保证无关后来任务绝不启动。

C7最后因成熟另一卡账号180均值69.9508%低于70%而正确保护退出，963.156秒/actual1/所有partial完整保留并排除，GPU已释放；不是新启动宽限bug。C8协调CPU8项实际0，完整CPU实际0/46.344秒；plain001/gated000 freeze实际0，study分别b77c98180086c0964efccf7b78795b2866ecf16a795ef9c97ff6cab3047e27fe、6a7df6cf4e987ae37c8eb8e1bbef14e41f8d72d1e032893d82bf05d97e7b8635。所有绑定源冻结。第一逻辑cap28800不再增加，累计17144.223/剩11655.777，其他逻辑14400且两半共享；正式failed13687.316与合格3456.907分列。现有正式资格仍C5三项各6景，gated0；C8下一plain001官方索引6–11，不晋升失败partial。

新最终汇总必须使用 `native_seven_report_v6` 和 `native_seven_resource_readout_v5` / parent campaign_v7；其他旧报告不用于这条新链。80%尚未达到，数学/候选78/seed0/物理/精度/wholecontrols均不变。前台watch30239于11:26:50起两小时，后续可轮换新只读watch_v2显示账号等待原因，watch结束不等于GPU结果。

当前状态更新（2026-11-10 10:59）：C7 正式续接已启动，unit `cx-tta-seven-official100-v7.service`，invocation `3dedde56330a4377bc4ab548871ce5f2`，MainPID3047568，Restart=no/35h。后台是 `native_seven_official_v7`，顺序 parent `native_seven_campaign_v6`。本次仅修复其他同账号新 GPU 进程的 Rule8 创建宽限及对应 raw replay；observer4/统计/准入/阈值/科学候选均未改变。C6 旧 batch actual1、1776.919秒和部分结果完整保留并排除；不能归因管理员 SIGINT。新独立 policy60项 CPU0、guard3 CPU0、整体CPU0/43.683秒；plain001/gated000 study实际freeze0，SHA分别79efe3e1a7f9f39e02f1fb1fa87f8fda6e34674aa7e602601fd10c80e7351b65、39f7953007f8110a5ab13cfb54525fe0fbffa9e830896be3509ba20edb51e051。所有新绑定源现已冻结。

继续直接引用 C5 plain000 actual0 的前三任务各6景，续接plain001官方索引6–11，再到100，再四项gated100。第一logic未来累计上限28800秒含全部历史16181.067秒，余12618.933；其他逻辑14400/两半共享不刷新；服务器规则不变。正式保留failed累计12724.160秒，与合格前缀3456.907分列。当前只确认新CPU队列启动，实际GPU/root/PID必须fresh读；主要前台watch60632从10:59每45秒至约12:59，watch退出不是GPU结束。最终只用新 `native_seven_report_v5` 与 `native_seven_resource_readout_v4` 读取C5 normal前缀+C7，不用旧report。80%尚未实现；最终七项700配对全部actual0/native/raw/resource/release后才报告成功率。

续接更新（2026-11-10 10:20）：C5第二批达项目预算退出，3984.366秒及原失败记录保留，未完成数据不计成功率。C6仅扩充未来固定时间预算，科学实现与C5逐文件namespace归一后相同；引用C5三个任务各6景合格前缀，从第7景继续，复用18份pure expert/q0准备。当前响应模型78、官方seed0、每任务100和全部原生控制不变。C6完整CPU/两组冻结均actual0，campaign_v5已启动。后续统计用report_v4与resource_readout_v3，计入旧failed10947.241秒且不双算合格前缀。

正式进度（2026-11-10 09:07）：C5首批正常退出并通过全部审计，plain三任务各6/100景；full→TTA成功数lift_pot 1→1、handover_mic 4→4、stack_bowls_two 1→2。小样本非最终收益结论，保持冻结继续官方seed0到100。四项门控组合仍排在plain组完成之后。

当前状态（2026-11-10 08:07）：early8调度的15条完整TRAIN轨迹/151次查询已与原合格实现逐字一致，资源与正常退出均通过。post300 GPU利用率45.04%，未达80%；正式C5最终CPU检查进行中，七项新100景尚未合格完成。详见AGENTS.md §425。

> 最新覆盖（2026-11-10）：density_v2 的12路TRAIN探针也已因真实Rule4停止，771.612秒、GPU完整释放。吞吐诊断累计6061.735/7200秒，原TRAIN累计3788.582/7200秒，下一探针取两账交集，最多1138.265秒，不扩预算。此前正式失败6962.875秒仍单独保留，正式七任务均0/100。
>
> 当前冻结入口是 `native_seven_plain_early_v1/plain_000`：固定3个第一TRAIN成员、8路、真实full q0来源核验且完整结果尚不存在时派发独立配对分支、最多一个原始取图参照并行，完整15回合控制不减少。CPU33.100秒实际退出0，身份 `4af92ad56bd0700e9fb80880b84319d571656689146f0d92c1cdeee08d1eaf1e`。独立服务 `cx-tta-plain-early8-v1.service` 已排队；最新七卡均占用，本账号0卡，未GPU。通过完整native/resource/actual0/release后，还必须由 `native_seven_early_acceptance_v1.audit` 核对与原合格P3全部15分支的观测／噪声／动作／物理轨迹／fast更新字节一致，才进行C5最终CPU和正式恢复。C5草稿已改plain8早派发、gated12保留原完整full后派发；不是已经启动或100景完成。历史描述以本覆盖为准。

> 最新状态（2026-11-10）：正式v4首批已因真实Rule4主动停止并释放GPU；三分钟利用率34.883%，不是观察器异常。full23/TTA18及已有控制均未晋升，新七任务正式合格进度仍各0/100。累计失败GPU6962.875秒保留，详见 `outputs/native_seven_failure_diagnosis_v2/diagnosis.json`。
>
> 当前仅运行独立 `native_seven_plain_density_v2/plain_000` TRAIN工程验证：固定6个纯TRAIN准备场景、12真实lane、任务轮转，response78和原科学数学不变。该探针同时计入既有TRAIN与吞吐预算，剩余上限1909.877 GPU秒，使用当前180秒资源规则；尚未完成native/resource/正常退出验收。`native_seven_official_v5` 的每任务6景短会话与共享逻辑预算实现只是未冻结候选，必须等此TRAIN整批通过和最终CPU预检后才正式启动。不能启动旧C4失败入口或把partial当100景结果。

# 七任务TTA与全事件门控组合评估

用户要求：`lift_pot`、`handover_mic`、`stack_bowls_two` 的TTA；以及 `beat_block_hammer`、`turn_switch`、`adjust_bottle`、`open_laptop` 的 event_full＋全事件门控＋TTA，并优化GPU吞吐。沿用官方 `--seed 0`、`demo_randomized`，每任务100个专家有效场景，成功失败同等计入。本文记录冻结方案与执行条件。两组方案均已冻结并进入正式队列，当前等待空卡；所有七任务新的正式进度仍为0/100。

## 两组底座与对照

|组|任务|TTA底座|新配对对照|
|---|---|---|---|
|普通TTA|lift_pot、handover_mic、stack_bowls_two|当前已训练的78f1849响应候选、原stacking读出与反馈后作用机制|同场景冻结 M_event_full|
|组合TTA|beat_block_hammer、turn_switch、adjust_bottle、open_laptop|原shared step250、alpha=0.5候选＋冻结v118全事件门控，再加入原已训练TTA辅助读出的fast增量|同场景、同数值入口的step250＋全事件门控，无TTA|

正式目标共700个TTA回合及700个配对无TTA回合，另有规定的完整零更新、影子更新与原生参照。当前TRAIN功能批次另外保留baseline，不能并入正式分母。

响应checkpoint：`78f1849a0f825398a6aa43cc3eec9cb09a814fc472a495179ac97379f5a044d5`；资格文件为 `outputs/native_fit7_postexit_recovery_v2/qualification.json`。它是后来完成五任务校准的候选，**不是**原冻结TTA093。原093的bowls完整100景67%、其余历史结果和本次新结果分开保存；不得拿旧full分母拼接新TTA。

pot/mic仅扩展语言与实例关联，使用 `native_pot_mic_semantics_v1`；真实对象缺失时保留missing，缺反馈或零更新的景仍进入100景分母。不使用这两个任务的专家标签训练新响应模型，也不将语义覆盖当作TTA成功。

## 为什么组合版保留两个读出

原全事件门控冻结版使用的候选为 `M_shared250_a050`：checkpoint SHA `6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7`，最终ARM限幅0.05 rad。门控v118 SHA为 `1534934341ee285b6dfb7653d1c9e06bbd0aa0c462e3d6963a6d42405550fdfc`。

CPU实测该候选的ARM读出权重及行空间投影与当前TTA不同，最大权重差0.331590。不能把原响应模型的固定buffer换成新头后继续冒称原训练资格。

组合版分别保留两个冻结读出：门控的step250读出，以及与已训练响应模型严格匹配的原stacking辅助读出。二者使用**同一当前合法输入、同一次base采样**，辅助读出不增加VLA查询或噪声采样。128个fast参数仍按原物理反馈更新，原辅助隐藏坐标、ARM矩阵、行空间投影和训练资格不变。

记门控候选为 `rG`，原辅助读出及其在线更新方向为 `rA0/rAφ`，先构造：

```text
fast_increment = rAφ - rA0
rGφ = clip(rG + fast_increment, -0.1, 0.1)
```

保留原fast信任约束，并额外检查组合方向对门控候选的信任约束。原事件机会由同一v118对**实际准备执行的当前候选**评分，批准后施加 `0.5 × rGφ`；夹爪保持base。

原反馈后fast-only作用机会也必须经过同一v118评分，使用真实已出现的源事件上下文；对于原生准入记录确认的第0次初始ARM来源，新版 `gated_arm_context_v1` 使用五个事件标记均为0以及真实来源年龄、介入次数和间隔，不伪造夹爪/进度事件、不插入事件队列。其他无来源情形不执行。通过后仅施加 `0.5 × (rGφ-rG)`，共用原冷却及最多三次介入预算。拒绝不能消耗真实预算，不能经fallback执行被拒绝的候选。零fast增量精确退化为同场景门控对照。

这属于新组合版本 `shared250_alpha05_all_v118_plus_qualified_auxiliary_fast_increment_ARM_context_v2`。门控的额外反馈机会属于迁移使用，未经重新校准；不称为新训练过的成功概率。新组合与其对照共用当前TTA数值入口，不将历史compiled门控结果直接当作该入口的逐字参照或严格配对结果。

## 代码与验证

- `native_seven_task_gate_expansion_v1/paired_heads.py`：分别核对和加载原门控及辅助读出；fast增量映射与组合信任检查。
- `composite_head.py`、`paired_actor.py`：复用原ActorQuery的owner CodeTypes、一次base及噪声，分别记录两读出。
- `gated_compose.py`：原全事件门控、alpha0.5，以及不能绕过否决的反馈作用。
- `gated_episode_v2.py`：原完整episode主链，仅增加明确的组合和逐候选记录；修改清单为 `gated_episode_v2_edits.json`。
- `gated_native_audit_v1/replay.py`：保留原生步数/命令回执/phase95与0.25秒/源与单次packet等检查，逐查询重算两读出、593维门控输入、评分、动作、预算及冷却。
- `native_seven_plain_dev_v1`：首个三任务TRAIN功能批；因Rule4失败，保留887.467秒及所有失败记录，不晋升部分回合。
- `native_seven_plain_dev_v2`：同六景的8路准备缓存恢复版，保留原全部控制及失败用时，不导入旧policy、掩码或训练标签。
- `native_seven_plain_dev_v3`：修正完整参照集合审计后的三任务功能批，27完整回合/正常0/资源释放均通过。6景都有更新及动作差分；pot0→0、mic2→2、bowls2→1，各分母2，只证明通路，不证明收益。
- `native_seven_gated_dev_v2`：修正双读出参照审计后的四任务功能批，36完整回合/正常0/资源释放通过；4景接受更新、实际动作改变0。
- `gated_arm_context_v1/{application,episode,replay}.py` 与 `native_seven_gated_dev_v3`：明确初始ARM来源，固定门控仍可否决。36完整回合及GPU子进程正常退出0；退出后的CPU守卫竞态导致旧外层host1。独立CPU补审正常0并完成所有科学/控制/原始180资源精确回放，见 `outputs/gated_postexit_recovery_v2/qualified_TRAIN/qualification.json`。旧host1与1644.661秒保留，没有重跑GPU或改写失败。
- `native_seven_official_v4`：正式官方seed0配对评估，12景/task/物理段、最后4景，共9段；每段必须完整退出及所有原生/数值/资源审计通过，再承接专家筛选状态。只有合格段进入分母，失败不自动原样重启。

截至本文建立，CPU已验证333项原门控归档；pot/mic共15,120条模板/描述展开；10条与85条真实缓存查询的辅助坐标、原门控决策和动作、零增量、否决、冷却、预算、夹爪等检查通过。模板枚举含重复，不等于15,120个独立场景。两次仅属CPU fixture的问题已保留原实际失败记录和后继检查，不能改标旧失败。

原生TRAIN验证使用普通组三任务每任务2景（实际种子池7136000起），组合组四任务每任务2景（7136200起），分别至少27和36个完整回合。先完成源摘要、真实视觉反馈覆盖、零/影子控制、门控动作回放、正常退出及资源释放，再冻结正式100景入口；不根据正式EVAL结果选择模型或阈值。

## GPU资源与吞吐

现行规则以 `/home/qiufw/服务器使用注意事项.md` 为准，使用v15口径：最近180秒有效样本，先分别平均SM与显存控制器再取MAX，按空闲卡数量选择门槛。全天同账号最多三卡，单PID单卡，显存额外上限24563MiB、每卡16逻辑CPU。CPU/RSS、先来用户、多卡日间门槛和动态守卫均保留。

此前12路工程批正常退出，但因hammer专家有效场景成员改变，完整同场景加速比较未通过。其启动300秒后平均利用率60.275%、最高180秒均值72.75%，**未达到80%**。共同九景42分支331查询逐字一致，但不能据此晋升完整45分支或宣称严格加速比。证据在 `outputs/native_eval_throughput_diagnosis_v1/`。

新批次继续复用真实分支并行、原生渲染等待重叠、图像读回、编码图及静态CPU堆优化，不增添填充计算、不加策略查询、不改精度/物理/步数。利用率和实际吞吐随真实工作一起验收，80%仍为待实测目标。三任务从12路降至8路是失败后的资源候选，不预先宣称更快或更有效。

只读监控命令，在TTA目录执行：

```bash
python3 -B -m native_seven_task_gate_expansion_v1.status --version native_seven_official_v4 --stage plain_000
python3 -B -m native_seven_task_gate_expansion_v1.status --version native_seven_official_v4 --stage gated_000
```

这两条不启动GPU。不要重启已存在的unit或修改已绑定源码；实际未正常退出、控制未通过或资源未释放的批次不能作为正式结果。

正式来源冻结：普通组 `plain_000` study `38eb78c8bcd74a2dcdd82414ae1ced34e222618ab02680ef53ac2c5c31004c8d`；组合组 `gated_000` study `3fe942a95db1a8211aa98da48a826619280a6b90cec072a13ef7c99f7aaab1e2`。`native_seven_official_v4` 源码全部冻结，不得热改。

当前队列入口 `native_seven_campaign_v3.main`，unit `cx-tta-seven-official100-v4.service`，两组顺序运行。队列状态为 `outputs/native_seven_campaign_v3/live_status.json`；组内合格进度为 `outputs/native_seven_official_v4/supervision_{plain,gated}/qualified_*.json`。只认每组 `official100_complete.json` 或最终 `outputs/native_seven_campaign_v3/all_seven_official100_complete.json` 为完整100景结果。没有空卡时仅CPU等待，不占GPU或抢卡。

GPU侧新正式使用 `native_governor180_v4` 的TTA自有CPU观察器，原v15采样/统计/政策保持，原LIBERO观察器不动。`native_gpu_drained_guard_v2` 仅在真实GPU子进程已正常wait0、fresh nonce/namespace/birth匹配、原唯一GPU PID已消失后允许CPU审计继续；此时别人的新GPU进程不再被视为本方GPU优先权冲突，不信用其利用率。有活跃/未知/失败/新GPU PID或缺正常退出证明时，仍执行全部原保护。

吞吐优化把每组会话从34减至9，每个会话的原wholezero/shadow/必需full参照都保留，700个TTA及700个配对控制不减。首批普通组36景/81完整回合、组合48景/108完整回合，单卡仍8/12个真实slot，7200秒实际GPU预算含失败。CPU串行/并发20种情形和原数值控制通过，GPU整批利用率与退出仍待实测，不预先宣称80%。

组合TRAIN补审结果：hammer full1/2→TTA1/2、switch0/2→0/2、bottle2/2→2/2、laptop0/2→0/2；4景接受更新，1景实际改变动作，没有净成功率提升。新原生GPU批300秒后完整raw利用率66.0477%、rolling180最大78.8167%，80目标仍未达到。这些均不代替正式100景。


最新恢复说明：前一正式入口在观察器单次采集异常后主动停止，已用2345.848秒、GPU全部释放，整批没有资格；部分full/TTA/控制结果不并入100分母。当前v4仅复用14份完整专家与q0准备（bowls12、mic2），重新CPU语义和原生q0核对，不导入旧policy/masks/学习标签。源TTA数学与78响应不变，补齐缓存首景与再生参照的去重，保留所有必需whole参照。

观察器v4在采集失败时另外记录errors.jsonl/last_error.json，上一有效快照字节、时间戳、mtime都不更新；原8秒新鲜度、9秒指标间隔、有效窗口和全部规则均不放宽。首个plain逻辑批总预算一次性10800秒，已用2345.848继承、新增最多8454.152秒；其余批次仍7200。依据用户此前允许增加实验时间预算，旧失败与预算保留，服务器门槛不变，不自动runtime延长或重复失败GPU尝试。

当前队列 `cx-tta-seven-official100-v4.service`，invocation `f8a534d499f24cee901745bb4a64b187`，CPU父PID1381337；观察器CPU PID1349343，仅TTA自有，原LIBERO共享观察器不动。当前七任务正式合格进度仍0/100，等待真实空卡。最终报告使用 `python3 -B -m native_seven_report_v2.report`，必须完整700配对后才生成。

# 已跑通 RoboTwin 的服务器：同步工作与恢复 Codex 会话

2026-10-25更新：**最新研究接续请先读 [TTA交接说明](TTA_HANDOFF_20261025.md)**，其中有本轮增量结果下载与下一步顺序。下文保留2026-10-24环境/聊天备份的恢复办法；聊天快照没有随本轮结果更新。

目标机已安装依赖并成功评估过本方案，**无需重建环境、重传基础 VLA 或 RoboTwin 资产**。沿用原来已跑通的评估环境和启动方式。

## 1. 更新代码

在目标服务器已有 clone 中执行：

```bash
cd "$HOME/cx_migration/cx_new_direction_guide_TTA"  # 改成实际 clone 路径
git pull --ff-only origin main
```

如果本地有未提交修改，先另行保存再 pull；不要 reset/clean 掉目标机的新工作。此次新增 TTA 结果审阅脚本、完整结果和plan更新，以及实验/聊天备份恢复工具。上一轮 TTA 与通用任务数据采集/训练代码也在仓库内。

当前实验入口以 [TTA plan第15节及以后](../multi_event_gate_v1/tta_event_v1/docs/plan.md) 和 [最新交接说明](TTA_HANDOFF_20261025.md) 为准。零更新、参数化对照、16景配对响应补采及7组时间跨度实验均已完成；下一步是复用缓存检查局部干预差分可识别性，尚不进入线上TTA扩评。

## 2. 按需补齐实验权重、数据与证据

代码和脱敏聊天分卷随 git pull 获取。实验数据放在同一私有仓库的 [migration-20261024 Release](https://github.com/flarewick/cx_new_direction_guide_TTA/releases/tag/migration-20261024)，避免将反复产生的大型训练数据写入 Git 历史。**git clone/pull 不自动下载 Release 数据**。

备份包括：现有加载器所需的 adapter、PCA、训练/来源证书及其依赖；当前多事件目录中的训练数据、候选/门控诊断与评估记录；两轮TTA完整轨迹、数值缓存和离线预算结果；旧正式评估与报告。排除运行环境、基础模型、仿真资产、编译缓存、视频和暂存目录。不是对所有历史被清理数据的恢复，也不是所有旧实验训练集的全集。

已部署服务器只为运行原有评估时，通常无需下载整个实验包；需要继续离线分析、共享adapter训练、追溯旧结果时再恢复。下载器复用已经通过SHA验证的分卷，未完成分卷重新下载；恢复器复用相同文件，遇到不同的现有文件立即停止，不覆盖新工作。

```bash
cd "$HOME/cx_migration/cx_new_direction_guide_TTA"

# 优先复用 gh 登录；未安装/未登录时会隐藏输入 GitHub 令牌。
# 不要将令牌写进URL、命令参数或仓库文件。
python3 deployment/github_research.py download \
  --destination "$HOME/cx_migration/research-20261024"

python3 deployment/research_backup.py restore \
  --bundle "$HOME/cx_migration/research-20261024" \
  --project "$PWD"
```

清单含逐文件SHA；下载先核对GitHub远端摘要，恢复再核对分卷及每个新文件。保留原绝对路径/身份记录，沿用原部署的容器路径映射，不改checkpoint、journal或源码指纹来绕过检查。不同硬件或编译进程的结果仍须分层；已有完整结果用于审阅，不把旧未完成条件组拼接进新服务器实验。

## 3. 恢复3条 Codex 用户会话

备份在 `deployment/backups/codex-20261024/`，含脱敏后的JSONL、会话名称索引，以及经过筛选的 `state_5.sqlite` / `thread_history_1.sqlite`。只保留3条用户会话；不含auth.json、.env、config.toml、登录凭据、后台审查会话、队列、日志数据库或插件配置。**新服务器需使用自己的 Codex 登录，历史记录不等于账号登录状态。**

主会话原JSONL已经存在损坏记录，本次不把它们改写成新内容；保留原字节偏移与数据库中的分页历史。识别出的令牌已脱敏。真实恢复测试分别使用 `codex-cli 0.163.0` 和 `0.164.0-alpha.11.1`：两版本中3条均通过 thread/read、历史分页读取和 thread/resume，未启动模型回合。建议使用已验证版本或兼容更新版本；不同客户端UI、插件和登录状态没有随备份复制。

为避免覆盖另一台服务器现有历史，**默认使用一个独立且为空的Codex目录**：

```bash
cd "$HOME/cx_migration/cx_new_direction_guide_TTA"

python3 deployment/codex_history.py restore \
  --codex-home "$HOME/.codex-cx-history" \
  --workspace "$PWD"

# 此目录只需恢复一次；已经继续聊过后，不要重新覆盖它。
env CODEX_HOME="$HOME/.codex-cx-history" codex login

# 直接继续当前论文/实验主对话。
env CODEX_HOME="$HOME/.codex-cx-history" codex resume \
  01a0bdcc-3e98-7881-b124-b3ca027de12f -C "$PWD"
```

三条会话ID：

| 会话 | ID |
| --- | --- |
| QFW_TTA_PI0.5：当前项目主对话 | `01a0bdcc-3e98-7881-b124-b3ca027de12f` |
| QFW_排查 Codex 历史记录恢复失败 | `01a0ed0e-2017-7c72-a97a-dcd42ac89d6e` |
| QFW_计算两个路径占用空间 | `01a0f631-a3bd-7901-b172-7e0aa77a252e` |

也可运行 `env CODEX_HOME="$HOME/.codex-cx-history" codex resume --all` 选择会话；CLI官方支持按ID恢复并跨目录列出历史。[官方CLI说明](https://learn.chatgpt.com/docs/codex/cli)

如果使用VS Code，需要让远端Codex扩展进程使用同一个CODEX_HOME，再重启该扩展；在终端临时设置变量不保证已运行的扩展会自动切换目录。不要直接覆盖目标机已有的 `.codex`。恢复脚本改写的是新副本的会话文件路径和工作目录，原服务器内容保持不动。

可选：新机只测试读历史/恢复、不发送模型请求：

```bash
python3 deployment/verify_codex_history.py \
  --codex-home "$HOME/.codex-cx-history" \
  --output /tmp/cx-codex-restore-check.json
```

官方区分“读取历史”“恢复会话”和“启动新回合”；这里的验收没有调用 turn/start。[App Server说明](https://learn.chatgpt.com/docs/app-server)

## 4. 接续工作提示

聊天快照截至2026-10-24，后续新消息不会自动写入Git。继续工作前先让Codex读取 [最新TTA交接说明](TTA_HANDOFF_20261025.md) 及TTA plan第15节及以后；代码/报告才是当前实验状态的依据。当前TTA尚未确认有效，不能把迁移成功、零更新恢复测试或代理loss下降当作新算法成功率证据。

如果只需要代码更新和恢复聊天，完成第1、3节即可。环境已就绪时不必再次执行原 `DEPLOY.md` 的安装流程。

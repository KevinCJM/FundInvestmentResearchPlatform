# 本地分支与 Dev 差异审核

日期：2026-09-13。范围：当前实际工作区（含未提交、未跟踪文件）与 fetch 后的 `origin/Dev`；排除文档，保留业务代码、测试及 CI 配置。

## 结论

**当前内容不能作为新改动提交回 Dev。差异全部来自本地落后，没有发现尚未进入 Dev 的本地新增代码。** 直接用本地内容覆盖 Dev 会撤销已合入的修复。审核未通过，因此本轮不 commit、不 push、不创建重复 PR、不执行合并。

- 当前分支：`codex/frontend-design-quality`，HEAD `2329845b345665a9bacc95e36b814868a8e288ea`。
- 最新远端：`origin/Dev`，`f6f85d0ff967d223d59784b305e2e19069e75518`。
- [PR #14](https://github.com/KevinCJM/FundInvestmentResearchPlatform/pull/14) 已于 2026-09-13 合入 Dev，来源为 `df8dbce1a80a71cb121b4240085ef32ea1d6a040`；该来源与 Dev 的完整文件树一致。
- [PR #12](https://github.com/KevinCJM/FundInvestmentResearchPlatform/pull/12) 也已合入 Dev，包含指标日期说明和对比度修复。

## 差异归属

| 项目 | 结果 |
| --- | --- |
| 与 Dev 一致的非文档文件 | 996 |
| 不同业务代码文件 | 14 |
| 不同测试文件 | 12 |
| 本地缺少的远端 CI 配置 | 4 |
| 尚未进入 Dev 的本地新代码 | 0 |

26 个本地不同文件均逐字匹配 `2329845` 或 `b61ab11` 的历史 Git blob；两者均为最新 Dev 的祖先。4 个 CI 文件仅远端存在。本地当前 HEAD 和远端同名开发分支均无 Dev 尚未包含的独有提交。不能把“相对旧 HEAD 的未提交改动”直接解释为“尚未合入 Dev 的改动”。

## 阻断问题：将本地内容提交回 Dev 会发生什么

| 级别 | 回退影响 | 当前本地位置 | 远端处理 |
| --- | --- | --- | --- |
| P1 | PIT 设置未知时仍按今天继续诊断/保存；未知状态与明确关闭 PIT 混淆，部分时钟变化也不清除旧确认 | `frontend/src/pages/InvestmentObjectivesWorkspace.tsx:18`、`:59`、`:93`；`frontend/src/pages/StrategicAllocationWorkspace.tsx:65` | `df8dbce` 区分 undefined/null/日期，并阻止未知时钟下操作 |
| P2 | 原始目标保存入口绕过诊断与限制确认；SAA 没有要求目标具有确认记录 | `backend/strategic_allocation/routes.py:41`；`backend/strategic_allocation/service.py:86`、`:339` | `df8dbce` 删除原始入口，并增加 `MANDATE_CONFIRMATION_REQUIRED` |
| P2 | 从 TAA 返回 SAA 丢失实际选中的政策版本，重新进入草稿编辑器 | `frontend/src/app/allocationJourney.ts:116`；`frontend/src/components/tactical-allocation/BaselineSetup.tsx:42`；`frontend/src/pages/StrategicAllocationWorkspace.tsx:32` | `df8dbce` 传递 baseline ID，从服务端读取对应不可变政策 |
| P2 | 标量数据丢失成立/上市日期及来源覆盖，公告字段缺失时的说明与披露状态不准确 | `backend/custom_indicators/series_provider.py:360`、`:650`；`frontend/src/components/metrics/MetricDisplay.tsx:123` | PR #12 已保留原身份和来源信息，并区分公告字段不可用等状态 |

其他差异包括已修复的深色背景文字对比度，以及对应回归测试；本地也缺少远端现行分支保护配置。这些均没有形成新的业务需求。

## 本轮验证

1. 刷新远端，读取实际 PR 合并状态；核对完整 SHA、文件树、Git 祖先关系及每个非文档文件的内容和模式。
2. 阅读完整业务代码差异，使用 CodeGraph 导航，核对 `df8dbce` 的原始修复及其测试。
3. 从 Dev 原样取出 `test_indicator_input_date_context.py`，选择标量身份/来源覆盖和缺失公告字段两组共 16 项，同一测试文件对照运行：**当前本地 16 failed；Dev 对应完整文件树 16 passed**。全部使用临时 Parquet，不修改正式数据。此结果为定向回归，不是全量测试。
4. 差异文件在审核期间未变化，暂存区为空。
5. 独立审核代理 `/root/saa_taa_pr_review` 对目标确认执行相同探针对照：本地原始 `/mandates` 返回 201，未经确认的目标产生 4 个政策候选并成功保存政策；Dev 原始入口返回 404，预览和保存均拒绝并返回 `MANDATE_CONFIRMATION_REQUIRED`。这是实际服务/路由调用证据，不只是源码判断。
6. 同一独立审核代理执行两条前端用例：`未知时钟禁止SAA比较，明确无PIT允许比较，重新未知后丢弃候选`、`从TAA返回精确的已保存SAA版本，不依赖其他目标的本地草稿`。**当前本地 2 failed，Dev 同树 2 passed**。分别确认比较按钮错误启用、指定历史政策未展示。
7. 独立审核最终结论：**BLOCKED，不批准用本地旧内容覆盖 Dev；没有未合入的新代码需要提交。** 所有复现进程已结束。

可复现证据位于 `.run/local-dev-review-20260913/`：`provenance.json`、`remote-to-local.diff`、`test_remote_input_date_context.py`、`date-context-local.log`、`date-context-dev.log`。该目录为本机审计产物，不应纳入代码提交。

独立证据位于上述目录的 `independent/`：`confirmation-local.log`、`confirmation-dev.log`、`frontend-repro.log`、`frontend-dev.log`。调试临时测试加载路径的失败不计入以上业务对照结果。

结束前再次执行 `git ls-remote origin refs/heads/Dev`，远端仍为上述 `f6f85d0`。

## 后续处理

需要的是保留本地文档和工作记录后，将本地代码同步到已经完成合并的 Dev；不需要再次把旧代码向 Dev 提 PR。本轮仅审核并记录，没有切换分支、覆盖工作区或重启服务。

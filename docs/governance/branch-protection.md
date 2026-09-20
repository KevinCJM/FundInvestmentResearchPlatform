# 分支保护实现

需求以 [提交规范](branch-submission-rules.md) 为准，执行顺序见 [代码提交全流程](submission-workflow.md)。本文区分 GitHub 已有原生配置与执行者必须遵守的审核规范；文档变更本身不部署远端规则。

| 规则 | 配置文件 | 范围 | 豁免 |
| --- | --- | --- | --- |
| 必须 PR、禁止强推/删除、仅 merge commit | `.github/branch-ruleset.json` | main、Dev | 无 |
| 必须 1 个有效 Approve，新提交撤销旧批准 | `.github/approval-ruleset.json` | main、Dev | 仅 KevinCJM，限 PR 合并 |
| 必须通过 branch-policy | `.github/main-branch-ruleset.json` | main | 无 |

原生批准豁免按合并操作者判断。Owner 可以免普通 Approve 合并自己的或其他人的 PR；普通协作者仍需有效批准。规则固定到 KevinCJM 的用户 ID，不自动惠及其他管理员。该原生能力不表示 AI 已获准绕过 Bot 审核。

**Bot 审核是提交规范中的强制合并条件。** 正常合并必须确认指定 Bot 已通过当前最新 HEAD；Owner 可作例外，AI 代为执行则必须先取得本次人类明确允许，并逐条回复 Bot 的意见及代码/测试依据。是否拥有 Owner 凭据不能代替这项允许。

**技术覆盖边界：** 现有三组 ruleset 和下述 `branch-policy` 工作流没有解析 Bot 的 review、评论或审核提交，也没有独立的 `bot-review-gate` required check。它们可能在 Bot 尚未通过时允许有权限者操作，但执行者仍不得违反提交规范。当前用规范约束 AI 主动请求人类允许，不建设用于区分 AI/人类身份的技术系统；不得声称这次文档更新已经部署自动 Bot 门禁。

## 最小工作流

`.github/workflows/branch-policy.yml` 在 PR 打开、更新、重开、就绪或修改目标时执行。main 必须来自同一仓库的 Dev，其他目标通过。不检出代码、权限为空、无需额外凭据或 Python 校验框架。

工作流从开发分支通过 PR 合入 Dev；后续 Dev → main PR 自带该工作流并触发检查，无须为安装检查而立即发布 Dev 中其他业务变更。Owner 豁免不能让没有必需检查的 main PR 通过。

## 终端部署

使用已有 gh 授权。共享 PR 规则 ID 为 22899065，main 方向规则 ID 为 23132154。先按名称查找批准规则，已存在则 PUT 更新对应 ID，不重复创建。

```bash
gh api repos/KevinCJM/FundInvestmentResearchPlatform/rulesets
# 首次安装批准规则：先创建，再从共享规则中移出批准要求，避免普通协作者出现无批准窗口
gh api --method POST repos/KevinCJM/FundInvestmentResearchPlatform/rulesets --input .github/approval-ruleset.json
# 后续维护改为 PUT .../rulesets/查询到的批准规则ID

gh api --method PUT repos/KevinCJM/FundInvestmentResearchPlatform/rulesets/22899065 --input .github/branch-ruleset.json
gh api --method PUT repos/KevinCJM/FundInvestmentResearchPlatform/rulesets/23132154 --input .github/main-branch-ruleset.json

gh api repos/KevinCJM/FundInvestmentResearchPlatform/rules/branches/main
gh api repos/KevinCJM/FundInvestmentResearchPlatform/rules/branches/Dev
# 逐组读取，核对 bypass_actors 和 current_user_can_bypass
gh api repos/KevinCJM/FundInvestmentResearchPlatform/rulesets/实际ID
```

批准规则仅放 `pull_request` 及批准参数；不能把必需检查、禁止强推/删除等规则混进去。Owner 使用终端 `gh pr merge --merge --admin --match-head-commit <SHA>` 行使已配置的 PR 内豁免前，AI 须确认 Bot 已通过，或已完成提交规范中的人类授权、逐条回复和最终核对。不能借此绕过其他规则，未经用户授权不得扩大豁免账号。

## 验证与权限边界

- 核对规则范围、批准数和唯一豁免用户；PR-only 与 main 方向规则的 bypass 必须为空。
- 执行实际 YAML 的 main/Dev 来源正反例，并用 actionlint 校验。
- 通过 GitHub 实际 PR 检查验证工作流。原生 Owner 无普通 Approve 的合并能力验证，须使用 Bot 已通过或已获本次例外允许的 PR；不得为了测试权限直接绕过 Bot，也不进行真实主线推送测试。
- GitHub 原生检查按名称和 App 匹配；本方案不隔离恶意管理员或刻意伪造同名 Actions 检查的写权限人。工作流修改须检查真实差异；只读配置验证不能冒充另一个账号的实际测试。

GitHub API 支持指定用户及仅 PR 的规则豁免：[REST rulesets](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset)。

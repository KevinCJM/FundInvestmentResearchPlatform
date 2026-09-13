# GitHub 提交门禁设计

需求由 [branch_submission_rules.md](../branch_submission_rules.md) 统一规定。本设计实现 `branch-policy`、`ai-review`、`quality-gate` 三个独立检查，保留独立 Codex 审核，不要求人类 approval。

## 执行链路与权限

1. PR、评论、main/Dev 推送、质量测试完成、手动重检及定时补查触发 `.github/workflows/ai-review.yml`。
2. **只有当前 main 版本负责发布**。Dev 事件仅转发到 main；发布任务 checkout `github.workflow_sha` 的三个校验脚本，读取 GitHub API，不执行候选代码或 artifact。
3. `scripts/check_submission.py` 获取所有 open PR 的实时来源 ref、目标 ref、完整 SHA 和审核证据，先将三项检查置为 pending，再分别发布结果。
4. 普通 `pull_request` 运行 `.github/workflows/quality-gate.yml`，候选代码在无发布凭据、只读 token、无持久化 checkout 凭据的独立 runner 上测试。禁止以持有写权限的 `pull_request_target` 执行候选代码。
5. 发布端核验质量运行的仓库、事件、工作流路径、精确 PR/HEAD/base 标题、完整 job 集合、最新 attempt，以及工作流 Git blob 与受保护目标版本一致。修改了质量工作流的 PR 不可自证；改由受保护目标分支 dispatch 相同 HEAD/base 的测试。
6. 发布结果写到 PR 的真实 HEAD，不使用 Actions 测试 merge SHA。最后重新读取 PR、源/目标 ref 和 main 策略 SHA，版本变化时不发布旧成功。旧 main 任务停止写入，由当前 main 重新检查。

发布 job 的 `GITHUB_TOKEN` 为 contents/read、pull-requests/read、issues/read、checks/read、actions/write；最后一项用于 dispatch 受保护测试工作流。Checks write 使用独立的短期 GitHub App 安装 token。

专用 App 仅安装到本仓库，仅启用 Checks write 和必需的 Metadata read。Environment `ai-review-publisher` 仅允许 **branch 类型的精确 main**，无 tag、开发分支或人类 reviewer。App ID 存变量 `AI_REVIEW_APP_ID`，PEM 存 secret `AI_REVIEW_APP_PRIVATE_KEY`；不能复制到仓库级 secrets，也不能用管理员 PAT 替代。

现有 Codex App 负责审核，发布 App 只认证并写入检查。Ruleset 的三项检查必须绑定专用发布 App 的实际 integration ID，不能绑定通用 GitHub Actions App `15368`。修改受保护工作流、App 配置和 ruleset 的管理员权限属于系统信任边界。

## 分支检查

`scripts/submission_policy.py` 的 `branch_decision` 检查：

- PR 为 open、非 Draft，同一仓库；普通开发或同步分支只能进入 Dev。
- main 仅接受本仓库 Dev；`main → Dev`、`release/* → main` 等方向均拒绝。
- PR HEAD 等于实时来源 ref，来源包含实时目标 SHA。
- 同一 HEAD 不得同时被多个面向 main/Dev 的 open PR 使用。GitHub 检查按 commit 绑定，不能让一个 PR 的 success 覆盖另一个 PR 的 failure。

合并执行者仍须最后核对 HEAD/base，并使用 HEAD 匹配接口。事件与 API 更新不是原子操作；strict required checks 补充最新基线约束。本设计不支持 merge queue。

## AI 审核证据

`scripts/check_ai_review.py` 使用已由真实 API 核实的身份：Codex App ID `1144995`、bot user ID `199175422`、login `chatgpt-codex-connector[bot]`。Issue comment 还须验证 `performed_via_github_app.id`；显示名、任意 bot 或作者复制的 PASS 不是可信证据。

每次观察上下文记录 PR、base_ref、base_sha、head_sha、policy_sha、observed_at，仓库为固定常量。HEAD、base 或 main 策略变化后，旧上下文不复用。支持以下两类通过证据：

1. 官方 Codex 发布的 `codex-review/v1` 结构化报告：唯一顶层 JSON 代码块、字段和完整版本精确匹配、PASS、findings 和 limitations 均为空；reviewer_identity 必须为官方 bot login，并包含非空 review_run_id、reviewed_scope 和 HTTPS evidence 列表。引用中的报告、多份矛盾报告、被编辑的 issue comment 或与 CHANGES_REQUESTED 冲突的 PASS 不放行。
2. 官方原生输出：当前 HEAD 的 Code Review 摘要 Completed，短 SHA 经 GitHub API 唯一解析；并有晚于本次精确版本审核请求的新官方 👍，或未经编辑、明确绑定当前提交的官方 “Didn't find any major issues” 评论。摘要和正面结论须晚于最新请求、观察边界及最新问题，且无更晚的阻断审核。

Completed 本身、旧 👍、普通 PR reaction、无评论、超时均不代表通过。所有官方行内问题须处理并复核，outdated 不豁免；仅关闭讨论不构成新审核结论。格式无法识别时失败关闭，不靠宽泛关键词猜测。

等待新 `ai-review` 上下文后，获授权的提交 AI 生成并原样发布审核请求：

```bash
python3 scripts/check_ai_review.py --pr 123 --request-body > /tmp/codex-review-request.md
gh pr comment 123 --repo KevinCJM/FundInvestmentResearchPlatform --body-file /tmp/codex-review-request.md
```

请求带 `@codex review`、完整 HEAD/base 和不可编辑的隐藏版本标记。版本变化时发布新请求，禁止编辑旧请求。生产工作流不自动发送评论或请求模型，避免重复消耗审核额度；定时补查只读取证据。

只读查询使用 `python3 scripts/check_ai_review.py --pr 123`，默认建立新的观察时间。复核已有证据时，从可信检查读取真实 observed_at、HEAD/base，传给 `--observed-at`、`--expected-head`、`--expected-base`。此脚本不能发布检查；唯一发布入口是受保护的 `scripts/check_submission.py --publish`。

## 质量检查与适用范围

`scripts/ci_quality.py` 从受保护目标版本执行，通过不可变 `compare/{base}...{head}` 选择测试；不读取会随 PR 变化的文件列表。改名同时考虑新旧路径；比较达到 GitHub 的 300 文件上限时保守运行全部业务套件。

| 改动 | 必须执行 |
| --- | --- |
| 所有 PR | 受保护版本的 AI Hermes 校验、R03 路由、覆盖检查、diff 格式检查；候选门禁单元测试 |
| 纯文档、路由 JSON、门禁治理 | 上述治理检查；业务任务由可信计划明确标记不适用 |
| 前端或设计/i18n 检查器 | 全量 Vitest、TypeScript、构建、i18n、设计检查，以及全部 Playwright |
| 后端或 .github/requirements-ci.txt 依赖锁定 | 后端全量 pytest、全部 Playwright |
| 未归类的执行代码/配置/数据，或包含业务变更的 Dev → main | 前后端及 Playwright 全部执行 |

工作流固定包含 prepare、governance、policy-tests、frontend、backend、e2e、quality-result 七个 job。汇总实际执行并核对所有适用 job 成功；缺失、失败、取消、超时或意外跳过均阻断。仅可信计划判定不适用的三个业务子任务可 skipped；三项顶层检查不能 skipped/neutral。

治理校验器来自受保护 base，并与候选单元测试分处不同 runner，防止候选测试修改校验器。候选 policy-tests runner 还对 skills 中全部 Python 文件做编译检查，并实际执行候选 validate、R02/R03 route 和 evolve 覆盖回归，不能只测试旧版路由工具。业务测试执行待合入 HEAD；来源必须包含 base，因此该 HEAD 已包含目标代码。

CI 使用 Node 22、Python 3.12，后端依赖由 `backend/requirements.txt` 与 `.github/requirements-ci.txt` 共同安装。后端单元测试禁止外网 socket，允许本地回环和 Unix socket；数据及 Numba 缓存使用临时目录，Tushare token 为空。Playwright 使用本地测试服务，同时设置 INDICATOR_TEST_PYTHON 与 TEST_PYTHON；真实数据写入用例仍按其既有显式授权开关跳过，不能报告为正式数据验收。首次真实运行暴露的环境或既有失败必须调查，不能新增跳过来换取绿色。

候选单元测试通过本身不能授予 quality-gate；发布端只接受受保护定义和精确 job 结果。API 分页超限、格式无法核实或超时均失败关闭，不保留假成功。

## 首次安装与验收

初始化授权和限制以提交规范第 12 节为准，本设计不另行授予例外。

1. 隔离分支完成实现、失败案例、独立 AI 审核和记录；开发分支 → Dev → main 均走普通 PR，不直推、不用管理员 bypass、不关闭现有保护。
2. 配置专用 App 和 main-only Environment。缺 App ID/私钥或 App 未安装时不得声称门禁完成，也不安装无法产出结果的占位必需检查。
3. base 尚无质量工作流时，首次 PR 无法读取可信计划。只有明确授权且满足第 12 节的首次安装可以按初始化记录推进；启动失败不能伪装成生产 success，业务测试失败不得豁免。
4. 工作流进入 main 后，真实 PR 验证正确方向/最新版本可通过，错误方向、缺审核、旧版本和不可信质量结果均阻断。分别核验普通 Actions 测试与专用 App 三项检查。
5. 完成真实验证后，将三项检查按专用 App ID 绑定到 ruleset，开启 strict latest-base，保留 PR-only、禁止强推/删除、空 bypass、0 原生 approval、merge commit。
6. 重新读取远端规则、检查发布者和 PR 状态，确认 main、Dev 均命中约束。发布 merge commit 经专用同步分支和 PR 回到 Dev。

创建 App 的入口为 [GitHub Apps](https://github.com/settings/apps/new)。采用 private、仅当前账户、禁用 webhook、仅 Checks write，安装时只选本仓库。私钥仅经本机受控文件/标准输入进入 GitHub Environment，不能进入聊天、Git、日志。终端仓库授权不等于浏览器已登录。

## 本地验证

按项目要求选择 Python 环境，先安装已声明的 CI 测试依赖（包含 PyYAML），再执行验证：

```bash
python3 -m pip install -r backend/requirements.txt -r .github/requirements-ci.txt
python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v
python3 skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
python3 skills/ai-hermes-self-evolve/scripts/route_task.py --route-id R03 --mode context
python3 skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py --diff-range origin/Dev...HEAD --json
```

另使用官方 actionlint 验证三个 YAML 的表达式、事件和语法。离线测试不证明 GitHub 安装或业务回归通过；当前远端阶段、SHA、审核与运行证据记录在安装 PR，不把易过期状态作为永久规范。

官方依据：[Environment 分支限制](https://docs.github.com/en/rest/deployments/branch-policies)、[App token](https://github.com/actions/create-github-app-token)、[Checks API](https://docs.github.com/en/rest/checks/runs)、[必需检查来源](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)、[工作流事件](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)。

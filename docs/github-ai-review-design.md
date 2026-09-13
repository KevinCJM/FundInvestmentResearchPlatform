# GitHub 提交门禁设计

需求由 [branch_submission_rules.md](../branch_submission_rules.md) 统一规定。本设计实现 `branch-policy`、`ai-review`、`quality-gate` 三个独立检查，保留独立 Codex 审核，不要求人类 approval。

## 执行链路与权限

1. PR、评论、main/Dev 推送、质量测试完成、手动重检及定时补查触发 `.github/workflows/ai-review.yml`。
2. **只有当前 main 版本负责发布**。Dev 事件仅转发到 main；发布任务 checkout `github.workflow_sha` 的三个校验脚本，读取 GitHub API，不执行候选代码或 artifact。
3. `scripts/check_submission.py` 获取所有 open PR 的实时来源 ref、目标 ref、完整 SHA 和审核证据，先将三项检查置为 pending，再分别发布结果。
4. 普通 `pull_request` 运行 `.github/workflows/quality-gate.yml`，候选代码在无发布凭据、只读 token、无持久化 checkout 凭据的独立 runner 上测试。禁止以持有写权限的 `pull_request_target` 执行候选代码。
5. 发布端核验质量运行的仓库、事件、工作流路径、精确 PR/HEAD/base 标题、完整 job 集合、最新 attempt，以及工作流 Git blob 与受保护目标版本一致。修改了质量工作流的 PR 不可自证；改由受保护目标分支 dispatch 相同 HEAD/base 的测试。
6. 发布结果写到 PR 的真实 HEAD，不使用 Actions 测试 merge SHA。最后重新读取 PR、源/目标 ref 和 main 策略 SHA，版本变化时不发布旧成功。旧 main 任务停止写入，由当前 main 重新检查。

发布 job 仅使用 GitHub Actions 内置的短期 `GITHUB_TOKEN`，权限为 contents/read、pull-requests/read、issues/read、checks/write、actions/write；后者用于 dispatch 受保护测试工作流。不需要新建 App、私钥、额外 secret、Environment 或浏览器登录，禁止上传个人管理员 PAT。已有官方 Codex App 继续负责独立审核。

Ruleset 将三个检查绑定 GitHub Actions 的实际 integration ID（当前为 `15368`）。GitHub 原生规则只能识别 App 和检查名称，不能区分同一 App 下的工作流；有仓库写权限的人可以提交另一个申请 checks/write 的工作流。因此不能仅凭检查颜色、details_url、external_id 或 output 声称防止伪造。

`scripts/check_ai_review.py` 通过 Actions API 选择当前 main SHA、固定路径、允许事件的真实已完成发布运行，从对应 attempt 的 `publish` job 日志读取上下文与结果，不从 check output 继承 observed_at。其他工作流不能把内容追加到这个 job 的日志。日志缺失则失败关闭；当前策略没有先前记录时创建新的观察边界。

终端 `scripts/check_submission.py --verify` 校验本机三个验证脚本的 Git blob 与远端当前 main 一致，读取上述真实发布记录，再重新采集并计算分支、官方 AI 审核和质量证据，最后核对公开检查及 HEAD/base/策略未变化。执行者必须使用下文从 main 提取的隔离脚本，不能由候选 PR 自证。此命令是 AI 的执行协议，GitHub 不会强制其他客户端调用；原生保护与终端补充验证的边界必须如实报告。

## 分支检查

`scripts/submission_policy.py` 的 `branch_decision` 检查：

- PR 为 open、非 Draft，同一仓库；普通开发或同步分支只能进入 Dev。
- main 仅接受本仓库 Dev；`main → Dev`、`release/* → main` 等方向均拒绝。
- PR HEAD 等于实时来源 ref，来源包含实时目标 SHA；codex/sync-main-* 同步分支还须包含实时 main，不能用只包含 Dev 的空同步通过。
- 同一 HEAD 不得同时被多个面向 main/Dev 的 open PR 使用。GitHub 检查按 commit 绑定，不能让一个 PR 的 success 覆盖另一个 PR 的 failure。

合并执行者仍须最后核对 HEAD/base，并使用 HEAD 匹配接口。事件与 API 更新不是原子操作；strict required checks 补充最新基线约束。本设计不支持 merge queue。

## AI 审核证据

`scripts/check_ai_review.py` 使用已由真实 API 核实的身份：Codex App ID `1144995`、bot user ID `199175422`、login `chatgpt-codex-connector[bot]`。Issue comment 还须验证 `performed_via_github_app.id`；显示名、任意 bot 或作者复制的 PASS 不是可信证据。

每次观察上下文记录 PR、base_ref、base_sha、head_sha、policy_sha、observed_at，仓库为固定常量。HEAD、base 或 main 策略变化后，旧上下文不复用。支持以下两类通过证据：

1. 官方 Codex 发布的 `codex-review/v1` 结构化报告：唯一顶层 JSON 代码块、字段和完整版本精确匹配、PASS、findings 和 limitations 均为空；reviewer_identity 必须为官方 bot login，并包含非空 review_run_id、reviewed_scope 和 evidence 列表。每条 evidence 必须精确对应 GitHub API 已抓取的本仓库、本 PR 官方 review/comment，核对当前提交及时间；issue comment 还必须未经编辑并在正文绑定当前版本。任意 HTTPS、外部网站、不存在记录、其他 PR、过期审核及可变摘要链接不作为结构化审核证据。引用中的报告、多份矛盾报告、被编辑的 issue comment 或与 CHANGES_REQUESTED 冲突的 PASS 不放行。
2. 官方原生输出：当前 HEAD 的 Code Review 摘要 Completed，短 SHA 经 GitHub API 唯一解析；并有晚于本次精确版本审核请求的新官方 👍，或未经编辑、明确绑定当前提交的官方 “Didn't find any major issues” 评论。摘要和正面结论须晚于最新请求、观察边界及最新问题，且无更晚的阻断审核。

Completed 本身、旧 👍、普通 PR reaction、无评论、超时均不代表通过。所有官方行内问题须处理并复核，outdated 不豁免；仅关闭讨论不构成新审核结论。格式无法识别时失败关闭，不靠宽泛关键词猜测。

等待新 `ai-review` 上下文后，获授权的提交 AI 生成并原样发布审核请求：

```bash
python3 scripts/check_ai_review.py --pr 123 --request-body > /tmp/codex-review-request.md
gh pr comment 123 --repo KevinCJM/FundInvestmentResearchPlatform --body-file /tmp/codex-review-request.md
```

生成器先比对 PR 元数据和实际来源 ref；刚 push 后元数据滞后会拒绝输出，刷新后再执行，不得发布旧请求。请求带 `@codex review`、完整 HEAD/base 和不可编辑的隐藏版本标记。版本变化时发布新请求，禁止编辑旧请求。生产工作流不自动发送评论或请求模型，避免重复消耗审核额度；定时补查只读取证据。

只读查询使用 `python3 scripts/check_ai_review.py --pr 123`，默认建立新的观察时间。复核已有证据时，从真实受保护发布 job 日志读取 observed_at、HEAD/base，传给 `--observed-at`、`--expected-head`、`--expected-base`。此脚本不能发布检查；唯一发布入口是受保护的 `scripts/check_submission.py --publish`。正式合并前使用下文 `--verify`，不手填观察时间替代生产证据。

## 质量检查与适用范围

`scripts/ci_quality.py` 从受保护目标版本执行，通过不可变 `compare/{base}...{head}` 选择测试；不读取会随 PR 变化的文件列表。改名同时考虑新旧路径；比较达到 GitHub 的 300 文件上限时保守运行全部业务套件。

| 改动 | 必须执行 |
| --- | --- |
| 所有 PR | 受保护版本的 AI Hermes 校验、R03 路由、覆盖检查、diff 格式检查；候选门禁单元测试及受保护版本的契约测试 |
| 纯文档、路由 JSON、门禁治理 | 上述治理检查；业务任务由可信计划明确标记不适用 |
| 前端或设计/i18n 检查器 | 全量 Vitest、TypeScript、构建、i18n、设计检查，以及全部 Playwright |
| 后端或 .github/requirements-ci.txt 依赖锁定 | 后端全量 pytest、全部 Playwright |
| 未归类的执行代码/配置/数据，或包含业务变更的 Dev → main | 前后端及 Playwright 全部执行 |

工作流固定包含 prepare、governance、policy-tests、protected-policy-tests、frontend、backend、e2e、quality-result 八个 job。汇总实际执行并核对所有适用 job 成功；缺失、失败、取消、超时或意外跳过均阻断。仅可信计划判定不适用的三个业务子任务可 skipped；三项顶层检查不能 skipped/neutral。

protected-policy-tests 从受保护 base 读取不可由本 PR 删改的测试源，通过固定 FIRP_GATE_ROOT 指向候选门禁模块及配置，并拒绝空测试集；候选新增测试另行执行且同样必须非空，防止合入空套件后锁死后续提交。在导入候选代码或运行任何候选测试之前，protected-policy-tests 先由受保护工作流步骤读取 base 的 .github/workflow-contracts.json，校验两份候选生产工作流的完整规范化配置 SHA256，覆盖事件、权限、ref、计划、测试、汇总、运行目录、环境、条件及失败传播；注释和无语义的 YAML 排版不影响指纹。不能用 true、echo、if:false 或 continue-on-error 绕过业务、治理或契约测试。这样删除候选测试不能使回归缺陷被零测试成功掩盖。治理校验器来自受保护 base，并与候选单元测试分处不同 runner，防止候选测试修改校验器。候选 policy-tests runner 先对 skills 中全部 Python 文件做编译检查，并实际执行候选 validate、R02/R03 route 和 evolve 覆盖回归，并在任何候选单元测试之前执行这些检查，避免测试修改受检文件；不能只测试旧版路由工具。业务测试执行待合入 HEAD；来源必须包含 base，因此该 HEAD 已包含目标代码。

CI 使用 Node 22、Python 3.12，后端依赖由 `backend/requirements.txt` 与 `.github/requirements-ci.txt` 共同安装。后端单元测试禁止外网 socket，允许本地回环和 Unix socket；数据及 Numba 缓存使用临时目录，Tushare token 为空。Playwright 使用本地测试服务，同时设置 INDICATOR_TEST_PYTHON 与 TEST_PYTHON；真实数据写入用例仍按其既有显式授权开关跳过，不能报告为正式数据验收。首次真实运行暴露的环境或既有失败必须调查，不能新增跳过来换取绿色。

候选契约 JSON 也须包含实际候选工作流的合法指纹，避免合入空表或遗漏现行版本后锁死后续 PR；候选自洽检查不代替 base 的授权校验。

修改生产工作流时，先让独立 AI 审查拟议的完整配置及验证证据，在第一阶段 PR 中将拟议规范化指纹加入受保护契约，同时保留现行指纹且保持生产 YAML 不变。第一阶段经 Dev（必要时 main）生效后，第二阶段 PR 才应用已审核的配置；最后删除不再使用的旧指纹。工作流行为改变仍须执行对应实际验证；不能把未审核的任意指纹当成兼容列表，不能在同一 PR 改 YAML 并自行改断言来认证自己。涉及发布器/质量 API 的不兼容变化还须分阶段保持在用协议兼容。

候选单元测试通过本身不能授予 quality-gate；发布端只接受受保护定义和精确 job 结果。API 分页超限、格式无法核实或超时均失败关闭，不保留假成功。

## 首次安装与验收

初始化授权和限制以提交规范第 12 节为准，本设计不另行授予例外。

1. 隔离分支完成实现、失败案例、独立 AI 审核和记录；开发分支 → Dev → main 均走普通 PR，不直推、不用管理员 bypass、不关闭现有保护。
2. 使用已有终端授权检查仓库管理权限和 Actions 可用性；工作流自身使用内置 token，无须新增账号或凭据。先保证真实结果能产生，再配置必需检查。
3. base 尚无质量工作流时，首次 PR 无法读取可信计划。只有明确授权且满足第 12 节的首次安装可以按初始化记录推进；启动失败不能伪装成生产 success，业务测试失败不得豁免。
4. 工作流进入 main 后，真实 PR 验证正确方向/最新版本可通过，错误方向、缺审核、旧版本和不可信质量结果均阻断。分别核验隔离测试、真实 main 发布记录与三项检查。
5. 完成真实验证后，将三项检查按实际 GitHub Actions App ID 绑定到 ruleset，开启 strict latest-base，保留 PR-only、禁止强推/删除、空 bypass、0 原生 approval、merge commit。
6. 重新读取远端规则、检查发布者和 PR 状态，确认 main、Dev 均命中约束。发布 merge commit 经专用同步分支和 PR 回到 Dev。

## 终端操作

所有命令使用现有 `gh` 授权，不要求浏览器登录。先读取仓库权限、当前规则及 PR：

```bash
gh api repos/KevinCJM/FundInvestmentResearchPlatform --jq '{permissions,default_branch}'
gh api repos/KevinCJM/FundInvestmentResearchPlatform/rulesets
gh pr view 123 --repo KevinCJM/FundInvestmentResearchPlatform --json state,isDraft,headRefOid,baseRefName,statusCheckRollup
gh workflow run ai-review.yml --repo KevinCJM/FundInvestmentResearchPlatform --ref main
gh run list --repo KevinCJM/FundInvestmentResearchPlatform --workflow ai-review.yml
```

首次验收完成后，复核已有 ruleset 与版本化配置的差异，保留更严格且无冲突的已有规则，再更新实际 ruleset ID（本仓库当前为 `22899065`）：

```bash
gh api repos/KevinCJM/FundInvestmentResearchPlatform/rulesets/22899065
gh api --method PUT repos/KevinCJM/FundInvestmentResearchPlatform/rulesets/22899065 --input .github/branch-ruleset.json
```

以下示例从当前 main 提取验证器，执行完整验证并在成功后立即匹配 HEAD 合并；将 `123` 替换为实际 PR。base 在验证期间变化会失败，之后变化还受 GitHub strict latest-base 约束。失败后重新核对，不能加 `--admin` 重试。

```bash
set -e
submission_pr=123
submission_repo=KevinCJM/FundInvestmentResearchPlatform
git fetch origin main Dev
submission_dir=$(mktemp -d /tmp/firp-submit.XXXXXX)
git archive origin/main scripts/check_submission.py scripts/check_ai_review.py scripts/submission_policy.py | tar -x -C "$submission_dir"
submission_head=$(gh pr view "$submission_pr" --repo "$submission_repo" --json headRefOid --jq .headRefOid)
submission_base_ref=$(gh pr view "$submission_pr" --repo "$submission_repo" --json baseRefName --jq .baseRefName)
submission_base=$(gh api "repos/$submission_repo/git/ref/heads/$submission_base_ref" --jq .object.sha)
python3 "$submission_dir/scripts/check_submission.py" --verify --pr "$submission_pr" --expected-head "$submission_head" --expected-base "$submission_base"
gh pr merge "$submission_pr" --repo "$submission_repo" --merge --match-head-commit "$submission_head"
gh pr view "$submission_pr" --repo "$submission_repo" --json state,mergedAt,mergeCommit
```

不要在启用 `set -e` 失败退出的脚本外盲目单独补跑 merge。提交 AI 仍需满足规范第 9 节的授权、范围、最新规则与合并后核实要求。

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

官方依据：[内置工作流授权](https://docs.github.com/en/actions/security-for-github-actions/security-guides/automatic-token-authentication)、[Checks API](https://docs.github.com/en/rest/checks/runs)、[必需检查来源](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)、[工作流事件](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)。

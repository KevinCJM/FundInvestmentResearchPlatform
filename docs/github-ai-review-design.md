# GitHub Codex AI 审核门禁

需求依据：[branch_submission_rules.md](../branch_submission_rules.md)。本设计只实现 `ai-review` 及其自身测试，不把门禁脚本测试当成业务 `quality-gate`。

## 数据流

PR 更新 / Codex 摘要评论变化 / main、Dev 更新 / 手动重检 / 定时补查 → 受保护版本的工作流 → GitHub API → 身份、HEAD、base 和问题校验 → PR 实际 HEAD 上的 `ai-review` Check Run。

- `.github/workflows/ai-review.yml` 发布门禁，只 checkout `github.workflow_sha` 的校验脚本，不拉取或执行 PR 代码；不持久化 checkout 凭据。
- `scripts/check_ai_review.py` 使用 GitHub CLI 与只读 `GITHUB_TOKEN` 收集证据；写检查使用单独的 GitHub App 安装 token。不调用模型 API，不读取项目正式数据，不需要 OpenAI API Key。
- `.github/workflows/ai-review-tests.yml` 在普通 PR 的隔离、只读 job 中测试候选校验器和路由。它不持有 `checks:write`，不能自行发布通过结果。
- `scripts/tests/test_ai_review.py` 使用纯本地夹具验证放行/阻断逻辑。
- 路由中的 Python 回归命令统一经 `python3` 从当前环境解析，移除只在开发者 Mac 存在的解释器绝对路径。执行业务回归前仍须按 AGENTS.md 激活推荐虚拟环境；CI 的路由校验只验证引用和命令契约，不代表执行全部业务测试。

## 信任和版本

审核身份由 PR #10 的真实 API 元数据核实：Codex App ID 为 `1144995`；bot user ID 为 `199175422`，login 为 `chatgpt-codex-connector[bot]`。摘要 issue comment 同时校验 `performed_via_github_app.id`，正式 review 和行内 comment 校验 GitHub 返回的稳定 user ID。

检查由专用 GitHub App 发布，ID 从受保护 Environment 的 `AI_REVIEW_APP_ID` 读取；现有 Codex App 继续只负责审核。不能绑定通用 GitHub Actions App（`15368`），否则其他开发分支工作流也可能发布同名检查。

Environment `ai-review-publisher` 只允许 **branch 类型**的精确 `main`、`Dev`，不允许 tag 或开发分支，不设置人类 reviewer。私钥只放在该 Environment，禁止复制到 repository/organization secrets。发布 job 不 checkout PR、执行候选代码或消费候选 artifact；普通 PR job 没有 App 凭据。专用 App 只安装在此仓库，只授予 Checks write 和 GitHub 必需的 Metadata read。

每个检查保存 PR 编号、完整 HEAD/base SHA、base ref 和首次观察时间。复用上下文前核对专用 App、关联运行的 workflow 路径、事件与仓库；`pull_request_target` 的执行 ref 由 Environment 和 job 条件限制，不能把运行元数据里的 PR 源分支误当执行来源。仓库管理员修改规则/凭据的权限属于信任边界；对受保护工作流或凭据使用方式的变更也必须独立 AI 审核。

## 放行条件

来源必须为本仓库，目标为 Dev/main；进入 main 的来源只能为 Dev；来源须包含最新目标分支。HEAD 或 base 变化后，旧上下文不再有效。

GitHub 必需检查绑定 commit SHA，不按 PR 的 `external_id` 隔离。同一 HEAD 同时用于多个目标为 main/Dev 的 open PR 时，全部阻断；先关闭重复 PR 再重新审核，避免一个 PR 的 success 覆盖另一个 PR 的 failure。

支持两种证据：

1. 官方 Codex 发布的结构化 JSON，schema 为 `codex-review/v1`，包含 repository、pr_number、base_ref、base_sha、head_sha、conclusion、findings、limitations。所有版本字段精确匹配，结论必须为 PASS，findings 和 limitations 均为空。
2. 现有 Codex 原生输出：官方摘要里的 Code Review 已 Completed，摘要 commit 通过 GitHub API 唯一解析为当前完整 HEAD，完成时间不早于本次上下文与最新问题；同时官方 bot 在下述精确版本审核请求上给出新 👍，且没有更晚的审核或未解决问题。

旧 👍、Completed 单独出现、普通用户粘贴 PASS、作者自评均不通过。原生无问题结论是多项证据联合判定，不能把某个表情单独当作 approval。官方行内问题即使 outdated 也不能跳过；关闭讨论后，仍须有晚于问题的新独立通过证据。

首次启用及 HEAD/base 变化后，先等待新 `ai-review` 上下文，再由获授权的提交 AI 生成请求正文：

```bash
python3 scripts/check_ai_review.py --pr 123 --request-body > /tmp/codex-review-request.md
gh pr comment 123 --repo KevinCJM/FundInvestmentResearchPlatform --body-file /tmp/codex-review-request.md
```

该正文含 `@codex review`、完整 HEAD/base 和固定审查要求。必须原样发布为新评论；不编辑已有请求，不删改隐藏版本标记。门禁只接受创建时间不早于当前上下文、从未编辑且版本精确匹配的请求上的官方 👍；普通 PR reaction 不用于放行。工作流不自动发送评论，避免重复消耗审核额度。Codex 未给出可绑定证据时保持 pending，可请求其按结构化契约重新报告，不能由开发者自己粘贴 PASS。

## 状态与运行

- success：已取得当前版本的有效通过证据。
- failure：分支/版本不合法、存在未解决问题或明确阻断/不完整报告。
- in_progress：等待当前版本的有效审核证据。服务/API 错误不得产生 success。

查看 GitHub 当前检查状态：

```bash
gh pr checks 123 --repo KevinCJM/FundInvestmentResearchPlatform
```

下列只读命令建立**新的观察边界**，用于开始一次证据采集；它不会复用线上上下文，不能用它查询此前审核是否通过：

```bash
python3 scripts/check_ai_review.py --pr 123
```

复核已有证据时，从可信 `ai-review` Check Run 的 `output.text.context` 读取真实 `observed_at`、`head_sha`、`base_sha`，分别传入 `--observed-at`、`--expected-head`、`--expected-base`。这些参数仅用于只读验收；`--publish` 禁止回填观察时间或仅扫描指定 PR。定时补查用于补足 reaction/讨论状态无直接可用触发事件的情况，不自动发起新审核，不发送评论。

发布 job 的 `GITHUB_TOKEN` 只有 contents/read、pull-requests/read、issues/read、actions/read、checks/read；短期 App token 只有 checks/write。所有 API 分页完整读取，超出明确上限、字段格式变化或超时都失败关闭；单个 PR 出错不跳过其他 PR。先创建 pending 检查，再更新同一检查为最终状态，避免留下孤立 pending。

全仓库串行执行，每个事件扫描全部 open PR，避免 pending run 替换丢失其他 PR 的 base 重评。GitHub 事件、定时任务和检查发布非原子，可能延迟；合并执行者仍须最后核对 HEAD/base，并启用 strict required checks。此适配器尚不支持 merge queue。

## 专用发布 App 的一次性接入

1. 在 [GitHub App 新建页面](https://github.com/settings/apps/new) 创建私有 App，例如 `FIRP AI Review Publisher`；主页填本仓库地址，取消 Active webhook，不配置 OAuth 回调，选择 Only on this account。
2. Repository permissions 仅启用 Checks: Read and write；Metadata: Read-only 为 GitHub 必需项。只安装到 FundInvestmentResearchPlatform，不选择全部仓库。
3. 将 App ID 写入 [ai-review-publisher 环境](https://github.com/KevinCJM/FundInvestmentResearchPlatform/settings/environments) 的变量 `AI_REVIEW_APP_ID`。
4. 由账号持有人生成私钥，并在同一 Environment 新建 secret `AI_REVIEW_APP_PRIVATE_KEY`，保存完整 PEM。不要把私钥发到聊天、提交到 Git 或写入日志。无需 OpenAI API Key，也不要把管理员 PAT 作为替代。
5. 工作流通过固定版本 `actions/create-github-app-token` 创建仅本仓库、仅 checks/write 的安装 token，任务结束撤销。缺 ID/私钥时工作流失败，不能启用占位的 required check。

## 首次上线顺序

1. 从 origin/Dev 创建开发分支，提交规则、实现和测试，通过 PR 进入 Dev，再由 Dev 发布到 main；保留现有 PR 保护，不直推、不使用管理员绕过。
2. 初始目标分支没有工作流时，候选 PR 测试可运行，但评论驱动的生产门禁还不能运行。先完成离线测试、独立 AI 审核和人工执行者的版本核验；这只是上线准备，不冒充生产 ai-review。
3. 任何首次合并仍受提交规范约束；如果所需门禁尚不存在，则保留 PR 等待明确的初始化处理，不能偷偷绕过规范。
4. 专用 App 接入且工作流进入受保护分支后，用真实 PR 验证缺审核、旧 SHA、错误来源、有问题会阻断，以及当前审核能成功；然后才在 ruleset 中要求 `ai-review`，来源绑定专用 App 的实际 ID，并启用 strict latest-base 检查。
5. 原生 approval 保持 0，禁止强推/删除及绕过。此变更不自动创建业务 `branch-policy` 或 `quality-gate`，这些检查需要各自独立实现和验证。

## 最小验收

```bash
python3 -m unittest discover -s scripts/tests -p 'test_ai_review.py' -v
python3 skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
python3 skills/ai-hermes-self-evolve/scripts/route_task.py --route-id R03 --mode context
```

## 接入状态及依据

2026-09-11 已通过 API 核实：现有 Codex 审核 App 可用；ruleset `22899065` 要求 main/Dev 通过 PR、禁止强推/删除/绕过、原生 approval 为 0；`ai-review-publisher` Environment 已创建，只允许 main、Dev 两条 branch，无人类 reviewer。

专用发布 App 凭据、生产工作流事件、真实成功检查和 required-check 绑定尚待完成。离线测试不证明已上线；本次不把业务 `branch-policy`、`quality-gate` 记为已实现。

官方依据：[Environment 分支限制](https://docs.github.com/en/rest/deployments/branch-policies)、[App 安装 token Action](https://github.com/actions/create-github-app-token)、[Checks API](https://docs.github.com/en/rest/checks/runs)、[工作流事件](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)。实际启用与检查结果以现场 GitHub 数据为准。

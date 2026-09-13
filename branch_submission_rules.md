# 提交、审核与合并规范

本文件是 AI 执行分支、提交、PR、审核及合并的统一规则。AGENTS.md 的必读入口保持有效。按用户 2026-09-13 的要求采用以下简化方案，替代此前强制 AI 审核和三项自定义门禁方案。

## 必须遵守

1. 使用实际分支名 `main`、`Dev`；不得另建大小写不同的主线。
2. `main`、`Dev` 禁止直接 push、强推、删除及管理员绕过，只能通过 PR 合并。
3. `main` 只接受**本仓库 `Dev` → `main`** 的 PR。其他开发分支先进入 `Dev`。
4. 两条主线的 PR 都必须获得 **1 个有效 Approve**。采用 GitHub 原生批准要求；审核者须为 PR 作者以外、有仓库写入或管理权限的账号。不能自批，也不能让 AI 冒充人类批准。
5. Codex 评论审核可继续使用，但不是强制门禁，也不能替代有效 Approve。不要求自定义 `ai-review` 或 `quality-gate` 检查。
6. `main` 额外要求 `branch-policy` 检查成功；该检查只验证分支方向。CI 通过不代表已经获得审核批准。
7. 新提交会使旧批准失效；处理未解决的审核讨论后，使用 merge commit 合并，不使用 squash/rebase 或 `--admin`。
8. 配置、状态查询、提交、审核与合并必须能通过终端完成，复用已有 `gh` 授权，不要求网页登录、新建 App 或上传个人令牌。

## 开发和验证

- 先读 AGENTS.md、AI Hermes 路由和本文件；存在 CodeGraph 索引时按 AGENTS.md 导航。
- 先检查当前分支、远端及未提交文件，再 fetch 最新 Dev/main。开发分支默认命名 `codex/<主题>`，从最新 Dev 创建。
- 保留他人或其他任务的未提交文件；必要时使用隔离 worktree。禁止为提交而 reset、clean 或擅自 stash。
- 只暂存本任务的明确文件。不得提交密钥、正式数据、缓存及构建产物。
- 按实际影响运行相关测试、构建和路由检查，必要时更新设计文档。仅文档及分支配置变更不需要业务全量回归。
- 更新 AGENTS.md 或路由时执行 AI Hermes validate；执行 evolve 检查变更覆盖。
- 记录 commit SHA、目标 base、验证结果及未执行项。不要把失败、跳过、未运行写成通过。

## PR 内容

写明变更目的、来源与目标分支、提交范围、验证结果、限制与回滚方式，并附需求和技术文档链接。使用 `--body-file` 保存真实换行。发布 Dev → main 时审核完整差异，不能把其他已累积的业务变更隐瞒为配置修改。

## 终端操作

```bash
# 当前身份、PR、批准和检查
 gh api user --jq .login
 gh pr view PR号 --json state,isDraft,headRefName,baseRefName,headRefOid,reviewDecision,statusCheckRollup
 gh pr checks PR号

# 由作者以外的合格审核者，在自己的终端账号下执行
 gh pr diff PR号
 gh pr review PR号 --approve --body '已审核当前变更，同意合并。'

# 发起者在批准和检查完成后执行，SHA 替换为刚核实的完整 HEAD
 gh pr merge PR号 --merge --match-head-commit 完整HEAD_SHA
 gh pr view PR号 --json state,mergedAt,mergeCommit
```

- 合并前刷新 PR，确认来源正确、非 Draft、有效批准、必需检查成功、无冲突，随后立即匹配 HEAD 合并。
- 不能执行作者自批、代冒审核者或临时降低保护来合并。缺少审核者时完成代码、推送、PR 和配置，然后明确报告缺少有效 Approve。
- main 发布后的历史同步使用独立开发分支和 PR 回到 Dev，禁止直接推送；回滚使用普通 revert PR。

## 配置及能力边界

具体配置和部署命令见 [分支保护实现](docs/branch-protection.md)。文档存在不代表工作流已安装或规则已启用；每次配置后必须读取实际 GitHub 状态。

该方案使用 GitHub 原生批准规则及一份简短的分支方向工作流，不建立额外审核服务或自定义证据发布系统。GitHub 原生检查按名称及 App 匹配，不能隔离恶意管理员或刻意伪造同名 Actions 检查的仓库写权限人；涉及工作流变更必须由审核者检查真实差异。

# 提交、审核与合并规范

本文件是 AI 执行分支、提交、PR、审核及合并的统一规则。采用适合 Owner 主导开发的简化流程：保留 PR 记录和分支方向限制，Owner 可以免批准合并，普通协作者仍需批准。

## 固定规则

1. 使用实际分支名 `main`、`Dev`；不得另建大小写不同的主线。
2. 所有人，包括 Owner，都禁止直接 push、强推或删除 main/Dev；变更必须通过 PR，使用 merge commit 合并。
3. main 只接受**本仓库 Dev → main**。开发分支先提 PR 进入 Dev，再按需要发布到 main。
4. 普通协作者执行合并前，必须有 **1 个有效 Approve**；审核者须为 PR 作者以外、有写入或管理权限的账号。新提交会撤销旧批准。
5. **Owner KevinCJM 可以在 PR 内豁免批准要求**，包括自己创建的 PR。无需另找账号或伪造自批。这项权限依据执行合并的账号判断，因此 Owner 也可以免批准合并别人的 PR；不自动授予其他管理员。
6. Owner 豁免仅适用于独立的批准规则，不豁免 PR-only、禁止强推/删除和 main 的分支方向检查。
7. Codex 评论审核可选，不强制 `ai-review`、`quality-gate`、CODEOWNERS、最后推送者批准、所有讨论关闭或额外批准人数。确认存在的实质问题仍应处理；测试按实际变更范围执行。
8. 全流程通过终端完成，复用现有 gh 授权；不要求网页登录、新 App、额外令牌或复杂校验服务。

## 日常流程

1. 先读 AGENTS.md、AI Hermes 路由和本文件；有 CodeGraph 索引时按项目协议导航。
2. 检查工作区和分支，fetch 最新 Dev/main，从 Dev 创建 `codex/<主题>` 开发分支。
3. 完成变更、相关测试及文档同步；只暂存本任务文件。保留其他任务的脏文件，必要时使用隔离 worktree，禁止擅自 reset、clean 或 stash。
4. commit、push 开发分支，向 Dev 提 PR。正文写清目的、范围、验证、限制和文档链接。
5. Owner 检查差异及结果后直接合并；普通协作者等有效批准后合并。两者均匹配最新 HEAD，不把 pending、失败或未运行写成通过。
6. 发布时另提 Dev → main PR，确认 `branch-policy` 成功，并检查整个发布差异；不能把累积的业务变更隐瞒为配置修改。
7. main 历史回同步 Dev、回滚也走开发分支和 PR，不直推主线。

## 验证要求

- 按改动运行相关测试、构建和路由检查；不要求每次文档或配置修改都跑全量业务测试。
- 不提交密钥、正式数据、缓存或构建产物。自动测试使用可重复的本地夹具，不污染正式数据。
- 更新 AGENTS.md/路由时执行 AI Hermes validate；通过 evolve 检查变更路径覆盖。
- 合并前刷新 PR：非 Draft、无冲突、来源正确、必要检查通过，且当前账号有合并或批准豁免权限。
- 合并后核实 PR 状态、目标分支及 merge commit。Owner 豁免是固定授权机制，不通过临时关闭规则实现。

## 终端命令

```bash
# 查看当前身份、差异和状态
gh api user --jq .login
gh pr diff PR号
gh pr view PR号 --json state,isDraft,headRefName,baseRefName,headRefOid,reviewDecision,statusCheckRollup
gh pr checks PR号

# 普通协作者：由合格审核者在自己的账号下批准，再普通合并
gh pr review PR号 --approve --body '已审核当前变更，同意合并。'
gh pr merge PR号 --merge --match-head-commit 完整HEAD_SHA

# Owner：已核对规则及检查后，显式使用 PR 内的批准豁免
gh pr merge PR号 --merge --admin --match-head-commit 完整HEAD_SHA

gh pr view PR号 --json state,mergedAt,mergeCommit
```

`--admin` 在本流程中只能使用 GitHub 已授予 KevinCJM 的批准规则豁免。PR-only 与 main 来源规则的 bypass 列表必须为空；如果 main 的检查失败，应修正来源或检查，不能改规则放行。普通协作者没有该豁免时不能使用此流程自批或合并未获批准的 PR。

具体规则文件、部署和权限边界见 [分支保护实现](docs/branch-protection.md)。文档存在不代表远端生效，配置后必须读取 GitHub 实际状态。

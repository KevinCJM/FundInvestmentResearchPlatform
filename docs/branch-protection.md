# 分支保护实现

需求以 [提交规范](../branch_submission_rules.md) 为准：主线只接受 PR，main 只接受本仓库 Dev，两个目标均需 1 个有效 Approve。Codex 原有评论审核保留为辅助，不作为必需检查。

## GitHub 配置

- `.github/branch-ruleset.json`：覆盖 main/Dev，禁止删除、强推及 bypass，要求 PR、1 个批准、解决审核讨论，仅允许 merge commit；push 后撤销旧批准。
- `.github/main-branch-ruleset.json`：只覆盖 main，要求 GitHub Actions（App 15368）提供 `branch-policy` 成功检查。
- `.github/workflows/branch-policy.yml`：PR 打开、更新提交、重开、就绪或修改目标时执行。不检出代码、不使用额外凭据，权限为空；检查 main 的来源分支及仓库，其他目标通过。首次安装 PR 可以从候选分支启动检查。
- 原生 Approve 由 GitHub 核验审核权限及作者身份。当前 CLI 用户若也是 PR 作者，不能自批；需要另一个合格账号在自己的终端审核。GitHub 仓库管理员仍可修改仓库设置，因此这里不声称隔离恶意管理员或伪造同名检查的写权限人。

## 终端配置

使用已有 gh 授权，先获取并比较现有规则，保留更严格且无冲突的设置。当前共享规则 ID 为 22899065；main 规则首次创建后，后续使用返回的 ID 更新，不能重复创建。

```bash
gh api repos/KevinCJM/FundInvestmentResearchPlatform/rulesets
gh api --method PUT repos/KevinCJM/FundInvestmentResearchPlatform/rulesets/22899065 --input .github/branch-ruleset.json
gh api --method POST repos/KevinCJM/FundInvestmentResearchPlatform/rulesets --input .github/main-branch-ruleset.json
# main 规则已存在时改用 PUT .../rulesets/实际ID

gh api repos/KevinCJM/FundInvestmentResearchPlatform/rules/branches/main
gh api repos/KevinCJM/FundInvestmentResearchPlatform/rules/branches/Dev
```

工作流通过开发分支 → Dev → main 的 PR 安装。先在真实 PR 验证普通开发分支 → Dev 成功、其他开发分支 → main 失败，再绑定 main 的必需检查。首次 PR 也需要有效 Approve；不得冒充批准或降低保护。尚未携带工作流的 main PR 会因缺少必需检查被阻止，需先将工作流通过 PR 合入 Dev。

## 验证与维护

- 对实际 YAML 检查 main/Dev 来源组合，使用 actionlint 检查语法。
- 现场读取两个目标的有效规则，并核对实际 PR 的检查及 reviewDecision。
- 不用危险的真实主线推送测试保护。配置读取能证明规则生效，但不把 dry-run 当作服务器拒绝推送的证据。
- 工作流无需 Python 校验框架、额外 App、密钥、Environment 或浏览器登录。业务测试按变更执行，不强制安装额外 quality-gate。

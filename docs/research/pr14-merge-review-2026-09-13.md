# PR #14 合并审核记录

日期：2026-09-13。目标分支 `Dev`，来源 `codex/saa-taa-frontier-grid`。本轮审核起点为 `b61ab113771f4ae8709bfb6e83b6707c21864cb5`，基线为 `f77f47d42e377e705485e6bb689e4784e8b46ba4`；最终提交和实际合并结果绑定到 [PR #14](https://github.com/KevinCJM/FundInvestmentResearchPlatform/pull/14) 的审核评论。

## 结论与修复

确认并修复三项问题，相关回归通过，可以按当前 Owner 主导的提交规范合并。此前完整 PR 的独立审核记录保留；本轮重点复核目标确认、研究时钟、SAA/TAA 交接以及新增修复，不把此前审核或测试描述成本轮重跑。

| 问题 | 修复及验证 |
| --- | --- |
| P1：读取失败或未就绪的 PIT 设置被当成无 PIT，允许按今天诊断、计算及保存 | 严格区分 `undefined`、`null` 和已知日期；未知状态禁止诊断、比较和写入。时钟重新未知会取消在途请求并清除未保存候选与确认状态。明确无 PIT 仍可研究，历史版本保持只读可见。单测和真实浏览器覆盖失败、恢复及失效。 |
| P2：原始目标保存 API 绕过诊断、限制确认和边界说明 | 删除原始 `POST /mandates`、`save_mandate` 及前端入口，统一经过 preview/confirm。新政策拒绝没有确认记录的旧目标；历史仍可读取，需复制并重新诊断确认才能用于新政策。路由和服务测试验证拒绝路径不落盘、有效确认成功。 |
| P2：从 TAA 返回 SAA 时丢失所选政策及目标版本 | 链接携带实际选择的不可变 baseline ID；SAA 从服务端读取精确版本，展示权重、采纳理由、目标与 CMA 引用。拒绝错误版本响应；进入历史不重新计算或写入。桌面及手机验证真实 SAA 采用、进入 TAA、返回原版本。 |

## 本轮实际验证

全部在隔离 worktree 与本地测试夹具中执行，未修改原工作区、正式数据或原前后端服务。

| 检查 | 结果与范围 |
| --- | --- |
| 后端回归 | 233 passed，10 个相关测试文件；覆盖目标确认与资金诊断、SAA、前沿网格/采样/NJIT、TAA 服务/桥接/分段检验和研究日期；不是全后端 |
| 前端全量 Vitest | 126 文件、961 passed |
| Playwright / 真实隔离 API 与 NJIT | 10 passed，桌面 1440 和手机 390；覆盖 20/200 网格、权重离散、资金目标、PIT 失败恢复及 SAA/TAA 往返 |
| TypeScript、生产构建、design:check、i18n | 通过；构建仍有既有大包提示，未放宽设计预算 |
| AI Hermes validate / evolve、git diff --check | 通过；路由和 P22 同步三项已验证契约 |

复现命令：使用项目依赖环境，先安装 `frontend` 锁定依赖。

```bash
PYTHONPATH=.:backend RESEARCH_ROOT=/private/tmp/firp-pr14-research \
  /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_strategic_allocation.py backend/tests/test_investment_mandate.py \
  backend/tests/test_mandate_diagnostic_integrity.py backend/tests/test_frontier_grid.py \
  backend/tests/test_frontier_sampling.py backend/tests/test_optimizer_strategy_numba.py \
  backend/tests/test_tactical_walk_forward.py backend/tests/test_tactical_allocation_service.py \
  backend/tests/test_tactical_allocation_bridge.py backend/tests/test_strategy_research_dates.py -q
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
npm run test:e2e --prefix frontend -- --config=playwright.strategic.config.ts
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend
npm run design:check --prefix frontend
node scripts/check_i18n.mjs
python3 skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
python3 skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py --diff-range origin/Dev --json
git diff --check
```

浏览器用例为 PIT 设置提供明确的离线响应；模拟读取失败后通过真实重试按钮恢复。计算、目标确认和政策存储进入真实隔离 API。截图见 [桌面返回原政策](screenshots/pr14-merge-review/saved-policy-desktop.png) 和 [手机返回原政策](screenshots/pr14-merge-review/saved-policy-mobile.png)。

此前 `b61ab11` 的 520 项后端、957 项前端、96 组独立资金递推及性能/零拷贝证据属于 [上一轮复核](../../investment-mandate-fix-recheck-2026-09-13.md)，与本轮计数分开。旧 HEAD 的一条 branch-policy 失败由 GitHub runner 连续五次未能接单引起，随后同一 HEAD 已有两次成功；最终仍须核实最新提交的实际检查。

## 保留边界

- 既有第五轮 P2 仍存在：重复资产暴露造成奇异协方差时，QP 端点可能耗尽预算并返回 `range_unresolved`；这不是不可行证明。本轮没有修改该数值行为或宣称问题关闭。
- 概率仍依赖声明的 CMA 与月度对数正态模型；模拟误差区间不构成收益保证、投资有效性或正式 PIT 认证。
- 依照当前提交规范，Owner 可在 PR 内豁免批准要求；AI 以实际审核评论留痕，不伪造人类 Approve，也不调整远端保护规则放行。

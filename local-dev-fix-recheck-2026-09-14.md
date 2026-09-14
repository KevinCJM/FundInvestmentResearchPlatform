# 本地旧版本问题整改与复核

日期：2026-09-14。对应 [原始审核记录](local-dev-diff-review-2026-09-13.md)。

结论：**本轮四项问题复核通过**，当前工作区已使用 Dev 中的修复。提交及合并状态以本次 PR 的实际 HEAD、检查和合并记录为准。

## 整改方式与范围

四项缺陷的修复已经通过 [PR #12](https://github.com/KevinCJM/FundInvestmentResearchPlatform/pull/12) 和 [PR #14](https://github.com/KevinCJM/FundInvestmentResearchPlatform/pull/14) 进入 Dev。本轮修复当前工作区的版本落后问题，复用这些已经合入的实现。

- 当前开发分支：`codex/frontend-design-quality`。
- 同步前 HEAD：`2329845b345665a9bacc95e36b814868a8e288ea`。
- 同步及测试基线：`f6f85d0ff967d223d59784b305e2e19069e75518`（Dev），完整树 `fea7273201ce23808a27f6db3cd39d41a3c1f281`。
- 用户明确授权备份后同步。96 项本地改动先保存到本地 Git 恢复引用 `refs/codex-recovery/frontend-design-quality-20260914`，提交 `2d24fc3737dd9ab1d8577c60a73468e9e4d5af29`；逐个验证备份内容，再处理 95 个与快进冲突的路径。
- 保留原始审核报告字节内容。第三、四、五轮报告与 Dev 的原有空白差异仍留在本地，不纳入本次提交。
- 同步后全部非文档文件与 Dev 的内容、文件模式一致，没有新增业务代码差异。本次 PR 仅补入上轮原审记录、本轮复核记录及其路由登记。

## 四项问题复核

| 原问题 | 当前实现与验收要求 |
| --- | --- |
| P1：PIT 未知时仍允许操作 | 区分未知 `undefined`、明确关闭 `null` 和已知日期；未知时禁止目标诊断、SAA 比较及写入，时钟变化清除未保存结果并拒绝迟到响应。已保存历史版本仍可读取。 |
| P2：未确认目标可以建立政策 | 删除原始 `POST /mandates` 写入入口；统一通过 preview/confirm。旧的未确认目标可以读取，但新政策预览与保存均返回 `MANDATE_CONFIRMATION_REQUIRED`。 |
| P2：TAA 返回时丢失 SAA 政策 | 按实际选中的 baseline ID 返回 SAA，从服务端读取精确的不可变版本，保留目标、CMA、权重及采纳理由；错误 ID 响应不能替代选定版本。 |
| P2：指标日期和披露信息缺失 | 保留成立/上市日期、标量来源覆盖及实际公告字段状态；公告字段缺失不能报告为已经完成披露筛选。 |

前沿的约束随机游走、逐轮投影、单产品/分组上下限、权重精度、收益/风险选择，以及 20/200 个目标的整条网格求解均沿用 Dev 实现。没有重写数值算法或减少原有参数。

## 本轮实际验证

以下均为本轮重新执行的结果，没有引用旧测试计数代替本轮执行。

| 检查 | 结果 |
| --- | --- |
| 本地后端，8 个相关文件 | **217 passed**；覆盖目标确认与资金诊断、SAA、输入日期/披露、前沿网格/采样/NJIT 和 TAA 分段检验。不是全后端。 |
| 本地前端全量 Vitest | **126 文件，961 passed**。 |
| 本地真实浏览器与隔离 API | **10 passed**，桌面 1440 与手机 390；覆盖约束与精度配置、20/200 个真实目标求解、失败状态、资金诊断、未知 PIT 重试及 SAA/TAA 往返。 |
| TypeScript / 生产构建 / design:check / i18n | 通过；保留既有大包和弃用提示，设计预算未放宽。 |
| AI Hermes validate / evolve、暂存差异检查 | 通过；本次 3/3 路径覆盖，`git diff --cached --check` 通过。 |
| 独立审核 `/root/saa_taa_pr_review` | 四项范围 **PASS**；独立重新执行日期/披露 **16 passed**、前端定向 **4 passed**，以及目标确认探针。被测 `df8dbce1a80a71cb121b4240085ef32ea1d6a040` 与本地/Dev 的完整测试基线同树。 |

上轮失败的 18 个检查已包含在本轮相关全文件测试与独立定向复核中，均已通过。独立探针再次确认原始写入口返回 404，未确认历史目标的 policy preview/publish 均被拒绝；正常 preview/confirm 仍可建立目标。独立审核只认证上述四项，不代替全仓算法审核。

复现命令：

```bash
PYTHONPATH=.:backend NUMBA_CACHE_DIR=.run/local-dev-fix-20260914/numba \
  /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_strategic_allocation.py \
  backend/tests/test_investment_mandate.py \
  backend/tests/test_mandate_diagnostic_integrity.py \
  backend/tests/test_indicator_input_date_context.py \
  backend/tests/test_frontier_grid.py backend/tests/test_frontier_sampling.py \
  backend/tests/test_optimizer_strategy_numba.py backend/tests/test_tactical_walk_forward.py -q
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
npm run test:e2e --prefix frontend -- --config=playwright.strategic.config.ts \
  --output=../.run/local-dev-fix-20260914/browser
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend
npm run design:check --prefix frontend
node scripts/check_i18n.mjs
```

## 证据与边界

本机日志、备份清单、源码一致性及独立复核证据位于 `.run/local-dev-fix-20260914/`，不提交运行缓存或正式数据。测试使用临时夹具；浏览器计算进入独立的真实 API/NJIT 服务，不向日常运行服务写入研究结果。本轮未重启日常前后端服务，不据此声称旧服务进程已经加载新代码。

已豁免的第五轮 P2 保持原状态：重复暴露导致奇异协方差时，QP 端点可能耗尽预算并返回 `range_unresolved`；该状态不是不可行证明。本轮不宣称该限制已经修复，也不把技术回归解释为投资有效性或正式 PIT 认证。

需求与设计继续以 [投资目标优化方案](docs/research/investment-mandate-optimization-2026-09-13.md)、[投资目标前后端设计](docs/research/investment-mandate-research-design-2026-09-13.md) 和 [前沿恢复设计](docs/research/frontier-restoration-design-2026-09-13.md) 为准。

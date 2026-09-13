# 投资目标与边界：独立代码审核

审核日期：2026-09-13。结论：**主流程符合这次强化方向，仍有两个已复现的 P2，不能视为全部关闭。** 本轮未发现可确认的 P1。

审核对象是当前工作区，分支 `codex/frontend-design-quality`，HEAD `2329845b345665a9bacc95e36b814868a8e288ea`。投资目标及关联 SAA 文件目前尚未 Git 跟踪，结论针对磁盘源码，不代表该 HEAD 已包含这些实现，也不属于正式 PR 放行。

本轮仅审核并记录证据，没有修改业务源码、原回归测试或既有设计/验收报告，没有提交、推送或合并。

## 1. P2：所需本金声称达到采纳门槛，回灌同一路径却未达标

位置：[goal_kernels.py:205](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/strategic_allocation/goal_kernels.py:205)。相关计算位于同文件的 `capital_gate_kernel`、路径递推和成功判定。

**触发条件：**较大金额下，反推资本与逐月财富递推的浮点误差超过成功判定使用的固定绝对容差 `1e-8`。两个计算在实数数学上等价，浮点执行结果却可能落在边界两侧。

确定性复现使用合法资金计划：总资金 10 亿元，组合外储备 1 亿元，可投资本金 9 亿元；十年、每月支付 200 万元，第 61 月投入 1 亿元，期末目标 20 亿元。组合年简单收益均值 7%、波动 17%、年费用 1%，500 条路径、种子 0，采纳门槛 80%。

| 指标 | 返回的资本测算 | 原样回灌本金后的实际诊断 |
| --- | ---: | ---: |
| 可投资本金 | 2,181,143,227.8779964 | 完全相同 |
| 成功路径 | 418 / 500 | 417 / 500 |
| 成功概率 | 83.6% | 83.4% |
| Wilson 区间下界 | 80.1005363526% | **79.8864683803%** |
| 达到 80% 门槛 | 是 | **否** |

回灌保持组合、现金流、目标、费用、路径和种子不变，没有重新搜索 SAA。独立探针经过 `MandateRequest` 校验及真实 `diagnose_funding` 编排，再调用同一个固定签名 NJIT 内核验证。

**影响：**“按采纳口径测算本金”返回 `solved` 和达标概率，但该本金在实际判定中不通过。这是追加资本诊断与采纳规则不一致；本次没有复现将未达标组合直接绕过政策采纳门禁。

**修复要求：**资本反推与实际成功判定采用统一、随金额尺度合理变化的数值边界；返回所需本金前，使用实际递推规则核验对应成功计数，必要时向上调整至可验证的资本值。不要仅修改展示概率。补充大额资金、不同金额尺度、临界路径、必要支付及金额展示精度下的回灌测试，仍须满足固定签名 NJIT 要求。

复现脚本：[capital-repro.py](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/.pytest_cache/investment-mandate-review-2026-09-13/capital-repro.py)；原始输出：[capital-repro.log](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/.pytest_cache/investment-mandate-review-2026-09-13/capital-repro.log)。脚本最后的正确行为断言当前失败。

```sh
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 \
  .pytest_cache/investment-mandate-review-2026-09-13/capital-repro.py
```

## 2. P2：SAA 知识截止日改变后，旧候选仍能发出采纳请求

位置：[StrategicAllocationWorkspace.tsx:146](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/StrategicAllocationWorkspace.tsx:146)。确认按钮的禁用条件也有同样遗漏，见同文件第 193 行。

**触发步骤：**

1. 在知识截止日 2026-09-12 下引用同日 CMA，比较政策候选，进入复核并填写采纳理由。
2. 将知识截止日调早至 2026-09-11。
3. 页面已经显示“假设研究日晚于平台知识截止”错误，但旧候选和“确认采用此长期政策”按钮仍保留。
4. 点击按钮，实际发出一次 `POST /api/strategic-allocation/policies`。

**根因：**`platformDay` 变化只影响 `dateIssue`；没有使未保存的政策预览、候选和在途响应失效。`adopt()` 及确认按钮也没有检查 `dateIssue`。虽然“验证假设”和“比较候选”检查了日期，最后采纳没有继承同一门禁。

**影响：**用户已经选择更早知识边界，仍能将含较晚 CMA 的旧预览送入采纳流程，违背页面自己声明的日期规则。复现使用真实 React 页面和请求客户端、模拟 API 响应，证明的是错误的采纳请求已发出，不冒称在正式成果库写入了版本。

**修复要求：**参照投资目标页对未保存诊断的处理，在研究时钟变化时使 SAA 未保存预览和候选失效，并递增请求代次或取消在途请求；采纳函数和按钮均检查日期资格。已保存不可变政策仍可作为历史记录读取。补测复核页改时钟、请求在途改时钟及历史版本继续可读三类场景。

复现测试：[clock-repro.test.tsx](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/.pytest_cache/investment-mandate-review-2026-09-13/clock-repro.test.tsx)；输出：[clock-repro.log](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/.pytest_cache/investment-mandate-review-2026-09-13/clock-repro.log)。测试断言“不得发送采纳请求”，当前失败，日志记录 `AUDIT_POLICIES_POSTS 1`。

```sh
node frontend/node_modules/vitest/vitest.mjs run \
  --config .pytest_cache/investment-mandate-review-2026-09-13/vitest.review.config.mjs
```

## 3. 已核对的业务意图

| 业务要求 | 当前源码与专项验证结果 |
| --- | --- |
| 绝对收益、金额与期间支付、相对基准目标分别处理 | 已落实；金额目标没有被强行换成 CMA 算术收益下限 |
| 组合外储备、组合内流动性预算分开 | 已落实；本金只扣一次外部储备，流动性取人工下限与资金计划要求中更严者 |
| 现金流、费用、通胀真正参与计算 | 已落实；先月度增长及费用，再投入和支付，曾经支付不足不会被后续入金抹除 |
| 单资产和联合授权向 SAA 传递 | 已落实；授权绑定大类方案，实际候选执行上下限，不能通过请求放宽冻结授权 |
| 基准相对风险与 TAA 相对 SAA 风险分别计算 | 已落实；没有把两种主动风险混用 |
| 缺少 CMA 或诊断不等于通过 | 已落实；可保存输入或未达标研究，SAA 对选中组合重新诊断并检查门槛 |
| 已确认资金模拟种子和 SAA 搜索种子独立 | 已落实；专项覆盖 917 与 19/53 的分离 |
| 历史保存与当前应用资格分开 | 已落实到目标页及后端政策到期检查；SAA 未保存预览的时钟失效仍有第 2 项缺陷 |

从流程上看，先明确目标、资源和约束，再结合资本市场假设形成 SAA，与 CFA 对 IPS 和战略配置关系的描述一致。这是对本项目实现方向的判断，不意味着已覆盖完整税务、监管或客户适当性系统。[CFA 原文](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/basics-of-portfolio-planning-and-construction)

Wilson 区间公式本身已通过独立对照；第 1 项问题发生在资本数值与实际成功计数之间，不是要求更换区间算法。Wilson 是比例的置信区间方法，不能据此解释为现实收益保证。[NIST 公式说明](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)

当前月度独立对数正态模型、四类代表组合比较、人工流动性标签均已在界面和设计中说明边界。未把未实现的厚尾、多目标优化或逐产品赎回模型列为本轮缺陷，也不建议为解决以上两项问题减少已有参数或功能。

## 4. 本轮实际验证

| 验证范围 | 结果 |
| --- | --- |
| `test_investment_mandate.py`、`test_mandate_diagnostic_integrity.py`、`test_strategic_allocation.py` | **84 passed**，43.00 秒，1 条 Starlette 弃用警告 |
| 投资目标页面、SAA 页面、战略配置客户端三个前端测试文件 | **37 passed**；存在 React `act` 测试警告 |
| 真实浏览器：金额诊断及目标→CMA→SAA→TAA，1440 桌面与 390 手机 | **4 passed**，约 1.2 分钟；隔离临时 API 与数据 |
| 新增审核探针，断言修复后的正确行为 | **2 项失败复现**：资本回灌、SAA 时钟采纳 |
| 资本复现中的执行审计 | `fully_warmed=true`、`python_fallback=0`、`request_time_compilation=0`；资金内核版本 `1.1.0` |

后端和前端专项命令：

```sh
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_investment_mandate.py \
  backend/tests/test_mandate_diagnostic_integrity.py \
  backend/tests/test_strategic_allocation.py -q

npm run test --prefix frontend -- --run \
  src/pages/InvestmentObjectivesWorkspace.test.tsx \
  src/pages/StrategicAllocationWorkspace.test.tsx \
  src/services/strategicAllocation.test.ts

npm run test:e2e --prefix frontend -- --config=playwright.strategic.config.ts \
  --workers=1 --grep 'real funding goal|goal, CMA, policy adoption'
```

浏览器日志：[browser.log](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/.pytest_cache/investment-mandate-review-2026-09-13/browser.log)。审核源码摘要：[source-manifest.json](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/.pytest_cache/investment-mandate-review-2026-09-13/source-manifest.json)。缓存目录为本地审核证据，未纳入 Git；复现的关键输入、预期和实际结果已写在本文。

本轮依据 AGENTS、路由文档及 CodeGraph 缩小范围，再以当前源码和独立复现验证。未重复执行全仓测试、生产构建或正式数据投资有效性评估；不沿用整改记录中的全量通过数字作为本轮实测结果。

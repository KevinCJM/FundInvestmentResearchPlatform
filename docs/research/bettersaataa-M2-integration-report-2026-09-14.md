# M2 实际前后端接入报告

日期：2026-09-14；工作区：`ISSUE2609/BetterSaaTaa`。

已完成实际 CMA → 唯一政策搜索 → 资金诊断 → 不可变 SAA → TAA 政策门禁接入。不是独立模型演示。先阅读 AGENTS、R87 路由、设计准则、01–04需求/设计、M2内核报告和审核发现，并先使用 CodeGraph。M1 完成标记出现且读取交付报告后，才编辑共享文件；等待期间仅准备新文件和 `/private/tmp` 隔离副本检查。

没有修改 M3 的 TAA/historical_regimes 文件、协调器 E2E/config、共享研究/设计/验收文档、路由 JSON、AGENTS、正式数据或秘密；没有 Git 写操作，没有创建子代理。

## 交付文件

- 后端：`contracts.py`、`service.py`、`routes.py`、`policy_gate.py`（均在 `backend/strategic_allocation/`）；新增 `cma_application.py`。
- 前端：`services/strategicAllocation.ts`、`pages/StrategicAllocationWorkspace.tsx`、`components/strategic-allocation/PolicyCandidates.tsx`。
- 新组件：`ModelAssumptionFields.tsx`、`EffectiveCmaResult.tsx`、`RiskBudgetEditor.tsx`，位于 `frontend/src/components/strategic-allocation/`。
- 新测试：`backend/tests/test_cma_model_integration.py`、`frontend/src/pages/StrategicAllocationModels.test.tsx`、`frontend/src/services/strategicAllocationModels.test.ts`、`RiskBudgetEditor.test.tsx`、`PolicyCandidatesM2.test.tsx`。
- 原 `AssumptionEditor`、`ForwardAssumptions`、独立 BL/情景内核及协调器的候选搜索内核均未修改。

## 实际计算与冻结契约

`CmaRequest.model` 是可选、按 `method` 判别的 BL/情景联合契约；公共类型改为从 `common_contracts.py` 导入/重导出，避免循环。省略模型保留人工数学口径；人工模式仍要求全部收益、波动和相关性。显式模型可省略原始人工数值，父契约校验模型和 CMA 的完整资产顺序、研究日、币种、年化算术总收益口径。资产角色和流动性不从模型推断。

`_cma_calculation → cma_application.apply_model → evaluate_cma_model` 实际计算并保存：

- `definition`：经过契约验证的原始请求，缺失人工数值保留为空，不补假数据。
- `effective_assumptions`、`effective_returns`、`effective_covariance`、既有 `covariance`：实际数值结果与完整经济元数据；显式 `mean_uncertainty` 原样保留。
- `model_result`：原模型、后验均值协方差、方法限制、NJIT审计及独立内容哈希。BL后验均值协方差不作为资产风险，也不转为稳健半宽。
- 冻结只读 NPY：`effective_returns`、`covariance`、`mean_uncertainty`，BL额外含 `posterior_mean_covariance`。ArtifactRepository 保存每个数组的 dtype/shape/SHA256 以及整个 manifest 内容哈希。

`_candidate_calculation → frozen_numeric_inputs → policy_candidates_with_budget_kernel` 使用冻结有效均值/风险；同一候选的实际矩进入原资金诊断。没有第二个候选搜索，没有修改协调器算法。政策保存有效 `assumptions`、原始 `raw_assumptions`、模型结果/审计/哈希以及有效均值/协方差；`policy_gate.check_policy` 读取这些冻结假设并验证血缘一致性，实际用于 TAA 的预期收益、总风险和主动风险。门禁增加 `expected_return` 展示值。

旧 CMA/政策无模型字段时按原人工语义读取，不补算、不回写；新模型缺有效血缘时失败。模型风险不能沿用无关的 `historical_reference` 标签。确认重新计算并核对 preview_hash；预览不保存文件。模型输入错误和 `LinAlgError` 返回人类可读领域错误；未预热 API 返回503。每个 worker 的实际 `StrategicAllocationService.warm()` 预热模型并合并完整性检查，保留 M1 机构审计；请求不调用 warm、不编译新签名、不回退 Python。

必要分配限定在 JSON/NumPy 边界、小矩阵内核输出和持久化；政策一次读取3个已校验的只读 mmap 数组，并持有其 owner 至计算结束。不宣称整条既有产品读取链零分配。模型数值和只读 stride/owner 证据继续由原62项测试覆盖。

## 风险预算与界面

`PolicyRequest.risk_budget` 可省略；提供时须完整覆盖资产轴、非负、合计1。缺省仍返回原4候选；显式预算使用同一个搜索返回第5个 `risk-budget`。返回有符号 Euler 风险贡献及平方距离，明确是有限搜索匹配，不承诺精确风险平价。第5候选也执行资金目标、基准、现金及其他原目标门禁。

所有可行候选风险贡献均未定义时，第5项返回 `available=false`、空权重、null指标、原因；禁止发布。前端 API 边界把这项转入 `unavailable_candidates`，页面显示原因，不渲染伪造权重或可采纳按钮。有效第5项仍在 `candidates`，ID贯穿预览、选择和发布。

现有 SAA 假设步骤已挂载 `CmaModelEditor`。选择直接填写时保留原人工/历史参考和独立战略表单；选择模型时只保留经济角色、流动性、理由、来源和显式稳健半宽，不重复要求人工收益/风险。浏览器不执行模型数学。验证后展示实际有效收益、风险与限制，再允许确认。

风险预算复选框默认关闭；开启后所有百分比为空，须完整填写并合计100%，禁用原因直接可见。模型保存后只读，显式复制后才能修改；草稿只增加已保存 CMA 的 ID 引用，重载重新读取该版本，不保存计算证据。修改输入/研究时钟/来源使未保存结果及迟到响应失效；切换范围/映射不能继续带用旧模型；保存假设复用仍校验范围、映射、币种和期限。

## 最终验证

后端全部使用 `PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12`，离线测试数据仅位于临时目录。

|检查|结果|
|---|---|
|`test_cma_model_integration.py` + 原 strategic/investment/mandate integrity + models62 + riskbudget10 + scopejourney3 + M1|**215 passed，36.94s**|
|新增实际服务集成用例|**25项**，包含 BL 先验/相对观点改变真实权重、情景组间协方差进入资金/TAA、原始/有效血缘、只读数组、预览文件不变、错误轴/日期/币种/收益口径/哈希、PID与无请求预热、第五候选/零方差/资金门禁、旧记录无补算、无产品战略范围和现金门禁|
|Vitest：StrategicAllocationModels、strategicAllocationModels、StrategicAllocationWorkspace、InvestmentObjectivesWorkspace、strategicAllocation、CmaModelEditor、RiskBudgetEditor、PolicyCandidatesM2、StrategicScopeWorkspace、allocationJourney.m1|**75 passed，4.01s**|
|`node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json`|通过；M1报告提到的并行 CMA 类型错误已消除|
|`npm run build --prefix frontend -- --outDir /private/tmp/bettersaataa-m2-integration-build`|通过，5.70s；保留既有大bundle提示|
|`npm run design:check --prefix frontend`|通过，无回归|
|`node scripts/check_i18n.mjs`|`errors=[]`|
|`git diff --check`|通过|
|显式16文件 `evolve_ai_routing.py --changed-file ... --json`|3个新增测试尚未登记，见下方；没有修改路由JSON|

日志：`/private/tmp/m2-integration-{backend,vitest,build}-final.log`、`/private/tmp/m2-integration-design.log`、`/private/tmp/m2-integration-i18n.log`、`/private/tmp/m2-integration-routing.json`。现有 React act 和 Starlette/httpx 弃用提示没有作为测试通过的替代证据。

## 剩余精确交接

1. **真实浏览器验收未执行**。遵照指令没有在沙箱启动浏览器/监听端口。协调器通过 DevSpace 运行其自有 `frontend/e2e/better-saataa-01-04.spec.ts` 和配置，核对320/768/1440布局、键盘、真实交互状态与实际文字对比度。Vitest/tsc/build/design检查不代表浏览器视觉验收。
2. 协调器在路由索引登记 `backend/tests/test_cma_model_integration.py`、`frontend/src/pages/StrategicAllocationModels.test.tsx`、`frontend/src/services/strategicAllocationModels.test.ts`，再运行 evolve/validate。本轮未编辑路由JSON或共享验收文档。
3. 上述结果认证当前 M2 接入和列明回归；不认证生产发布、真实交易、策略投资有效性或来源已获正式历史PIT认证。

# 投前01–04：最终开发、审核与验收

验收日：2026-09-14。分支：`ISSUE2609/BetterSaaTaa`。
基线：`aa8ca7a8a707ed1cc10f4ef624f1ceb5894ebada`（创建时及最终 fetch 核对的 `origin/Dev`）。

关联：[调研](bettersaataa-01-04-research-2026-09-14.md) · [需求](bettersaataa-01-04-requirements-2026-09-14.md) · [详细设计](bettersaataa-01-04-design-2026-09-14.md) · [最终审核](bettersaataa-review-findings-2026-09-14.md)。

## 1. 本次实现

|里程碑|实现结果|验收范围|
|---|---|---|
|M1：目标与投资范围|机构背景、经济快照、现金用途下限、人工核验；独立战略范围、不可变实施映射与缺口；原产品池和分类路径保留。|契约、真实临时数据 API、目标→无产品 SAA→明确映射、现金及下游门禁通过。|
|M2：CMA 与 SAA|人工 CMA 保留；增加 Black–Litterman、显式概率情景矩、风险预算候选；有效均值/风险真实进入唯一政策搜索、资金诊断、冻结政策及 TAA 检查。|数值参考、只读内存、历史版本、不可变发布、真实浏览器全链通过。|
|M3：TAA|多信号组合；决策与执行频率、滞后、最小持有期、阈值、独立费用口径；非交易期漂移；直接产品交接共享门禁。|因果性、样本外隔离、费用对账、实际持仓越界、旧日频兼容及浏览器状态通过。|
|M4：最终集成|修复战略旅程导航、到期原因缺失、Python 逐行校验；补真实模型→SAA→TAA 浏览器用例和审核文档。|后端452项、前端999项、浏览器28项通过；路由同步结果见第7节。|

本记录覆盖下列实际命令与代码范围，不等于生产发布、全仓后端测试全通过、投资业绩有效或正式历史 PIT 认证。各 M1/M2/M3 单独报告是开发阶段证据；其中的“浏览器尚未执行”“类型尚未完成”“待集成”等历史状态由本次最终记录更新。

## 2. 分支与工作区边界

已从 `origin/Dev` 创建、切换并首次推送本开发分支，建立远端跟踪。最终提交前再次 fetch 核对主线，未发现新主线提交需要合并。

本需求的源码、测试、设计与路由更新属于交付范围；原有 `AGENTS.md` 的工作区改动保留，不纳入本需求提交。没有创建/合并 PR，没有推送 `Dev` 或 `main`，没有修改远端保护设置。代码提交及推送以本开发分支 Git 日志和远端引用为准。

测试仅使用临时 Parquet/研究目录或受控 HTTP 夹具，不修改正式市场数据、投资记录或秘密。未新增依赖版本，`npm ci` 使用原锁文件。构建产物与测试缓存不提交。

## 3. 自动测试结果

|检查|最终结果|说明|
|---|---|---|
|受影响后端集成与兼容回归|420 passed，175.10s|目标、机构、战略范围、映射、CMA、风险预算、TAA、产品研究、策略、分析接口及窗口。|
|原有效前沿采样回归|32 passed，25.44s|单独执行避免组合命令超时；与上项是不同测试文件。|
|全量前端 Vitest|135 files / 999 passed，69.40s|包含新增组件、上下游引用、晚返回、只读与导航测试。|
|新01–04真实API浏览器链路|9 passed|3条流程×320/768/1440：无产品战略研究、BL→风险预算→SAA→真实TAA、情景CMA→风险预算→SAA→真实TAA。|
|机构/战略映射浏览器|1 passed|真实临时数据API，目标→独立范围→前瞻SAA→显式映射，覆盖多屏宽与只读。|
|原SAA浏览器回归|10 passed|桌面/手机的历史前沿、网格、资金目标、未知PIT、CMA政策与TAA。|
|TAA浏览器回归|8 passed|受控HTTP夹具下的真实界面：观点、保存/交接、新时钟、多信号和缺实际持仓阻断。此组不是实际后端计算证明；新9项组另验证真实后端。|
|TypeScript、Vite生产构建|通过|构建5.67s，原大包警告保留。|
|设计规则、i18n引用检查、git diff --check|通过|不放宽规则；i18n检查不等于全部新增中文已完成英文翻译。|

开发前基线为后端183项、相关前端85项。开发中失败没有计入通过：最初TAA编译错误、并行接入类型错误、浏览器监听受限均已在当前环境解决并重跑。最终综合后端命令曾被180秒工具超时中断；中断不算通过，随后拆分为420项和32项，分别取得完整成功退出。浏览器先后修复了精确label查找和用例提前结束导致的失败，最终28项均完整通过。

### 可复现后端命令

先按项目环境设置解释器及临时研究目录：

```sh
PYTHON=/Users/chenjunming/Desktop/myenv_312/bin/python3.12
export PYTHONPATH=.:backend
export CUSTOM_INDICATOR_DATA_DIR=/private/tmp/bettersaataa-final-tests
"$PYTHON" -m pytest \
 backend/tests/test_bettersaataa_final_audit.py \
 backend/tests/test_bettersaataa_m1.py \
 backend/tests/test_bettersaataa_scope_journey.py \
 backend/tests/test_cma_model_integration.py \
 backend/tests/test_cma_models.py \
 backend/tests/test_strategic_risk_budget.py \
 backend/tests/test_tactical_allocation_clocks_signals.py \
 backend/tests/test_strategic_allocation.py \
 backend/tests/test_investment_mandate.py \
 backend/tests/test_mandate_diagnostic_integrity.py \
 backend/tests/test_tactical_allocation_numeric.py \
 backend/tests/test_tactical_allocation_service.py \
 backend/tests/test_tactical_allocation_bridge.py \
 backend/tests/test_tactical_allocation_data.py \
 backend/tests/test_tactical_walk_forward.py \
 backend/tests/test_historical_regime_taa.py \
 backend/tests/test_portfolio_research.py \
 backend/tests/test_portfolio_regime_backtest.py \
 backend/tests/test_strategy_api.py \
 backend/tests/test_analytics_routes.py \
 backend/tests/test_window_slice.py \
 backend/tests/test_rebalance_window.py \
 backend/tests/test_backtest_output.py \
 backend/tests/test_research_input_checks.py -q
"$PYTHON" -m pytest backend/tests/test_frontier_sampling.py -q
"$PYTHON" backend/tests/benchmark_taa_clocks.py
```

### 可复现前端命令

```sh
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend
npm run design:check --prefix frontend
node scripts/check_i18n.mjs
npm run test:e2e --prefix frontend -- --config=playwright.better-saataa.config.ts --workers=1
npm run test:e2e --prefix frontend -- --config=playwright.m1.config.ts --workers=1
npm run test:e2e --prefix frontend -- --config=playwright.strategic.config.ts --workers=1
npm run test:e2e --prefix frontend -- --config=playwright.taa.config.ts --workers=1
```

## 4. 数值与资源证据

- BL 对照 NumPy 线性求解，包括无/弱/绝对/相对观点与奇异半正定输入；后验均值不确定性与资产收益风险分开。情景矩保留组间均值方差；无概率压力情景不进入概率模型。
- 原四类 SAA 候选与 Git 基线做3种子×2约束逐元素对照；原 TAA 日频核心做3种子×3费率、100期×4资产逐元素对照。未将原实现另存进生产源码。
- 输入只读和任意步长，测试实际 `np.shares_memory`、owner不污染、dtype、缺失与边界。新批量方向校验进入单次固定签名NJIT扫描，不在Python逐期调用内核。
- 最新 `benchmark_taa_clocks.py`：2000×8只读非连续输入，strides=(256,16)，shares_memory=true；输出688000B，tracemalloc峰值1396861B；5次约0.19–0.30ms，单一签名、python_fallback=0。进程峰值RSS359579648B包含解释器、导入及预热，不代表单请求增量；并非相对旧版加速倍数证据。
- 必要分配包括JSON/Parquet解码与轴对齐、I/O pivot、小矩阵求解工作区、信号/约束输出和递推缓冲。输入窗口复用视图；结果与独占工作区允许分配，不宣称全系统零分配。
- 真实隔离API生命周期完成模型及TAA预热；测试覆盖冷PID拒绝、禁止请求编译、无Python回退。正式 `backend/app.py` 沿用既有服务warm与readiness接线；本次未启动正式数据服务，不宣称正式部署的全部worker已验收。

## 5. 前端验收

真实浏览器覆盖320/768/1440，以及原流程390宽度；验证可见字段、必填/禁用原因、编辑使预览失效、预览与确认分离、模型保存后只读、历史恢复、战略先行导航、缺映射/缺实际持仓不可交接。执行页面宽度溢出检查和实际DOM文字对比度检查。

截图保存在 `.pytest_cache/better-saataa/browser/`、`.pytest_cache/bettersaataa-m1/browser/`、`.pytest_cache/top-down-allocation/browser/`；TAA独立夹具输出在其配置定义的临时目录。截图属于可重跑的本地验收产物，不提交缓存。自动DOM对比度覆盖不等于完整WCAG认证；不把未人工检查的图像、滤镜和所有未覆盖状态声明为通过。

## 6. 审核及适用边界

[最终审核记录](bettersaataa-review-findings-2026-09-14.md)逐项关闭R01–R15。核心保护：不改旧请求默认语义，不补算旧历史，不从名称猜映射，不因缺信号重分配权重，不用Holdout选强度，不把研究保存等同可应用。

本次没有实现或伪装完整随机负债ALM、私募逐笔pacing、税法/监管引擎、真实审批、外汇/衍生品清算或真实下单。外部多信号仍需有来源的数据输入；机构资料只用于公开方法研究，不声称复制机构内部专有系统。

存量Starlette/Pydantic弃用、React act、Browserslist版本和Vite大包警告仍在；未借本次需求扩大改造无关依赖、下载体系或全站设计。

## 7. 路由与交付核验

已更新 `docs/repo_map.json` 的战略配置、TAA及前端旅程事实和测试引用；更新 `docs/task_routes.json` 的R87/R86任务匹配；新增 `docs/pitfalls.json` 的P23，记录战略/产品范围、有效CMA、现金角色及执行时钟契约。没有修改AGENTS协议或检查器阈值。

`validate_ai_routing.py` 通过；R87、R86的 `route_task.py --mode context` 通过；对明确列出的91个本任务文件运行 `evolve_ai_routing.py`，结果为 `exit_code_reason=ok`、`uncovered_files=[]`、`missing_required_files=[]`、`routing_only_violations=[]`。工作区及暂存区的 `git diff --check` 通过。

交付仅提交并推送 `ISSUE2609/BetterSaaTaa`，不创建或合并PR，不推主线。最终提交号由 `git log -1 --format='%H %s'` 查询；远端一致性用 `git rev-list --left-right --count origin/ISSUE2609/BetterSaaTaa...HEAD` 核对。`AGENTS.md`原有工作区修改仍排除在本任务提交之外。

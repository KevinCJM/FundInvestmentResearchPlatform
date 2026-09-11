# 全项目代码与逻辑审核

审核日期：2026-09-10。对象：FundInvestmentResearchPlatform-frontend-framework 当前工作区。

基线：分支 `codex/frontend-investment-process-framework`，HEAD `1664c7b` 加当时所有未提交改动。本报告不是对该 Git commit 单独作出的结论。审核主批次于本地时间 22:55 启动，随后完成浏览器补跑和独立复现。主批次前后及复查时的业务源码哈希一致，没有发现审查期间源码漂移。

本轮只做检查、复现和记录：没有修改业务源码，没有提交或推送，没有重启用户正在运行的服务。复现使用临时合成数据，不把合成结果当作真实投资表现。

## 一、结论

确认了 **6 项实际逻辑／接口问题**，另有 **1 项导航翻译遗漏**。其中 4 项为高优先级：缺失数据压缩日期轴、并发请求串用 PIT 上下文、历史情景识别拒绝合法滚动计算、新版内置指标无法接入下游研究中心。

上一轮内置时序指标迁移在指标中心内部完成，但跨中心适配没有同步完成；“指标中心专项通过”不能代替“全项目集成通过”。现阶段不应将整个工作区标记为已经验收可发布。

严重程度定义：P1 为错误结果、知识时点污染或已有核心流程阻塞；P2 为特定配置下的结果／接口错误；P3 为展示或工程维护问题。没有把每个失败测试都计为一个独立业务缺陷。

## 二、执行结果

| 检查 | 结果 | 说明 |
|---|---|---|
| CodeGraph | 885 个文件、18,174 个节点、61,798 条关系 | 同步检查显示已最新；用于定位实际调用链，不代表逐行证明无错 |
| Python 语法 | 468 个文件通过 | 后端、scripts 及根目录 Python 文件；没有语法解析失败 |
| 后端全量 pytest | 2,958 通过，27 失败 | 共 2,985 项；0 收集错误、0 跳过；未被超时截断 |
| 前端全量 Vitest | 800 通过，1 失败 | 115 个测试文件；失败项为导航翻译覆盖 |
| TypeScript 类型检查 | 失败，2 条诊断 | 均在 EtlWorkflowEditor.test.tsx，不是生产页面的类型错误 |
| 前端生产构建 | 通过 | 构建输出到临时审计目录，没有覆盖正常 dist；保留大包体警告 |
| 国际化静态检查 | 通过 | 该检查未捕获导航翻译覆盖遗漏，不能替代相应单元测试 |
| 桌面浏览器验收 | 去重后 70 通过、14 失败、1 未启用 | 首批达到失败阈值后，对剩余项目补跑；对环境和布局失败单独复查；共 85 个独立用例 |
| git diff --check | 通过 | 当前改动格式检查 |
| Hermes 全仓校验 | 失败，74 条诊断 | 稳定路由引用的文件／目录尚未被 Git 跟踪，不能据此判定其业务算法错误 |

浏览器结果按最后一次有实际执行的结果去重，不累加重复通过数。未启用的一项是真实数据／写入型验收，本轮没有授权其执行。当前覆盖桌面项目；不能声称所有移动端与真实市场数据组合均已验收。

旧报告提到的 `ResearchWorkbenchContent.tsx` JSX 构建阻塞在当前源码中已不再复现。本轮整站构建通过，不沿用过期结论。

## 三、已确认的问题

### A01 · P1：缺失价格被删除，滚动窗口跨过缺失日计算

**代码位置**：`backend/custom_indicators/series_provider.py:1741-1746`；衍生收益率还会在其后约 1778-1785 行基于压缩后的净值重新计算。

日期轴来源的数值先把 NaN／Inf 归为缺失，随后执行 `dropna(subset=["date", anchor_physical])`。这会删除有明确日期、但价格缺失的整行，与时序服务“日期不压缩、缺失保留”的契约矛盾。

**独立复现**：八个交易日的收盘价为 `[10, NaN, 30, 40, 50, 60, 70, 80]`，执行 `rolling_apply(mean(market_close), 2)`，日期轴为 `market_close`。

- 2026-01-05 的缺失行直接消失。
- 2026-01-06 返回 20，即把前面的 10 与当日 30 当作连续的两期。
- 完整窗口且不跳日期的正确结果在该日应为空；到 2026-01-07 才可得到 35。
- 接口返回 `status=ok`，`warnings=[]`，没有向调用方说明日期被压缩。

**影响**：均线、布林带等可能跨缺失期计算；净值作为日期轴时，衍生收益率也存在跨缺口拼接的风险。零值与缺失值不能混用。

**修复方向**：把日期轴是否存在与该日期数值是否有效分开。保留已知日期上的无效值，滚动时按明确的缺失与最少观察数契约处理；不能简单删行或补零。追加从 Parquet 读取到接口输出的端到端测试，不能只测试已经对齐的 NumPy 内核。

**证据**：`.tmp_project_audit_20260910/probe_indicator_boundaries.py` 及同名 JSON 的 `missing_anchor_row`。

### A02 · P1：并发资产配置请求串用数据截止日和知识可得时间

**代码位置**：`backend/fit.py:257-294,399-407`；`backend/services/analytics_routes.py:93-95,148`；`backend/app.py:534-552`。

计算通过进程级 `_LAST_NAV_LINEAGE`、`_LAST_NAV_AVAILABILITY` 暂存最近一次取数上下文，再由调用方读取。这不是请求隔离的数据结构。另一个线程能够在计算与读取之间覆盖这两个全局对象。

**独立复现一**：请求 A 计算截至 2026-01-23 的净值，请求 B 随后计算截至 2026-02-13 的净值。A 的净值确实止于 01-23，但读取到的 lineage 变成 B 的 02-13，样本行数也从 16 变成 31。

**独立复现二**：A 的产品在 2026-01-06 的数据，正确可得日为 2026-01-09；B 的另一产品当日即可得。并发交错后，A 读取到的可得日变成 2026-01-06，提前了三天。

以上使用真实 `compute_classes_nav`、两个线程和显式同步事件重现，没有替换数值计算函数。同步事件只是稳定复现可能发生的请求交错。

**影响**：不仅是界面显示错日期。`save_allocation` 会把 `last_nav_availability()` 映射为落盘的 `available_at`。从实际调用链看，串用数据有进入下游历史可用性检查的风险。本轮没有向正式资产文件写入这类错误数据。

**修复方向**：计算返回值显式携带属于本次请求的 lineage／availability，沿调用链传递。不能仅锁住 getter，因为它仍可能读到另一个请求已写入的对象；不应依赖“上一次调用”的进程全局状态。

**证据**：`probe_concurrent_lineage.py/.json`、`probe_concurrent_availability.py/.json`，均在临时审计目录。

### A03 · P1：历史情景识别将合法的基础滚动公式误判为非因果

**代码位置**：`backend/historical_regimes/formula.py:64-105,281-329`；`backend/cal_indicators/operator_lowering.py:43-58`。

情景公式使用当前 DSL 编译。`rolling_mean` 等旧拼写会先展开为 `rolling_window + mean/std/min/max`，但后续允许集合仍只认识旧拼写，随后把已经展开的合法滚动节点判为禁止的全样本计算。

**独立复现**：以下四个公式均使用普通有限数据即可被拒绝，与网络和数据库无关：

```text
rolling_mean(value, 3)
rolling_std(value, 3, 1)
rolling_min(value, 3)
rolling_max(value, 3)
```

例如第一条返回 `FORMULA_NON_CAUSAL_OPERATOR`，诊断对象是 `["mean", "rolling_window"]`。

**影响**：现有历史情景画布的基础滚动节点、上游行情预览和部分公式往返不能正常准备执行。后端失败集中在 `test_regime_series_builder`、`test_regime_product_sources`、`test_regime_math_presentation`、`test_regime_indicator_nodes`；不是 24 个互不相关的错误。

**修复方向**：让情景校验理解真实的滚动窗口／执行作用域与版本契约。禁止直接把普通 `mean` 全面加入因果白名单，否则可能把全样本均值广播这种真正的未来信息泄漏放行。需同时测试合法窗口通过、未受窗口约束的全样本聚合仍被拒绝。

**证据**：后端全量失败记录、`regime_basic_rolling.json`。

### A04 · P1：新版 5 个内置时序指标无法作为下游研究中心的指标节点

**代码位置**：`backend/historical_regimes/indicator_nodes.py:119-136`；`backend/timing_research/catalog.py:35-68,131-142`；`backend/timing_research/graph.py:81-87`。

新版 `rolling_apply` 规范表达式显式包含系统日期和年度配置。下游适配器仍要求表达式中所有变量都是用户连接的一维时序，因而把 `annual_risk_free_rate_decimal` 这种系统标量上下文当作不支持的输入。择时公式校验也仍仅接受直接消费 `rolling_window` 的少数归约步骤。

**独立复现**：把当前内置定义交给真实的情景注册器与择时适配器：

| 当前内置 revision 3 | 情景指标节点 | 择时指标节点 |
|---|---|---|
| 20 日收盘价均线 | 不可用 | 不可用 |
| 20 日布林带 | 不可用 | 不可用 |
| 10 日成交量均线 | 不可用 | 不可用 |
| KDJ | 不可用 | 不可用 |
| 5 日滚动年化夏普 | 不可用 | 不可用 |

共同原因包含：“该指标还需要 年化无风险收益率 上下文，尚不能仅通过数值时序连接计算。”

对照：前四个指标的 revision 2 在择时适配器中可以使用，revision 3 不再可以。平均收益率标量指标经新派生逻辑转换后，也会因相同的系统上下文问题被拒绝。这里审核的是“从指标中心引用为计算节点”的链路，并不表示其他独立行情／快照入口全部失效。

**修复方向**：在适配层分清数值时序输入、系统日期、定义级年度配置和运行参数；复用同一套作用域计划，不让用户手工伪造系统上下文。同步改造下游作用域审计，并验证已锁定历史引用不会指向不同版本。

**证据**：`probe_integration.py/.json` 的 `cross_center`。该问题不等于 A03：修复旧公式允许集合后，新的系统上下文绑定与执行器接入仍需修复。

### A05 · P2：以收益率作日期轴时，滚动多跳过一个有效观察

**代码位置**：`backend/custom_indicators/series_definitions.py:1073-1084`；`backend/custom_indicators/series_provider.py:1828-1829`；`backend/cal_indicators/rolling_scope.py:289-300`。

指标定义允许选择 `returns` 作为日期轴。取数层在这种情况下删掉首个空收益率；滚动引擎却仍假设首行是净值起点对应的空收益，并再次要求窗口起点至少为 1。

**独立复现**：同一净值数据、同一 `rolling_apply(mean(returns),3)`：

- 日期轴为 `adjusted_nav`：2026-01-07 已有三期有效收益，结果约 0.0666667。
- 仅将日期轴改为 `returns`：该日结果变为空，下一日才开始输出。
- 两种定义都保存成功，计算都返回 `ok`，无警告。

**修复方向**：明确每种输入的观察单位与起点偏移，让取数、历史窗口推导、NJIT 和 Excel 使用同一份边界契约。若暂不支持收益率日期轴，应在定义校验时明确拒绝，而不是保存后静默缺失第一笔结果。

**证据**：`probe_indicator_boundaries.json` 的两份 returns 结果。

### A06 · P2：资产大类拟合的非法参数返回 500

**代码位置**：`backend/services/analytics_routes.py:69-75`。

`startDate` 解析失败和 `classes=[]` 都直接抛出 `ValueError`。真实路由挂载到 FastAPI 后，两类请求返回 HTTP 500，而不是参数错误。主应用未注册将这种异常统一转换为参数错误的处理器。

```json
{"startDate":"not-a-date","classes":[]}
{"startDate":"2026-01-02","classes":[]}
```

请求地址：`POST /api/fit-classes`。两者均得到 `Internal Server Error`，无需进入数据库或数值计算即可复现。

**修复方向**：用输入模型校验日期及非空列表，或明确返回 400／422 与字段原因；不要把所有异常一概转为 400，以免掩盖真正的服务错误。

**证据**：`probe_integration.json` 的 `invalid_parameter_requests`。

### A07 · P3：风险模型中心导航缺少系统翻译

**代码位置**：`frontend/src/app/processRegistry.ts:279`；`frontend/src/i18n/catalogs.ts:24-34`；`locales/navigation.json`。

实际合并后的 `builtinCatalogs.system` 缺少 `navigation.routes.settings.risk-models`。按真实合并规则遍历当前导航，仅确认该项缺失。不能只检查 `system.json`，因为导航翻译另由 `navigation.json` 合并。

**影响**：切换语言时该入口依赖原中文回退，系统翻译覆盖不完整。

**证据**：Vitest 的导航覆盖用例失败；`missing_navigation_translations.json`。

## 四、工程与测试问题：与实际业务错误分开

### E01：类型检查没有通过

`frontend/src/components/data-sources/EtlWorkflowEditor.test.tsx:18` 的 `beforeEach(() => vi.resetAllMocks())` 返回了 VitestUtils，与 HookCleanupCallback 类型不符。第 20 行 mock 数据带有接口类型中没有的 `published` 字段。应修复测试代码及接口夹具，不能因为 Vite 构建通过就称类型检查通过。

### E02：3 项 Excel 测试仍断言旧的公式写法

`backend/tests/test_custom_indicator_time_series_excel.py:96,136,165` 分别要求 `STDEVP(`、`STDEV(`、`MIN(`。当前通用区间展开使用了等价统计表达或带掩码的归约，失败首先表现为旧的字符串断言未迁移。

这 3 项不能直接证明 Excel 数值算错，也不能据此断言所有 Excel 重算都正确。更新测试时应核对导出公式的独立重算与数值口径，而不只是改成一个更宽松的 token 搜索。本轮没有把它们计入已确认的 6 项业务问题。

### E03：浏览器测试与新界面／新 DSL 存在契约漂移

最后仍失败的 14 项中，已定位的典型原因是：

- `indicator-canvas.spec.ts` 仍寻找 `rolling_mean` 节点；新定义已经使用真实的 `rolling_apply` 子图，因此对旧节点的查找得到 undefined。
- `indicator-formula-roundtrip.spec.ts` 仍要求 LaTeX 内含 `rolling_std` 旧拼写。
- `indicator-studio.spec.ts` 仍在预览页寻找已被移走的“层级计算 DAG”。
- “添加输出通道”和“关闭”选择器匹配多个按钮，触发严格模式失败；这并不等于用户无法点击具体按钮。
- 部分旧数据源／事件库流程和一个编辑弹窗用例停在元素定位或遮罩交互，需要更新夹具并做针对性人工复核；本轮没有将超时自动定性为生产代码错误。

两项初次失败已排除：颗粒度测试继承了 Python 3.10 环境，改用项目规定的 Python 3.12 后通过；图表全屏 resize 断言在单 worker 复查中通过，暂记为并行时序敏感，不能声称布局缺陷已稳定复现。

### E04：静态 Python 告警与覆盖风险

Ruff 在 backend 下得到 2 条 F821、85 条 F811。F821 分别是 `series_service.py:125` 的 `Iterable` 未导入和测试文件 `test_pit_workflow.py:396` 的 `TestClient`。生产模块使用延迟解析注解，因此前者不能直接等同于当前普通计算路径会抛 NameError。F811 主要位于测试代码，应复核重复定义／导入是否遮蔽了预期覆盖，但不把 85 条静态诊断当作 85 个独立业务错误。根目录 Python 文件的同类扫描无告警。

### E05：路由可复现性门禁未闭合

Hermes 全仓检查有 74 条未跟踪稳定引用诊断。工作区存在大量其他任务的未提交／未跟踪文件，本轮没有为了消除告警而暂存、删除或放宽规则。发布前必须明确提交范围并保证路由引用的源码和测试可从 Git 重现。

## 五、建议修复顺序与验收门槛

1. 优先修复 A01、A02：它们会静默改变结果或知识可得时点。增加数据缺口和受控并发交错测试，验证保存的结果、lineage 与 available_at 来自同一请求。
2. 联合修复 A03、A04：覆盖“指标中心定义 → 情景／择时目录 → 画布 → 准备 → 运行 → 发布门禁”。合法滚动应能使用，真正的全样本未来信息仍应被阻断；历史引用继续锁定原版本。
3. 修复 A05、A06、A07，并更新旧 Excel／浏览器契约测试；再跑全量后端、前端、类型检查与浏览器矩阵。

不能只把测试断言改成通过。每个修复要先加入能重现当前问题的回归，再修改最小相关链路；不为消除无关告警而进行全面重构。

## 六、证据位置与边界

所有原始日志、JSON、JUnit、浏览器 trace 和独立复现均位于项目根目录的 `.tmp_project_audit_20260910/`。

主要入口：

- `run_checks.py`：后端全量、前端全量、TypeScript、构建、国际化、Ruff、路由检查；各任务命令与退出码写入对应 `.status.json`。
- `backend.xml`、`backend.log`、`frontend.json`、`typescript.log`、`build.log`、`routing.log`。
- `browser.json`、`browser_remaining.json`、`browser_rechecks.json`：按 file + title + project 去重，后执行的真实结果覆盖先前结果。
- 四个 `probe_*.py` 脚本及对应 JSON：价格／收益边界、跨中心绑定、并发 lineage、并发 availability。
- `source_before.json`、`source_after.json`、`changed_during_checks.json`：主批次源码一致性证据。

复现脚本使用 AGENTS.md 规定的 Python 3.12 环境；部分需设置 `PYTHONPATH` 包含根目录和 backend，`CUSTOM_INDICATOR_DATA_DIR` 指向临时目录。正式外网连接在后端审计中被禁止；浏览器使用测试夹具或临时真实后端，不执行真实数据写入型验收。

本轮结论限定于当前工作区、已有测试以及被明确检查的关键路径。尚未覆盖真实第三方数据的所有异常组合、生产多进程长时压力、所有移动端视口，以及每一种导出工作簿在实际 Excel 中的重算。没有宣称不存在其他问题，也没有将原型页面当作已完成的生产能力。

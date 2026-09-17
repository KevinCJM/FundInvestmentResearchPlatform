# 产品研究「走势图」重构设计

适用页面：`/product-research/products/:id`（`frontend/src/pages/ProductDetail.tsx`）「走势与指标」标签页。

## 1. 现状与问题

| 位置 | 现在是什么 | 问题 |
| --- | --- | --- |
| 「价格与成交量」卡片 | ETF 画不复权 K 线 + 成交量；场外基金画复权净值折线 | 口径由数据可得性隐式决定，用户无法选择；标题只说「价格」，看不出是不是复权 |
| 「技术辅助线设置」折叠区 | 四个写死的开关 PRICE_MA / VOLUME_MA / BOLL / KDJ | 只有四个内置指标；参数不可改，控件里明文写着「不能在展示页面临时覆盖」；KDJ 的子图在 `chartOption` 里特判 |
| 「研究指标」区块的时序分组 | 每个时序指标一张独立卡片，各自一张图 | 与走势图各画各的，同一条时间轴被拆成 N 张图，无法对齐比较 |
| 「因子研究证据」 | `FactorEvidencePanel` 内联在本页底部 | 因子研究尚不完善，不应出现在产品研究主路径 |
| 「高性能计算已验证」 | `product-analysis-execution` 折叠区，展示内核覆盖率、指纹、Python 回退计数 | 后端执行审计，不是用户要看的信息 |

四个技术开关的本质是「把时序指标画到价格轴上」。用户现在要的就是把这件事一般化：任何时序指标（内置或自定义）都能加进来，能改参数，能选放哪条轴。做完之后那四个开关自然消失——它们本来就在目录里。

## 2. 目标

1. 走势图的价格口径由用户选：**复权净值走势 / 后复权 K 线 / 不复权 K 线（原始行情）**。
2. 走势图可叠加任意时序指标，参数可调（指标中心已开放的参数），位置可选 **同轴同图 / 右轴同图 / 独立子图**。
3. 时序指标只在走势图出现一次；「研究指标」区块只留标量。
4. 删除「因子研究证据」「高性能计算已验证」两块。

## 3. 后端设计

### 3.1 新端点 `GET /api/instruments/products/{product_id}/price-series`

```
?kind=etf|fund&basis=adjusted_nav|adjusted_kline|raw_kline
```

响应：

```jsonc
{
  "product_id": "510300.SH", "kind": "etf", "basis": "raw_kline",
  "label": "不复权 K 线（原始行情）",
  "available": true, "reason": null,
  "points": [{ "date": "2026-09-10", "open": 3.9, "high": 3.95, "low": 3.88, "close": 3.92, "volume": 1234 }],
  "warnings": ["..."],
  "bases": [{ "id": "...", "label": "...", "description": "...", "available": true, "reason": null }],
  "execution": { ... }
}
```

为什么是独立端点而不是给 `/products/{id}` 加参数：切口径不能把整页打回 loading，也不能重跑产品分析。本页详情请求同时加 `include_timeseries=false`，序列只取一次。

三种口径的数据来源：

| basis | 来源 | OHLC | 成交量 | 适用 |
| --- | --- | --- | --- | --- |
| `adjusted_nav` | `etf_daily_df.parquet` / `fund_nav_df.parquet` 的 `adj_nav` | 无（只有 close） | 无 | etf + fund |
| `adjusted_kline` | `etf_daily_candle_df.parquet` 的 `adj_open/adj_high/adj_low/adj_close`，由「ETF 复权价格」ETL 落盘（见 `adjusted_price_indicator_design.md`） | 有 | 有（不复权，见下） | etf |
| `raw_kline` | `etf_daily_candle_df.parquet`，即现有 `_load_timeseries` | 有 | 有 | etf |

口径规则：

- 采用**后复权**，基准固定在每只标的自己的首个交易日。前复权的基准是区间内最后一根 K 线，换个截止日整条历史都会变，本仓库对这种「未来锚」的用法有明确告警；展示图不需要为此付代价。
- **成交量不复权**，与 `backend/timing_research/data.py` 的既有约定一致，并在 warnings 里说明。
- **缺因子即失败，不补 1**：复权列缺失或为空时端点转成 `available:false` + 原因，绝不用 1.0 顶替。覆盖多少取决于 ETL 选的因子口径：默认只用数据源因子（当前快照 2/1778），显式选择前收盘价推导后可覆盖全部 1778 个。
- PIT：三种口径都经过 `_pit_cut(points, _pit_as_of())`。

`bases` 里的可得性是**廉价探测**（文件是否存在、该代码是否有非空复权列），真正的覆盖完整性要到实际计算时才知道；所选口径算不出来时由 `available/reason` 说明。

### 3.2 不做的事

- 不动 `raw_kline` 口径。它展示的是真实成交价，与指标入参是两回事。
- 不动 `/products/{id}` 的既有返回结构（`ProductCompare` 还在用 `timeseries`）。

> 2026-09-16 更新：原文这里写的是「不改指标引擎的价格口径、不新增复权行情变量」。
> 该决定已被 `adjusted_price_indicator_design.md` 取代：变量目录新增了
> `adjusted_open/high/low/close`（`price_basis = adjusted_market`），未复权行情变量
> 已停用；第 4.3 节的语义表同步更新。

## 4. 前端设计

### 4.1 信息结构

```
┌─ 走势图 ─────────────────────────────────────────────────────┐
│ 沪深300ETF · 不复权 K 线 · 2012-05-28 — 2026-09-10 · 3470 根 │
│ 价格口径 (●) 复权净值走势  ( ) 后复权 K 线  ( ) 不复权 K 线    │
├──────────────────────────────────────────────────────────────┤
│ [ 主图：K 线 / 净值折线 + 情景背景 ]                          │
│ [ 成交量 ]                                                    │
│ [ 子图：KDJ ]                                                 │
├──────────────────────────────────────────────────────────────┤
│ 叠加时序指标 2/6                            [ 选择时序指标 ▾ ] │
│──────────────────────────────────────────────────────────────│
│ 滚动波动率 · 工作区 v3          位置[独立子图 ▾]      [移除]  │
│   窗口 20            [应用参数][恢复默认]  实际取值：窗口=20   │
│──────────────────────────────────────────────────────────────│
│ 20 日收盘价均线 · 内置 v1       位置[同轴同图 ▾]      [移除]  │
│   固定参数：需要其他窗口请在指标中心另存一个版本               │
└──────────────────────────────────────────────────────────────┘
```

一个区块、两层结构（section → 列表行），列表用 `divide-y` 而不是卡中卡（准则 6.5 / VISUAL_DENSITY 7）。参数常驻可见，不折叠——数字脱离参数没有意义。

### 4.2 口径选择器

原生 `<input type="radio">` 包在 `<label>` 里，做成分段控件外观。用原生控件是为了白拿键盘与读屏行为；不可用的口径 `disabled` 并在 `title` 与下方一行里给出原因（例如「该产品缺少复权因子」）。

### 4.3 叠加指标的位置规则（由契约推导，不写白名单）

每个通道自带 `semantic_dimension` / `price_basis` / `unit` / `display_format`。主图与成交量图各有自己的语义：

| 图 | 语义 | 什么口径下存在 |
| --- | --- | --- |
| 主图价格轴 | `raw_market_price` | `raw_kline` |
| 主图价格轴 | `adjusted_nav` | `adjusted_nav` |
| 主图价格轴 | `adjusted_market_price` | `adjusted_kline` |
| 成交量图 | `volume` | `raw_kline` / `adjusted_kline` |

- **同轴同图**：通道语义与某条原生轴一致时可选，落到那条轴（价格语义→价格轴，成交量语义→成交量图）。不一致时该选项 `disabled`，并说明「口径与主图不同」。
- **右轴同图**：主图右侧独立刻度，任何指标都可选。多个指标共用右轴时刻度会互相迁就，这是用户自己的选择。
- **独立子图**：每个指标一格，同一指标的多个通道（KDJ 的 K/D/J）共用一格。

默认位置：能共轴的默认共轴，其余默认独立子图。子图不会和任何东西抢刻度，是最不会出错的默认值。

> 2026-09-17 更新：「每个指标一格」已被第 9 节取代。子图归属由**实例**决定，
> 位置选项里多出「并入某条已有子图」，口径一致的多条实例共用一格一轴。

实测的通道元数据（用于校核上表）：

| 指标 | 通道 | unit | semantic_dimension | price_basis |
| --- | --- | --- | --- | --- |
| 20 日复权收盘价均线（v4） | ma | 元 | adjusted_market_price | adjusted_market |
| 20 日布林带（复权，v4） | upper/middle/lower | 元 | adjusted_market_price | adjusted_market |
| 10 日成交量均线 | volume_ma | 份 | volume | — |
| KDJ | k/d/j | — | dimensionless | — |
| 滚动波动率 | value | % | return_decimal | adjusted_nav |
| 5 日滚动年化夏普 | value | — | count | — |

### 4.4 参数

`parameter_contract_version === '1.0'` 且 `parameter_schema` 非空的指标，行内直接渲染既有的 `IndicatorParameterInputs`（编辑不计算，点「应用参数」才重算，卡片回显服务端实际取值）。其余显示「固定参数」一行，并指向指标中心。参数不落 localStorage：选了哪些指标是「我常看什么」，参数值是「我现在问什么」。

### 4.5 图表装配

- grid：主图（必有）→ 成交量（口径带成交量时）→ 每个「独立子图」指标一格。KDJ 的特判随之删除。
- yAxis：主图左轴；有指标选「右轴同图」时给主图加一条 `position:'right'` 的轴；成交量轴；每个子图一条。
- dataZoom 串联所有 x 轴，默认窗口沿用现有规则（最近 252 根，或研究上下文/所选区间）。
- percent 通道乘 100 并给 `{value}%` 轴标签，与 `SeriesIndicatorCard` 原有处理一致。
- 颜色只取代码库里已有的十六进制值，不引入新色值（`design:check` 的 `bare-chart-hex` 统计的是去重后的色值个数）。

### 4.6 逐文件改动

| 文件 | 改动 |
| --- | --- |
| `backend/services/instrument_routes.py` | 新增 `/products/{id}/price-series` 与三个口径的读取；详情端点加 `include_timeseries` |
| `backend/tests/test_instrument_routes.py` | 三种口径、缺因子失败、PIT 截断的回归 |
| `frontend/src/services/productAnalysis.ts` | `fetchProductPriceSeries` 与类型 |
| `frontend/src/components/product-research/ProductTrendChart.tsx` | 新增：口径、叠加指标、参数、位置、ECharts 装配 |
| `frontend/src/pages/ProductDetail.tsx` | 删除 `chartOption` / `overlayOptions` / `SERIES_INDICATOR_IDS` / `renderOverlayControls` / `alignedChannelValues` / `buildRegimeMarkAreas` / `TimeSeriesPoint` 等约 630 行；删除 `FactorEvidencePanel` 与 `product-analysis-execution`；目录按 `applicable_product_kinds` 过滤 |
| `frontend/src/components/product-research/ProductTrendChart.test.tsx` | 口径切换、缺因子禁用、参数应用与迟到响应、共轴规则 |
| `locales/system.json` | 删除随 `SeriesIndicatorCard` 失效的 `indicatorParameters.choose` / `.customSeries` / `.period` |
| `frontend/src/components/metrics/ResearchIndicatorPanel.tsx` | 只留标量 |
| `frontend/src/components/indicator-parameters/SeriesIndicatorCard.tsx` | 删除（能力并入走势图） |

### 4.7 交互状态

加载 / 空 / 错误 / 禁用四态齐备：序列加载中、指标计算中、目录为空、未选指标、口径不可用、指标对本产品不适用（按 `applicable_product_kinds` 过滤后不出现在选择器里）。

## 5. 自测

- `backend/tests/test_instrument_routes.py`
- `frontend`：`vitest run`、`tsc --noEmit`、`npm run build`
- `node scripts/check_i18n.mjs`、`npm run design:check`
- `skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py`

## 6. 保留的承诺

原来「原始数据未完整披露 … 不使用 close 或 0 伪造缺失字段」的提示随旧图一起删掉会丢掉一条真实保证，改挂在走势图上：K 线口径下缺开高低价时退回收盘价折线并说明原因。`product-analysis-execution` 里「页面不会退回浏览器本地计算」一句移到分析失败的 `role="alert"` 上，其余 NJIT 审计字段不再展示。

## 7. 自测记录

| 检查 | 结果 |
| --- | --- |
| `pytest backend/tests/test_instrument_routes.py test_product_compare_routes.py` | 32 passed |
| `pytest backend/tests/test_product_analysis_routes.py` | 19 passed |
| `pytest backend/tests/test_series_runtime_parameters.py test_custom_indicator_routes.py test_independent_snapshot_execution.py` | 68 passed |
| `vitest run`（全量） | 136 files / 1013 tests passed |
| `tsc --noEmit`、`vite build` | 通过 |
| `npm run design:check` | 无回归（`uppercase-on-cjk` 11、`bare-chart-hex` 57 维持既有超标值，未新增色值） |
| `node scripts/check_i18n.mjs` | valid |
| `validate_ai_routing.py` | passed |
| `playwright e2e/metric-display.spec.ts`（320 / 768 / 1440） | 3 passed |
| `playwright e2e/product-scenario.spec.ts`、`e2e/historical-regime-workbench.spec.ts` | 1 + 4 passed |

## 8. 明确不做

- ~~不做同一指标多组参数并排叠加（一个指标在图上只有一份参数）。~~
  **2026-09-17 取消**：同一套指标换一组参数就是另一条曲线，正是要放在一张图上比的东西。见第 9 节。
- 不做前复权。
- 不持久化叠加指标选择与参数（与现有 `selectedOverlays` 的行为一致）。
- 不改 `ProductCompare`。

## 9. 同一指标的多条实例（2026-09-17）

### 9.1 要解决的问题

一套时序指标换一组参数就是另一条曲线，20 日均线和 60 日均线要放在一张图上才有意义。
改造前「叠加时序指标」的状态是三个以 `indicator.id` 为键的表（选中、参数、位置），
同一个指标只能有一份参数、一个位置，结果也只能按 `indicator_id` 认领。

### 9.2 后端：结果按实例键认领，而不是按位置或指标 ID

`evaluate_series` 本来就按 `indicator_instances` 逐条解析参数、逐条算、逐条产出，
顺序与请求一致（`custom_indicators/series_service.py`）。缺的只是让这条契约显式：

- `SeriesIndicatorInstance` 新增可选的 `instance_key`（≤64 字符，调用方自己起名）。
- `evaluate` 在装配响应时按请求顺序把 `instance_key` 回写到每条结果上。
- 同一时刻**顺带修掉一个别名隐患**：参数完全相同的两条实例命中同一份缓存，
  改造前 `results` 里会出现同一个 dict 对象两次，后续逐条写入（如 `data_context`）
  等于往一个对象里写两遍。回写时改为每条实例各持一个浅拷贝，通道数组仍然共享，不复制数值。
- `zip(..., strict=True)`：结果条数与实例条数一旦不等立即失败，不做静默截断。

不改的：算子、缓存键（本来就含参数）、
其他调用方（指标中心预览、Excel 导出各自只发一条实例，`instance_key` 为 `None`）。
（实例上限当时保持 `MAX_SERIES_INSTANCES = 10`，已被第 10 节取代。）

### 9.3 前端：一条实例一行

状态从三个 id 键表收敛为一个数组，`ProductTrendChart.tsx`：

```ts
interface OverlayInstance {
  key: string                       // 实例键，同时是发给后端的 instance_key
  indicatorId: string
  parameters: Record<string, number>
  placement?: 'native' | 'right' | `panel:<owner key>`
}
```

- **位置是一个字段而不是两个。** `panel:<owner key>` 里的 owner 是"这一格归谁"。
  指向自己＝独占一格，指向别人＝并进那一格。渲染时按 owner 分组，一组一个 grid、一条纵轴。
- **只允许一跳。** 只有"自己占一格"的实例可以被并入，因此不会出现 A→B→C 的链；
  宿主被移除或改了位置时，跟随者在推导里自动退回自己的一格，不需要清理副作用。
- **能不能并格由通道口径决定，不看指标是谁**（`axisIdentity`：`semantic_dimension|unit|display_format`）。
  同一指标的不同参数天然一致；跨指标只要口径一致也可以并。口径不一致的实例不出现在选项里。
- **名字必须唯一。** ECharts 图例按名字联动，两条同名曲线会一起开关。
  重复指标的行名与图例名带上实际计算参数（`可调均线 (窗口期数=60)`），
  仍然重名时按出现顺序补 `#2`；只有一条实例时的名字与改造前完全一致。

### 9.4 界面（第 12 节清单相关条目）

- 一行一条实例，`divide-y`，不套卡片（准则 6.5 / `VISUAL_DENSITY: 7`）。
- 位置是一个原生 `<select>`，用 `<optgroup>` 分成「画在主图」「画在子图」两组，
  并入选项直接写宿主的名字：`并入「可调均线 (窗口期数=20)」的子图`。原生控件白拿键盘与读屏行为。
- 「再加一条」按钮复制当前行的参数并**继承它的位置**，两条默认叠在一起——
  要比的就是它们；要拆开再改位置。
- 禁用态说明原因：固定参数的指标不给复制（再画一条也是同一条线）。
  （当时还有"画满 6 条"的上限文案与 `n/6` 计数，已被第 10 节取代。）
- 右轴的既有行为（多条共用、刻度互相迁就）保留，但**混口径时右轴不再标单位、不再套 `%` 格式**，
  并在行下方说明；原来按第一条的通道取单位，等于给另一条的刻度贴了错标签。

### 9.5 自测

| 检查 | 结果 |
| --- | --- |
| `pytest tests/test_series_runtime_parameters.py test_custom_indicator_routes.py test_custom_indicator_time_series.py test_independent_snapshot_execution.py` | 87 passed（新增 2 条：实例键回领与别名隔离、路由透传与可选性） |
| `vitest run`（全量，`--maxWorkers=2`） | 147 files / 1156 tests passed |
| `ProductTrendChart.test.tsx` | 13 passed（新增 4 条：多实例参数独立与序号兜底、并图与拆图、口径不一致不给并、上限说明） |
| `tsc --noEmit`、`vite build` | 通过 |
| `npm run design:check` | 无回归 |
| `node scripts/check_i18n.mjs` | valid |
| `validate_ai_routing.py`、`evolve_ai_routing.py` | passed / 16 files covered |
| 浏览器（真实后端，1440 / 768 / 320） | 同一指标两条参数各自出曲线；默认并图、可拆成两格、可再并回一格；KDJ（无量纲）不出现在百分比实例的并入选项里；固定参数指标「再加一条」禁用并说明；三个视口无横向溢出、无按钮被卡片裁掉；控制台无报错 |

浏览器验收顺带修掉的两处既有缺陷：叠加曲线的图例色块取自 `itemStyle`，原来只设 `lineStyle`，
色块与实际线色对不上（两条曲线要靠颜色区分时正是这里最要紧）；行内控件由三个变四个后在 320px
会被卡片 `overflow-hidden` 裁掉，改为可换行。

## 10. 取消条数上限与列表信息密度（2026-09-17）

### 10.1 条数上限

前端的 `MAX_TREND_OVERLAYS = 6` 是产品层面的限制，没有技术理由，删除：
「再加一条」不再有数量禁用态，区块标题只报当前条数。
`MetricSelector` 的 `maxSelected` 同时改为可选——不传就是不限：
计数显示成「已选 6」而不是「已选 6/6」，后者读起来还是一个配额。
其余调用方（研究指标、比较指标、评估方案等）都显式传了自己的上限，行为不变。

后端保留请求边界但不再是产品限制：`MAX_SERIES_INSTANCES` 10 → 50，
并让路由模型 `EvaluateSeriesRequest.indicator_instances` 直接引用这个常量，
不再各写各的字面量。一次请求仍然有界，坏调用方无法排队无限次计算。

### 10.2 列表信息密度

指标一多就要反复下拉，删掉的是读者不需要的内容，不是功能：

| 原来每行都印 | 现在 |
| --- | --- |
| `内置 v4 · 20 日复权收盘价均线`（来源、版本、通道名，多半重复指标名） | 移入行名的悬停提示 |
| `此版本未开放可变参数，按固定公式计算。` | 删除：没有输入框本身就是答案，理由留在「再加一条」的禁用提示里 |
| `口径与主图不同，不能与价格共轴；请用右轴或独立子图。` | 移到那个不可选的选项上：`同轴同图（口径与主图不同，不可选）`，理由贴在被禁用的控件本身（准则 8.4） |
| 参数区：标题 + 标签 + 整行输入框 + `[1, 20000] · 默认值 20 · 步长 1` + 两个按钮 + 实际参数，约 6 行 | 一行：`本次计算参数 窗口期数 [20] [1, 20000] … 实际计算参数: 窗口期数=20` |

参数区（`IndicatorParameterInputs`，指标中心与研究指标表格共用）的三条规则：

- **按钮只在能做事的时候出现**：输入与已应用值不一致才有「应用参数」，
  当前值不等于默认值才有「恢复默认」。少一个常驻的灰按钮，也少一句"参数已修改"的提示——
  按钮出现本身就是提示，输入框里回车等同于点它。
- **「实际计算参数」只在和输入框矛盾时出现**：它是服务端真正算的那组值。
  改了没应用（或服务端做了钳制）时，它与输入框里的数字并排且不一致，
  这个对照比一句说明更直接；两边一致时输入框本身就是答案，再印一遍只是噪声。
- **参数长在指标名旁边**：整块参数放进指标行本来就空着的中段，
  控件簇用 `ml-auto` 继续贴右。带参数的行因此也是一行。

浏览器实测行高：**每行 56–57px**，带参数的行不再多占一行（改造前固定参数约 110px、带参数约 330px）。

列表加 `border-t`：`divide-y` 只画行与行之间，标题和第一行之间原来没有分隔线。

同日另删：走势图标题里的「N 个观察值」。起止日期已经说明了区间，条数对读图没有用。

### 10.3 自测

| 检查 | 结果 |
| --- | --- |
| `pytest tests/test_series_runtime_parameters.py test_custom_indicator_routes.py test_custom_indicator_time_series.py test_custom_indicator_time_series_excel.py` | 87 passed |
| `vitest run`（全量） | 147 files / 1156 tests passed |
| `tsc --noEmit`、`vite build`、`design:check`、`check_i18n`、AI routing | 通过 / 无回归 |
| 浏览器（真实后端，1440 / 768 / 320） | 目录全选后选择器无禁用项、显示「已选 6」；复制到 12 条（超过原前端 6 条与原后端 10 条上限）全部出图，无错误横幅、无控制台报错；固定参数行高 56px、带参数行高 105px；不可共轴的原因印在那个选项上；三个视口无横向溢出、无控件被卡片裁掉 |

320px 顺带修掉：位置下拉的 `max-w-[14rem]` 在窄屏挤出卡片——选项文字变长后更明显。
下拉改为 `min-w-0 flex-1`，可以缩到可用宽度以内。

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
| `adjusted_kline` | `etf_daily_candle_df.parquet` OHLC × `fund_adj_factor_df.parquet` 的 `adj_factor`，复用既有 NJIT `adjusted_price_kernel(prices, factors, forward=0)` | 有 | 有（不复权，见下） | etf |
| `raw_kline` | `etf_daily_candle_df.parquet`，即现有 `_load_timeseries` | 有 | 有 | etf |

口径规则：

- 采用**后复权**（`forward=0`，基准固定为 1.0）。前复权的基准是区间内最后一根 K 线，换个截止日整条历史都会变，本仓库对这种「未来锚」的用法有明确告警；展示图不需要为此付代价。
- **成交量不复权**，与 `backend/timing_research/data.py` 的既有约定一致，并在 warnings 里说明。
- **缺因子即失败，不补 1**：`adjusted_price_kernel` 遇到缺失或非正因子直接抛错，端点转成 `available:false` + 原因，绝不用 1.0 顶替。当前快照里只有 2/1778 个 ETF 有复权因子，所以这条路径在真实数据上大多数时候会明确不可用——这正是要显式说出来的事。
- PIT：三种口径都经过 `_pit_cut(points, _pit_as_of())`。

`bases` 里的可得性是**廉价探测**（文件是否存在、该代码是否有因子行），真正的覆盖完整性要到实际计算时才知道；所选口径算不出来时由 `available/reason` 说明。

### 3.2 不做的事

- 不改指标引擎的价格口径。`market_close` 等变量仍是不复权行情，`adjusted_nav` 仍是复权净值——正因为契约里分得清，前端才能判断哪条曲线能和主图共轴。
- 不新增「复权行情」变量。目录里没有，就不假装有。
- 不动 `/products/{id}` 的既有返回结构（`ProductCompare` 还在用 `timeseries`）。

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
| 主图价格轴 | （无匹配语义） | `adjusted_kline`：复权行情在变量目录里没有对应语义，任何指标都不能共轴 |
| 成交量图 | `volume` | `raw_kline` / `adjusted_kline` |

- **同轴同图**：通道语义与某条原生轴一致时可选，落到那条轴（价格语义→价格轴，成交量语义→成交量图）。不一致时该选项 `disabled`，并说明「口径与主图不同」。
- **右轴同图**：主图右侧独立刻度，任何指标都可选。多个指标共用右轴时刻度会互相迁就，这是用户自己的选择。
- **独立子图**：每个指标一格，同一指标的多个通道（KDJ 的 K/D/J）共用一格。

默认位置：能共轴的默认共轴，其余默认独立子图。子图不会和任何东西抢刻度，是最不会出错的默认值。

实测的通道元数据（用于校核上表）：

| 指标 | 通道 | unit | semantic_dimension | price_basis |
| --- | --- | --- | --- | --- |
| 20 日收盘价均线 | ma | 元 | raw_market_price | raw_market |
| 20 日布林带 | upper/middle/lower | 元 | raw_market_price | raw_market |
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

- 不做同一指标多组参数并排叠加（一个指标在图上只有一份参数）。
- 不做前复权。
- 不持久化叠加指标选择与参数（与现有 `selectedOverlays` 的行为一致）。
- 不改 `ProductCompare`。

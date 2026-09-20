# 产品走势图与时序指标

本专业契约负责价格口径、时序实例、轴分配和异步批次。产品研究入口见[指标表](README.md)。
## 后端设计

### 新端点 `GET /api/instruments/products/{product_id}/price-series`

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
- **缺因子即失败，不补 1**：复权列缺失或为空时端点转成 `available:false` + 原因，绝不用 1.0 顶替。覆盖多少取决于 ETL 选的因子口径：由 ETL 的 factor_policy 决定，当前默认 source_then_pre_close；2/1778 与推导后1778/1778属于原验收快照，不能视为当前覆盖率。
- PIT：三种口径都经过 `_pit_cut(points, _pit_as_of())`。

`bases` 里的可得性是**廉价探测**（文件是否存在、该代码是否有非空复权列），真正的覆盖完整性要到实际计算时才知道；所选口径算不出来时由 `available/reason` 说明。

### 不做的事

- 不动 `raw_kline` 口径。它展示的是真实成交价，与指标入参是两回事。
- 不动 `/products/{id}` 的既有返回结构（`ProductCompare` 还在用 `timeseries`）。

指标变量使用后复权行情列，原始行情仍可展示；完整变量契约见[复权价格](../data/adjusted-price.md)。

## 前端设计

### 信息结构

走势图统一承载价格、成交量、情景背景和时序叠加；指标一实例一行，参数常驻，编辑后显式应用。共享设计令牌和两层信息结构，不复制另一套指标卡片。

### 口径选择器

原生 `<input type="radio">` 包在 `<label>` 里，做成分段控件外观。用原生控件是为了白拿键盘与读屏行为；不可用的口径 `disabled` 并在 `title` 与下方一行里给出原因（例如「该产品缺少复权因子」）。

### 叠加指标的位置规则（由契约推导，不写白名单）

每个通道自带 `semantic_dimension` / `price_basis` / `unit` / `display_format`。主图与成交量图各有自己的语义：

| 图 | 语义 | 什么口径下存在 |
| --- | --- | --- |
| 主图价格轴 | `raw_market_price` | `raw_kline` |
| 主图价格轴 | `adjusted_nav` | `adjusted_nav` |
| 主图价格轴 | `adjusted_market_price` | `adjusted_kline` |
| 成交量图 | `volume` | `raw_kline` / `adjusted_kline` |

- **同轴同图**：通道语义与某条原生轴一致时可选，落到那条轴（价格语义→价格轴，成交量语义→成交量图）。不一致时该选项 `disabled`，并说明「口径与主图不同」。
- **右轴同图**：主图右侧独立刻度，任何指标都可选。多个指标共用右轴时刻度会互相迁就，这是用户自己的选择。
- **独立子图**：按实例分格，多通道共格；口径一致的其他实例可显式并入。

默认位置：能共轴的默认共轴，其余默认独立子图。子图不会和任何东西抢刻度，是最不会出错的默认值。

实测的通道元数据（用于校核上表）：

| 指标 | 通道 | unit | semantic_dimension | price_basis |
| --- | --- | --- | --- | --- |
| N 日复权收盘价均线 | ma | 元 | adjusted_market_price | adjusted_market |
| N 日布林带（复权） | upper/middle/lower | 元 | adjusted_market_price | adjusted_market |
| N 日成交量均线 | volume_ma | 份 | volume | — |
| N 日 KDJ | k/d/j | — | dimensionless | — |
| 滚动波动率 | value | % | return_decimal | adjusted_nav |
| N 日滚动年化夏普 | value | — | count | — |

### 参数

内置时序指标都开放一个 `window`（窗口期数，2–1000，默认沿用原固定值），名称为「N 日…」。

`parameter_contract_version === '1.0'` 且 `parameter_schema` 非空的指标，行内直接渲染既有的 `IndicatorParameterInputs`（编辑不计算，点「应用参数」才重算，卡片回显服务端实际取值）。其余显示「固定参数」一行，并指向指标中心。参数不落 localStorage：选了哪些指标是「我常看什么」，参数值是「我现在问什么」。

### 图表装配

- grid：主图（必有）→ 成交量（口径带成交量时）→ 每个独立宿主实例一格。KDJ 的特判随之删除。
- yAxis：主图左轴；有指标选「右轴同图」时给主图加一条 `position:'right'` 的轴；成交量轴；每个子图一条。
- dataZoom 串联所有 x 轴，默认窗口沿用现有规则（最近 252 根，或研究上下文/所选区间）。
- percent 通道乘 100 并给 `{value}%` 轴标签，与 `SeriesIndicatorCard` 原有处理一致。
- 颜色只取代码库里已有的十六进制值，不引入新色值（`design:check` 的 `bare-chart-hex` 统计的是去重后的色值个数）。

### 交互状态

加载 / 空 / 错误 / 禁用四态齐备：序列加载中、指标计算中、目录为空、未选指标、口径不可用、指标对本产品不适用（按 `applicable_product_kinds` 过滤后不出现在选择器里）。

## 同一指标的多个实例
### 后端：结果按实例键认领，而不是按位置或指标 ID

`evaluate_series` 按 `indicator_instances` 逐条解析参数、逐条算、逐条产出，
顺序与请求一致（`custom_indicators/series_service.py`）。实例契约：

- `SeriesIndicatorInstance` 提供可选的 `instance_key`（≤64 字符，调用方自己起名）。
- `evaluate` 在装配响应时按请求顺序把 `instance_key` 回写到每条结果上。
- 相同参数实例可共享缓存和通道数组，但每条结果拥有独立浅拷贝，逐条写元数据不会互相覆盖。
- `zip(..., strict=True)`：结果条数与实例条数一旦不等立即失败，不做静默截断。

不改的：算子、缓存键（本来就含参数）、
其他调用方（指标中心预览、Excel 导出各自只发一条实例，`instance_key` 为 `None`）。

### 前端：一条实例一行

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

### 界面

- 一行一条实例，`divide-y`，不套卡片（准则 6.5 / `VISUAL_DENSITY: 7`）。
- 位置是一个原生 `<select>`，用 `<optgroup>` 分成「画在主图」「画在子图」两组，
  并入选项直接写宿主的名字：`并入「可调均线 (窗口期数=20)」的子图`。原生控件白拿键盘与读屏行为。
- 「再加一条」按钮复制当前行的参数并**继承它的位置**，两条默认叠在一起——
  要比的就是它们；要拆开再改位置。
- 禁用态说明原因：固定参数的指标不给复制（再画一条也是同一条线）。
- 右轴的既有行为（多条共用、刻度互相迁就）保留，但**混口径时右轴不再标单位、不再套 `%` 格式**，
  并在行下方说明；原来按第一条的通道取单位，等于给另一条的刻度贴了错标签。

## 数量、分批与加载期间复制

界面允许多条实例，名称由用户区分；不恢复旧产品层条数限制。单次时序请求最多 50 条，按顺序分批，全部成功后一次更新结果；按 instance_key 匹配，不能依赖响应顺序。中途失败不把部分结果当作完整完成，失效请求或卸载不能启动剩余批次。

加载期间复制继承显式位置，并保存默认子图宿主。通道返回后，有兼容原生轴就叠加，否则经口径校验共用宿主；连续复制沿用同一宿主，显式位置优先。宿主消失或口径不兼容按原规则独立显示。

请求键包含产品、类型、实例、锁定修订、参数及日期口径。复制、重命名或图表布局不修改源指标；参数改变必须重新计算。

## 验证

保留 50／51／101 条、乱序响应、后批失败、卸载取消、慢响应连续复制、右轴／子图兼容反例。软件测试不代表任意大规模渲染性能。历史结果见[工程纪要](../verification/engineering.md)。

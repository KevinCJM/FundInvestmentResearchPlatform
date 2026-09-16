# 复权价格口径：数据、ETL 与指标入参改造设计

日期：2026-09-16。

## 1. 问题

指标中心的技术类指标（均线、布林带、KDJ、最高/最低价）建立在 `market_close` /
`market_high` / `market_low` 上，这些是交易所未复权报价。未复权报价在分红、份额
折算当日会出现与真实收益无关的跳变：513100.SH 在 2022-01-14 的 5:1 折算让未复权
收盘价从 5.192 跌到 1.015（−80.5%），而真实收益是 −2.2%。任何跨越该日的窗口统计
都是错的。

全库 1778 只 ETF 中 340 只存在复权事件；其余 1438 只复权价与未复权价恒等，问题
不可见，但一旦发生就是灾难性的。

同时，变量目录里没有复权 OHLC，走势图的「后复权 K 线」口径找不到任何同语义的
指标通道，因此叠加指标的「同轴同图」永远是灰的。

## 2. 目标

1. 复权因子成为行情表的日频字段，来源明确、可追溯。
2. 复权 OHLC 由 ETL 依据因子产出，不在查询时临时计算。
3. 指标入参只提供复权口径；没有因子的标的，依赖复权口径的指标明确报为不可计算，
   不用未复权价格冒充。

## 3. 数据契约

### 3.1 行情表新增列（`etf_daily_candle_df.parquet`）

| 列 | 类型 | 含义 |
| --- | --- | --- |
| `adj_factor` | float64 | 后复权因子，每只标的首个交易日固定为 1.0 |
| `adj_factor_source` | string | `source`（数据源因子）或 `pre_close`（推导）；无因子时为空 |
| `adj_open` / `adj_high` / `adj_low` / `adj_close` | float64 | 未复权 OHLC × 当日因子 |

因子和复权价与净值、OHLC 同表同行，避免跨表对齐；净值侧的复权口径仍由
`etf_daily_df.adj_nav` / `fund_nav_df.adj_nav` 承担，本次不动。

### 3.2 因子口径

后复权，基准在每只标的自身的首个交易日。两条来源都必须归一到同一基准才可比：

- 数据源：`F[t] = adj_factor_raw[t] / adj_factor_raw[first]`。
  只有整段行情日期都有因子的标的才采用；覆盖不全的标的整体留空，不做插值。
- 前收盘价推导：`F[0] = 1`，`F[t] = F[t-1] · close[t-1] / pre_close[t]`。

推导式成立的前提是 Tushare `pre_close` 为除权调整后的前收盘价。该前提已核验：
`close[t] / pre_close[t] - 1` 与 `pct_chg/100` 在 1,577,335 行上无例外；由此得到的
`adj_close` 逐日收益与 `close/pre_close - 1` 相差 4.44e-16（浮点噪声）。

任一天的比值缺失或非正，则该标的从那天起因子整段作废——`cumprod` 默认会跳过缺口
继续累乘，等于用未除权比值冒充复权，必须显式截断。

### 3.3 口径选择由用户在 ETL 定义时决定

新 ETL 任务「ETF 复权价格」(`tushare.price_adjustment`) 带一个必填参数
`factor_policy`：

| 取值 | 含义 |
| --- | --- |
| `source`（默认） | 只用数据源因子；数据源没有覆盖的标的不产出复权价格 |
| `source_then_pre_close` | 数据源覆盖的用数据源，其余用前收盘价推导 |
| `pre_close` | 全部由前收盘价推导，用于核验两条路径是否一致 |

默认从数据源取。数据源没有覆盖时，用户必须显式选择推导口径；不选就没有复权 OHLC，
下游据此报不可计算。当前快照里官方 `fund_adj_factor_df.parquet` 只覆盖 2 只
（510300.SH、513500.SH），因此默认口径下 1776 只标的没有复权价格——这是如实暴露
数据缺口，不是静默降级。

两条路径同时可用时记录相对偏离：当前数据下 6542 行可比，最大相对差 2.27e-03。

### 3.4 任务位置

`price_adjustment` 是本地派生任务（`api_slots` 为空、`network: False`），依赖
`candle`，在 `fund_adjustment` 之后执行，因此同一次运行内刷新的数据源因子当场可见。
因子是累乘量，每次运行整表重算，不做增量拼接。

## 4. 指标入参目录

### 4.1 新增

| 变量 | 语义 | 价格基准 | 数据来源 |
| --- | --- | --- | --- |
| `adjusted_open` / `adjusted_high` / `adjusted_low` / `adjusted_close` | `adjusted_market_price` | `adjusted_market` | `etf_daily_candle_df.adj_*` |

`adjusted_market` 与 `adjusted_nav` 是**两个不同的价格基准**，编译期禁止混用。两者
都含分红再投资，但场内价相对净值有折溢价，两条路径的日收益并不相等（差异项为
溢价率变动与因子比，恒等式残差 6.66e-16）。把它们当成一个口径会引入真实误差。

### 4.2 停用

`market_open`、`market_high`、`market_low`、`market_close`、`previous_close`、
`price_change`、`price_return`、`unit_nav`、`accumulated_nav` 标记为 `retired`：

- 从 `variable_catalog()` 移除：界面不再提供这些入参。
- 从 `allowed_variables()` 移除：新建或保存引用它们的公式直接失败，错误码
  `VARIABLE_RETIRED`，消息直接给出替代变量。
- 保留在 `variable_types()`：编译器仍认识它们，已保存的指标定义和冻结的内置
  revision 可以原样重放，历史结果不会被静默改写。

`accumulated_nav` 被停用的额外理由：累计净值是把分红**加**回去而不是再投，
`accum_nav[t]/accum_nav[t-1]-1` 在数学上不是收益率。

`VARIABLE_REGISTRY_VERSION` 从 `2.1.0` 升到 `3.0.0`。

## 5. 内置指标

### 5.1 时序（新增 revision 4）

均线、布林带、KDJ 的 v4 改用 `adjusted_close` / `adjusted_high` / `adjusted_low`，
日期轴锚定 `adjusted_close`。成交量均线的 v4 公式不变，但日期轴改锚 `volume`
自身——成交量不参与复权，锚在复权收盘价上会让它在缺因子时一起不可计算。

v1–v3 保持未复权口径不变。它们是重放契约，不是并行实现；仓库既有的
`test_all_ten_historical_definition_hashes_are_unchanged` 继续守住公式不漂移
（`variable_registry_version` 随本次升级写入，哈希同步更新）。

### 5.2 标量（替换）

| 原 | 新 |
| --- | --- |
| `highest-market-price-v2` `max_value(market_high)` | `highest-adjusted-price-v3` `max_value(adjusted_high)` |
| `lowest-market-price-v2` `min_value(market_low)` | `lowest-adjusted-price-v3` `min_value(adjusted_low)` |
| `market-high-low-range-v2` | `adjusted-high-low-range-v3` |

标量内置没有 revision 机制，版本由 id 后缀表达，因此是替换而非并列。三者在
`BUILTIN_METRIC_IDS` 中的位置不变，融合 NJIT 核按位置编码，核代码不受影响。

## 6. 走势图

「后复权 K 线」不再在请求时用因子乘一遍 OHLC，改为直接读 `adj_*` 列；
`_adjustment_factors` 及其 NJIT 调用一并删除。缺列或缺值时仍然失败关闭，提示
用户去数据中心运行「ETF 复权价格」。

`ProductTrendChart` 的同轴判定表补上 `adjusted_kline → adjusted_market_price`，
后复权 K 线上的复权口径指标因此可以选择「同轴同图」。判定仍然来自通道语义，
不是 id 白名单。

## 7. 不做的事

- 不动净值侧：`adj_nav` 的口径、PIT 规则、`ann_date` 截断全部保持原样。
- 不删 `raw_kline` 走势图口径：那是展示真实成交价，不是指标入参。
- 不改 `timing_research` 与 `research_series` 现有的因子读取路径：它们仍走
  `fund_adj_factor_df.parquet`，本次不在范围内。
- 不为溢价率新增变量：它需要未复权收盘价与单位净值，与本次「入参只留复权」的
  决定冲突，留待单独设计。

## 8. 自审核与验收

### 8.1 因子推导的正确性（真实快照，1,579,113 行 / 1778 只 ETF）

| 检查 | 结果 |
| --- | --- |
| 推导因子覆盖 | 1778/1778 只标的全程无缺口 |
| 存在复权事件的标的 | 340 只；其余 1438 只复权价恒等于未复权价 |
| `adj_close` 逐日收益 vs `close[t]/pre_close[t]-1` | 最大偏差 **4.44e-16**（浮点噪声） |
| 数据源因子 vs 前收盘价推导 | 6542 行可比，最大相对差 **2.27e-03**（已作为 stats 输出，不静默吞掉） |
| 份额折算样本 513100.SH 2022-01-14 | 未复权 5.192→1.015（−80.5%）；复权 5.153→5.039（−2.2%） |

`pct_chg` 与 `close/pre_close-1` 本身有 5.5e-05 的供应商舍入差，那是数据源自己的精度，不是推导误差。

### 8.2 契约级检查

- 56 个当前内置指标（时序 + 标量）全部不再引用已停用变量；程序化枚举 `required_variables + axis_anchor` 逐一核对。
- 新公式引用 `market_close` 在校验、创建、模板组合三条入口一致失败，错误码 `VARIABLE_RETIRED`，消息直接给出 `adjusted_close`。
- `adjusted_close - adjusted_nav` 在编译期被 `SEMANTIC_DIMENSION_MISMATCH` 拒绝：场内复权价与场外复权净值差一个折溢价，不是同一个量。
- 内置时序 v1/v2 的冻结哈希只因 `variable_registry_version` 字段变化而更新；已用独立 worktree 与改动前逐字段比对确认公式、通道、方法论均未改写。

### 8.3 失败关闭

- 复权因子缺失 → 复权列留空 → 指标报 `SOURCE_FIELD_MISSING`；走势图「后复权 K 线」报不可用并指向该 ETL 步骤。
- 前收盘价某天不可用 → 该标的其后因子整段作废，不跳过缺口继续累乘。
- `factor_policy=source` 且没有 `fund_adj_factor_df.parquet` → ETL 直接失败，不退回推导。

### 8.4 执行的检查

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 后端专项（29 个套件） | `pytest tests/test_price_adjustment.py tests/test_custom_indicator_*.py tests/test_builtin_series_rolling_migration.py tests/test_typed_indicator_*.py tests/test_indicator_*.py tests/test_rolling_interval_*.py tests/test_instrument_routes.py tests/test_tushare_data_script.py tests/test_data_refresh.py …` | 810 passed |
| 参数与导出 | `pytest tests/test_series_runtime_parameters.py tests/test_custom_indicator_excel_export.py` | 139 passed |
| 前端全量 | `npx vitest run` | 136 文件 / 1031 tests passed |
| 全量类型检查 | `npx tsc --noEmit` | 0 项诊断 |
| 生产构建 | `npm run build` | 通过 |

本记录只声明上列检查通过，不代表全部后端测试或部署验收通过。本轮未提交、未推送；正式快照的改动见 8.5。

### 8.5 正式快照发布（2026-09-16）

`tushare_snapshot_20260910_merrill_macro01` 已按 `source_then_pre_close` 口径产出复权列并重新激活：

| 步骤 | 结果 |
| --- | --- |
| `save_price_adjustment(snapshot, "source_then_pre_close")` | 1,579,113 行；1776 只走 pre_close 推导，2 只走数据源因子；340 只有复权事件；0 行无因子 |
| 原有 13 列逐列比对 | 全部未变；新增 6 列；行数不变 |
| `rebuild_analytics_snapshot(snapshot, workspace_data_dir=data)` | 30,954 行，8 个配置指标，error=0 |
| `activate_tushare_snapshot(...)` | validation=passed，manifest 记录新的 61,432,261 字节 |
| 走势图后复权 K 线 | 513100.SH 3236 点、510300.SH 3470 点，均可用 |
| 时序内置 v4（513100.SH，1Y） | 4/4 ok，`python_fallback=0`，通道语义 `adjusted_market_price` |
| 标量复权高低价 v3 | 3/3 ok |
| 清理 | 已保存的自定义指标「N日收盘价均线」(`rolling_mean(market_close, window_1)`) 经用户确认删除；删除走 `delete_indicator`，评价方案/快照配置/滚动来源三处引用检查均无占用 |
| 删除后复检 | 当前版本 58 个指标，**0 个**引用已停用变量；`custom_indicators.json` 中已停用变量出现次数为 0 |

过程中的一个教训已修正并记录：改动行情表后必须重建分析快照，且**必须带上
`workspace_data_dir`**。缺这个参数会走旧的内置核路径，`max_drawdown_3y` 会落成负号，
与配置指标口径相反；快照验收的抽样复算正确地拦住了它，重建后与前一版快照逐值一致。

### 8.6 尚未做的事
- `timing_research` 与 `research_series` 仍各自读 `fund_adj_factor_df.parquet` 算复权价，没有改读新列。
- 溢价率没有加进变量目录，原因见第 7 节。

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
| `source` | 只用数据源因子；数据源没有覆盖的标的不产出复权价格 |
| `source_then_pre_close`（默认） | 数据源覆盖的用数据源，其余用前收盘价推导 |
| `pre_close` | 全部由前收盘价推导，用于核验两条路径是否一致 |

默认优先取数据源因子，缺的用前收盘价推导。Tushare `fund_adj` 对本账号只覆盖 2 只
ETF（510300.SH、513500.SH），所以 `source` 单独作默认意味着一次成功的全量同步之后
1776 只标的仍然没有复权价格；而同一次变更已经把未复权 OHLC 全部停用，这些标的会
丢掉全部价格类指标。推导式不是降级：`close[t]/pre_close[t]-1` 与 `pct_chg/100` 在
1,577,335 行上无例外，两条路径同时可用的 6542 行最大相对差 2.27e-03。正式快照
（历史证据见[工程纪要](../verification/engineering.md)）本来就是按 `source_then_pre_close` 产出并激活的，默认值与之对齐。

`source` 保留给只认数据源因子的场合，`pre_close` 保留给交叉核验。两条路径同时可用时
相对偏离作为 stats 输出，不静默吞掉。

### 3.4 任务位置

`price_adjustment` 是本地派生任务（`api_slots` 为空、`network: False`），依赖
`candle` 与 `fund_adjustment`。两个都必须写进 `REQUIRES`：任务拿到的工作区由
`requires` 合成（`plan_dependencies` → `step.inputs` → `materialize`），排在某个
任务之后并不会让它的产出文件进入本任务目录。只声明 `candle` 时，`fund_adj_factor_df.parquet`
不会被投喂：`source` 口径下首次全量同步直接失败，默认口径下那 2 只有数据源因子的
标的会被静默改判成推导口径。
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

### 5.1 时序

当前内置时序只保留一份 revision 1 的定义：价格均线、布林带、KDJ使用 adjusted_close/high/low，日期轴锚定 adjusted_close；成交量均线锚定 volume，不因复权因子缺失一起不可计算。窗口参数与历史兼容见[指标参数契约](../indicators/parameters.md)。

旧v1–v4是历史迁移阶段记录，不在源码中并行保留；已经保存的用户定义与研究制品按各自不可变契约处理，不能由文档合并触发重写。

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

## 验收与剩余边界

缺因子、缺前收盘价、source模式缺因子文件均失败关闭；语义不一致的 adjusted_close/adjusted_nav 不允许相减。历史真实快照和测试见[工程纪要](../verification/engineering.md)。择时与研究序列原有因子读取路径、溢价率变量不属于该迁移的完成范围。

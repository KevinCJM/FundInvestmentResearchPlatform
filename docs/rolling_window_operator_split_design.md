# 滚动窗口算子拆分详细设计

## 1. 结论

按 AGENTS.md 的“最小独立计算语义”规则，现有 `rolling_mean / rolling_std / rolling_min / rolling_max` 应拆分。

它们当前同时承担两件互相独立的事：

1. 确定“截至当前时点最近 W 个观察值”的因果滚动窗口；
2. 对该窗口执行均值、标准差、最小值或最大值归约。

窗口选择本身具有独立输入、输出、时点语义和参数；均值、标准差等已经是独立统计原语。因此作者界面继续提供四套 `rolling_*` 会重复表达同一“窗口”语义，不符合算子最小颗粒度要求。

新建公式统一表达为：

```text
时间序列 -> 滚动窗口 -> 普通归约算子 -> 时间序列
```

示例：

```text
rolling_std(r, 20, 1)
        ↓
std(rolling_window(r, 20), 1)
```

```text
rolling_mean(close, 20)
        ↓
mean(rolling_window(close, 20))
```

```text
rolling_min(low, 9, 1)
        ↓
min_value(rolling_window(low, 9, 1))
```

`rolling_apply(values, window, operator)` 不采用。它只是把“算哪个统计量”藏进一个函数参数，仍然是组合黑盒，并且会引入一等函数/动态调度，与当前受限 DSL、固定签名 NJIT 和可审计 DAG 契约冲突。

## 2. 新算子职责

新增唯一作者原语：`rolling_window(values, window[, min_periods])`。

它只负责：

- 以当前时点为右端点；
- 使用当前及过去数据，不读取未来；
- 定义最近 `window` 个观察位置；
- 声明至少需要 `min_periods` 个有限观察值；
- 保留原输入的数值量纲和价格基准。

它不负责均值、标准差、收益率、极值等任何业务/统计计算。

输出是编译器内部的“滚动窗口集合”中间类型，不允许直接作为指标最终输出，也不允许参与普通加减乘除。第一期只允许以下普通归约原语消费它：

- `mean`
- `std`
- `variance`
- `min_value`
- `max_value`

这已经完整覆盖现有四个 `rolling_*` 的能力；后续如果对 `sum/product/quantile` 增加固定签名融合内核，可以在不改变 `rolling_window` 语义的前提下扩展消费者。

## 3. 类型与 DAG

增加逻辑中间类型 `window`，名义轴为 `time × window`，例如：

```text
window<time,window>[T,W]
```

它不是物化矩阵；类型只用于编译期约束和画布表达。

例如 20 日滚动标准差的真实作者 DAG 为：

```text
复权净值普通收益率
        |
        v
滚动窗口(window=20)
        |
        v
标准差(ddof=1)
        |
        v
时间序列输出
```

因此前端“完整计算说明”必须展示两个真实计算步骤，而不是把 `rolling_std` 解释成单一节点。

## 4. 执行层：不生成 T×W 矩阵

画布/DAG 颗粒度与执行内核颗粒度分离。

`rolling_window` 是逻辑中间节点，生产计算绝不分配 `T × W` 窗口矩阵。编译器识别：

```text
mean(rolling_window(...))
std(rolling_window(...), ddof)
min_value(rolling_window(...))
max_value(rolling_window(...))
variance(rolling_window(...), ddof)
```

直接融合为现有/对应的固定签名 NJIT 滚动内核。`rolling_window` 自身只转交原始一维数组引用和窗口参数，不产生数据副本。

要求保持：

- `python_fallback = 0`
- `request_time_compilation = 0`
- 启动/保存/显式准备阶段预热
- 输入数组仍为连续 `float64 ndarray`
- 不在请求期创建 T×W 高内存中间量

## 5. 数值契约

拆分必须与旧接口完全一致：

- 窗口右端包含当前观察值；
- `window` 为正整数；
- `1 <= min_periods <= window`；
- 缺失值不填零，只统计有限观察值；
- `std` 的 `ddof` 保持原定义；
- 当有限观察数不足 `min_periods` 或 `finite_count <= ddof` 时返回缺失；
- 极值滚动的并列/缺失规则保持现有 NJIT 内核行为；
- 输入语义量纲与价格基准传递到最终序列，variance 使用平方量纲。

## 6. 版本兼容

不能原地重新解释已经保存的 2.3 定义。

因此：

- 2.3 作为历史协议继续支持 `rolling_mean / rolling_std / rolling_min / rolling_max`；
- 新建定义使用新的 typed DSL / operator registry 版本；
- 新版本资源目录只展示 `rolling_window` + 普通归约，不再展示四个旧 `rolling_*`；
- 新版本仍接受旧 `rolling_*` 拼写作为兼容输入，但编译时展开为 `rolling_window + 普通归约`，不保留第二套新生产算法；
- 内置时序指标保留旧 revision 以复现历史结果，同时提供新 revision 使用拆分公式；repository 必须能按 `indicator_id + revision` 读取内置历史版本。

## 7. 参数化

可变参数能力迁移到真正拥有窗口语义的节点：

```text
rolling_window.window
rolling_window.min_periods
```

不再把窗口参数分别绑定到 `rolling_mean/std/min/max`。

`std.ddof` 仍属于标准差算子本身；它不是“滚动参数”。这样参数职责更清晰：

- 窗口长度/最少观察数 -> 滚动窗口；
- 自由度 -> 标准差/方差；
- 其他统计参数 -> 各自普通统计算子。

运行时参数变化后继续重新推导 lookback、缓存 key、结果 `parameter_hash` 和 Excel 证据。

## 8. 标量转滚动时序

当前“从标量指标生成滚动时序”不能再把：

```text
mean(x) -> rolling_mean(x,W)
std(x,1) -> rolling_std(x,W,1)
```

改写为旧包装。

新规则为：

```text
mean(x) -> mean(rolling_window(x,W))
std(x,1) -> std(rolling_window(x,W),1)
variance(x,1) -> variance(rolling_window(x,W),1)
min_value(x) -> min_value(rolling_window(x,W))
max_value(x) -> max_value(rolling_window(x,W))
```

其余算术、年化、无风险收益等保持来源公式不变。

## 9. Excel

Excel 不能物化窗口矩阵，也不能出现无法解释的私有函数。

公式编译器识别“归约算子消费 rolling_window”的结构，继续生成普通 Excel 公式：

- `AVERAGE(range)`
- `STDEV/STDEVP(range)`
- `MIN(range)`
- `MAX(range)`
- variance 对应 `VAR/VARP` 或等价公式

窗口范围、最少有效样本和缺失判断与 NJIT 完全一致。

## 10. 因果性

`rolling_window` 本身是因果算子：每个 t 只依赖 `[t-W+1, t]`。

普通 `mean/std/min/max` 的时间语义由输入 DAG 传播；这样“是否实时可用”不再依赖 `rolling_std` 这个业务组合名，而是来自：

```text
输入时点语义 + rolling_window 因果性 + 普通归约
```

因果审计基线需增加 `rolling_window=causal`，旧 2.3 `rolling_*` 条目仅用于历史兼容验证。

## 11. 验收

至少覆盖：

1. 当前作者目录只显示一个“滚动窗口”，不显示四个旧滚动包装；
2. 公式、画布、完整计算说明真实出现“滚动窗口 -> 标准差/均值/极值”两个节点；
3. 新旧 MA、BOLL、KDJ、滚动 Sharpe 数值逐点一致；
4. NaN、窗口不足、`min_periods`、`ddof` 边界一致；
5. `rolling_window` 不能直接发布，也不能接入普通加减乘除；
6. 编译后没有 T×W 数组，不发生 Python fallback/请求期编译；
7. 可变参数识别从旧滚动包装迁移到 `rolling_window`；
8. 2.3 历史公式与内置 revision 仍可按原版本运行；
9. Excel、LaTeX、DAG 与执行结果一致；
10. causality audit 无 missing/changed/stale 基线。

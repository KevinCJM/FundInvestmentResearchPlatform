# 标量指标滚动转换为时序指标——详细设计

> 状态：已实现
> 版本：1.0
> 日期：2026-09-05

## 1. 问题

此前指标中心存在两种能力，但没有打通：

- 标量指标：在一个完整计算区间上返回一个数值；
- 时序指标：一次执行返回一个或多个具名时间序列通道。

旧的 `include_series` 只是按多个历史截止日重复执行标量计算，不是可保存、可预热、可快照、可导出 Excel 的正式时序指标，因此不能作为统一的滚动指标方案。

## 2. 目标

允许用户选择一个已保存的单产品 typed DSL 标量指标和固定观察窗口，生成一个普通的 `time_series` 指标草稿。生成结果必须：

1. 锁定来源指标 ID、revision 和定义哈希；
2. 把窗口固定在指标公式与版本中，不接受运行时覆盖；
3. 进入现有多根 Typed DAG、固定签名 NJIT、缓存和预热链路；
4. 可继续使用指标中心校验、预览、保存、快照和 Excel 导出；
5. Excel 只使用原生函数和直接单元格引用；
6. 无法保持原指标语义时明确拒绝，禁止 Python 循环或静默近似。

## 3. 转换模型

首版对标量公式的受控 AST 做编译期转换：

| 标量归约 | 固定 N 期滚动表达式 |
|---|---|
| `mean(x)` | `rolling_mean(x, N)` |
| `std(x, d)` | `rolling_std(x, N, d)` |
| `variance(x, d)` | `power(rolling_std(x, N, d), 2)` |
| `min_value(x)` | `rolling_min(x, N)` |
| `max_value(x)` | `rolling_max(x, N)` |

归约外层的加、减、乘、除、乘方、平方根、条件和限幅逻辑保持不变。未登记的归约算子失败关闭，并返回 `ROLLING_SCALAR_OPERATOR_UNSUPPORTED`。

这不是请求期逐日调用标量服务。转换后得到普通 typed DSL 时序公式，只编译和执行一次固定签名 NJIT 计划。

## 4. 来源锁定

生成定义保存：

```json
{
  "rolling_source": {
    "kind": "rolling_scalar",
    "transform_version": "1.0.0",
    "indicator_id": "...",
    "indicator_revision": 1,
    "indicator_name": "年化夏普比率",
    "definition_hash": "sha256...",
    "source_dsl_version": "2.1.0",
    "window_observations": 5,
    "minimum_observations": 5,
    "detached": false
  }
}
```

创建、更新、预热时重新读取来源 revision，并核对定义哈希与生成公式。来源不一致、公式被私自改写或计算参数漂移时拒绝执行。仍被滚动指标引用的自定义标量指标不能删除。

## 5. 日期和收益率

`returns` 与 `log_returns` 由相邻复权净值生成，并与复权净值日期轴对齐：

\[
r_t = \frac{NAV_t}{NAV_{t-1}}-1
\]

第一条记录保持为空。N 个收益率需要 N+1 个净值点，所以有限回看会额外读取一个复权净值观察值。缺失值不补零、不前向填充。

## 6. 固定标量参数

来源指标中的稳定日频口径在生成时固化：

- `observation_count = N`；
- `periods_per_year = 252`；
- 年化无风险利率沿用来源 revision；
- 单观察期无风险收益率按复利换算。

随窗口起止日期变化的动态标量暂不自动转换，避免同一公式在不同日期使用错误常量。

## 7. 五日滚动年化夏普比率

来源：`builtin-annualized-sharpe-v2@1`。

\[
SR_t^{(5)} =
\frac{\mu_{t,5}(r)-r_{f,d}}{s_{t,5}(r)}\sqrt{252}
\]

其中：

- `r` 为复权净值普通收益率；
- 均值窗口为 5；
- 标准差窗口为 5，`ddof=1`；
- 年化无风险利率沿用来源指标第 1 版；
- 分母为零或不足 5 个有限收益率时结果为空。

内置指标 ID：

```text
builtin-rolling-5d-annualized-sharpe-series
```

## 8. API 与界面

新增：

```text
POST /api/custom-indicators/derive-rolling-series
```

请求包含来源指标、可选锁定 revision、固定滚动观察数、可选名称和说明。接口返回尚未保存的时序指标草稿及完整校验结果；保存仍使用现有指标创建接口。

指标中心在选中可转换的标量指标时提供“生成滚动时序指标”入口。窗口值属于定义级固定常量，生成后可继续查看 LaTeX、DAG、预览结果和 Excel 计算逻辑。

主公式采用紧凑算子符号：滚动均值为 `\mu_{t,w}`，总体/样本滚动标准差分别为 `\sigma_{t,w}` 与 `s_{t,w}`，递归平滑为 `\mathcal S`。`min_periods`、缺失值和初始化规则放在计算说明中，不在主公式展开成分段函数。

## 9. Excel

转换后的指标复用时序 Excel 编译器。例如五日滚动夏普使用：

```excel
=AVERAGE(...)
=STDEV(...)
=SQRT(年化因子单元格)
```

工作簿不包含 DSL 函数、VBA/UDF、隐藏命名公式或请求期黑盒结果。

## 10. 验收重点

- 来源 revision/hash 锁定；
- 5 个收益率对应 6 个净值边界点；
- NJIT 正式链路，`python_fallback=0`；
- 与 pandas 受控参考值逐日期一致；
- 窗口固定且运行请求不可覆盖；
- Excel 使用原生函数；
- 不支持的归约明确失败；
- 原有标量指标与四个时序内置指标无回归。

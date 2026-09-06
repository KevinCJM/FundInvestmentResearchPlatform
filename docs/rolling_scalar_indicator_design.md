# 标量指标滚动提升为时序指标——设计与实现

> 状态：已实现 1.0
> 日期：2026-09-05

## 1. 当前问题

原指标中心有两套能力：

- 标量指标：对一个完整计算窗口得到一个数值；
- 时序指标：公式本身直接返回时间序列。

此前不能把一个已经锁定版本的标量指标，在每个时点对最近固定观察窗口重复计算并生成时间序列。

## 2. 目标语义

给定标量指标：

\[
y=f(X)
\]

固定窗口为 \(w\) 个观察值时，滚动提升结果为：

\[
y_t=f\left(X_{t-w+1:t}\right)
\]

首版窗口按“观察值数量”定义，不按自然日定义。窗口、最少有效观察数和源指标版本全部固化在时序指标定义中，运行请求不能覆盖。

## 3. 定义契约

生成后的时序指标保存：

```json
{
  "result_kind": "time_series",
  "rolling_source": {
    "kind": "rolling_scalar",
    "indicator_id": "源标量指标 ID",
    "indicator_revision": 1,
    "definition_hash": "锁定定义哈希",
    "indicator_name": "年化夏普比率",
    "window_observations": 5,
    "minimum_observations": 5,
    "transform_version": "1.0.0",
    "detached": false
  }
}
```

源版本哈希或展开公式不一致时拒绝保存，不能静默改用最新版本。

## 4. 编译方式

不在请求期逐日调用 Python 标量函数。系统在定义阶段把标量 DAG 展开成等价时序 DAG：

| 标量归约 | 滚动时序表达 |
|---|---|
| `mean(x)` | `rolling_mean(x, w)` |
| `std(x)` | `rolling_std(x, w, 1)` |
| `std(x, d)` | `rolling_std(x, w, d)` |
| `variance(x, d)` | `rolling_std(x, w, d) ** 2` |
| `min_value(x)` | `rolling_min(x, w)` |
| `max_value(x)` | `rolling_max(x, w)` |
| `sum(x)` | 完整窗口下为 `rolling_mean(x, w) * w` |

普通四则运算、平方根、对数、限幅和条件运算继续使用原 typed 算子并按标量广播规则运行。

转换后的普通 TypedSeriesBundlePlan 继续走：

```text
固定签名 NJIT 预热 → 请求期只执行已编译计划 → python_fallback=0
```

没有等价透明滚动语义的标量算子会明确返回 `ROLLING_SCALAR_OPERATOR_UNSUPPORTED`，不使用黑盒回退。

## 5. 运行时标量

首版支持按观察频率广播：

- `annual_risk_free_rate_decimal`
- `risk_free_rate_per_observation`
- `periods_per_year`

`observation_count` 在编译期替换为固定窗口数。

以下变量描述完整窗口的日历跨度，首版拒绝自动提升：

- `window_elapsed_days`
- `risk_free_return_window`

## 6. 5 日滚动年化夏普比率

源标量指标：`builtin-annualized-sharpe-v2@1`。

标量公式：

\[
SR=\frac{\bar r-r_f}{s_r}\sqrt{P_{year}}
\]

滚动公式：

\[
SR_t^{(5)}=
\frac{\operatorname{Mean}(r_{t-4:t})-r_f}
{\operatorname{Std}_{ddof=1}(r_{t-4:t})}
\sqrt{P_{year}}
\]

DSL 展开为：

```text
(rolling_mean(returns, 5) - risk_free_rate_per_observation)
/ rolling_std(returns, 5, 1)
* sqrt(periods_per_year)
```

前 4 个收益观察值为空；从第 5 个有效收益观察值开始输出。若样本标准差为零，沿用安全除法和非有限值诊断，不伪造为零。

## 7. 数学公式展示

公式展示与计算实现分层。DSL 和 NJIT 仍保留完整滚动语义；指标中心的主公式只显示紧凑数学符号：

| 计算算子 | 主公式符号 |
|---|---|
| 滚动均值 | `\mu_{t,w}(x)` |
| 总体滚动标准差 | `\sigma_{t,w}(x)` |
| 样本滚动标准差 | `s_{t,w}(x)` |
| 滚动最小值 / 最大值 | `\min_{i\in\mathcal W_{t,w}}x_i` / `\max_{i\in\mathcal W_{t,w}}x_i` |
| 递归平滑 | `\mathcal S_{p,s_0}(x)_t` |
| 带默认值安全除法 | `a\oslash_d b` |

`min_periods`、缺失值规则、`ddof` 和初始化方式继续在计算说明及 DAG 中展示，不在主公式中展开成分段函数。

五日滚动年化夏普的主公式为：

\[
SR_t^{(5)}=
\frac{\mu_{t,5}(r)-r_f}{s_{t,5}(r)}\sqrt{P_{year}}
\]

## 8. Excel

滚动提升后的指标不需要专用 Excel 黑盒。展开后的 DAG 继续使用原生 Excel 函数：

```excel
=AVERAGE(最近5个收益单元格)
=STDEV(最近5个收益单元格)
=(滚动均值-单观察期无风险收益)/滚动标准差*SQRT(年化因子)
```

工作簿不包含 DSL 函数、VBA、UDF 或隐藏命名公式。

## 9. 接口

```http
POST /api/custom-indicators/derive-rolling-series
```

请求：

```json
{
  "indicator_id": "builtin-annualized-sharpe-v2",
  "indicator_revision": 1,
  "window_observations": 5,
  "name": "5 日滚动年化夏普比率"
}
```

返回一个已经校验、带锁定来源信息的普通时序指标草稿。草稿之后仍通过原指标创建接口保存。`/api/custom-indicators/rolling-scalar-draft` 仅作为旧客户端兼容入口，内部委托给同一转换逻辑。

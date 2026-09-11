# 自定义指标 Excel 计算复现——详细设计与实现说明

> 状态：已开发并完成核心回归
> 版本：1.3
> 日期：2026-09-04

## 1. 需求定义

在指标中心的“校验与预览”区域，于“预览指标”按钮下方增加：

```text
下载 Excel 计算逻辑
```

用户选择一个指标、一个或多个产品、计算周期和可选历史截止日后，下载的 Excel 必须包含：

1. 本次指标实际传入计算引擎的全部直接变量；
2. 时间序列变量对应的实际日期；
3. 按 Typed DAG 顺序展开的全部中间计算；
4. 由 Excel 函数实现的算子逻辑；
5. Excel 公式得出的最终结果；
6. 平台固定签名 NJIT 的正式结果；
7. 两个结果的差异与一致性检查。

这里的“原始入参数据”是指：

> **本次运行真正传给指标 NJIT 计划的变量值。**

不是整张原始 Parquet，也不是与公式无关的行情字段。

## 2. 目标示例

指标：

```text
平均收益率 = mean(returns)
```

产品：上证 50 ETF
周期：近 1 年

Excel 中生成直接入参：

| 日期 | 复权净值普通收益率 `returns` |
|---|---:|
| 2025-09-05 | 0.003241 |
| 2025-09-08 | -0.001827 |
| ... | ... |

并生成 Excel 公式：

```excel
=AVERAGE(B18:B278)
```

公式直接引用产品 Sheet 中可见的入参单元格，不通过 `P01_RESULT`、`P01_VAR_returns` 等命名区域间接跳转。

工作簿同时保留：

```text
Excel 公式结果
平台 NJIT 结果
绝对差异
一致 / 不一致
```

用户修改收益率入参后，Excel 会重新计算平均收益率以及后续全部依赖步骤。

## 3. 功能边界

### 3.1 当前支持

- 指标中心当前的 `single_product` 单产品域；
- 当前编辑草稿、内置 typed 指标、工作区 typed 指标；
- ETF 与场外公募基金；
- 每次一个指标；
- 每次 1 至 10 个产品；
- 当前全部运行周期；
- 可选 `as_of` 历史截止日；
- 当前指标中心公开的全部单产品变量；
- 当前指标中心公开的全部 70 个单产品算子；
- 4 个历史 typed 兼容算子：`active_returns`、`annualized_return`、`cumulative_return`、`total_return`。

### 3.2 当前不属于指标中心单产品域

组合域的以下数据结构不在本接口范围内：

- `vector<asset>`；
- `matrix<time, asset>`；
- `matrix<asset, asset>`；
- 组合权重路径；
- 组合运行快照。

因此 `matmul`、`matvec`、`solve`、`trace`、资产轴归约等组合/矩阵算子不从当前 IndicatorStudio 暴露，也不属于本次单产品 Excel 导出契约。

如果未来这些结构进入指标中心，必须新增二维 Excel 区域编译器；现阶段收到矩阵节点时明确失败，不伪造结果。

## 4. 当前支持的直接变量

Excel 导出覆盖指标中心当前全部 23 个单产品变量。

### 4.1 时间序列

```text
returns
log_returns
adjusted_nav
market_open
market_high
market_low
market_close
previous_close
price_change
price_return
volume
turnover_amount
unit_nav
accumulated_nav
accumulated_dividend
net_asset
total_net_asset
```

### 4.2 运行时标量

```text
observation_count
window_elapsed_days
risk_free_return_window
annual_risk_free_rate_decimal
risk_free_rate_per_observation
periods_per_year
```

### 4.3 日期对齐规则

- `adjusted_nav` 保留计算收益所需的边界净值，因此通常比 `returns` 多一行；
- `returns` 与 `log_returns` 的日期从第二个净值点开始；
- 行情、成交及净值字段使用它们真正传入运行时的日期和值；
- 标量使用“序号 + 数值”展示；
- Excel 不重新推导或猜测变量，只写入运行时上下文中的真实输入。

## 5. 算子覆盖

Excel 公式注册表版本：

```text
EXCEL_FORMULA_REGISTRY_VERSION = 1.3.0
```

### 5.1 基础及逐元素算子

```text
absolute, add, clip, divide, exp, log,
maximum, minimum, multiply, negate, power,
reciprocal, sign, sqrt, subtract
```

典型映射：

```excel
=ABS(x)
=x+y
=IF(ABS(y)<1E-12,NA(),x/y)
=IF(x<=0,NA(),LN(x))
=IF(x<0,NA(),SQRT(x))
=MIN(MAX(x,lower),upper)
```

### 5.2 比较与布尔算子

```text
equal, not_equal,
greater_than, greater_equal,
less_than, less_equal,
logical_and, logical_or, logical_not
```

每一个时间点生成独立的 Excel 布尔公式。

### 5.3 条件与掩码归约

```text
where
count_true
max_consecutive_true
max_where
mean_where
median_where
min_where
quantile_where
std_where
sum_where
variance_where
```

条件归约使用可见辅助列：

```excel
=IF(mask_cell,value_cell,NA())
```

然后通过 `AGGREGATE(...,6,...)` 忽略未选中行产生的 `#N/A`，避免把未选中值当作 0。

### 5.4 序列与路径算子

```text
lag
difference
cumulative_sum
cumulative_product
cumulative_max
cumulative_min
drawdown_series
new_high_mask
```

例如回撤：

```excel
=current_level/MAX(first_level:current_level)-1
```

严格创新高：

```excel
=current_level>MAX(first_level:previous_level)
```

首个观察值按当前运行时语义为 `TRUE`。

### 5.5 时序滚动与递归算子

时序指标工作簿不把 DSL 函数名写入公式，也不通过命名区域隐藏计算。固定窗口会直接展开成可见的 A1 区间：

```excel
=IF(COUNT(B16:B35)<20,"",AVERAGE(B16:B35))
=IF(COUNT(B16:B35)<20,"",STDEVP(B16:B35))
=IF(COUNT(B16:B24)<1,"",MAX(B16:B24))
=IF(COUNT(B16:B24)<1,"",MIN(B16:B24))
```

递归平滑使用可见的“递归状态”辅助列：当前值缺失时状态沿用上一有效值，但结果单元格保持为空；后续有效值再从该状态继续计算。工作簿不使用 `LOOKUP` 搜索历史结果，也不生成 `S01_NODE_*`、`S01_CHANNEL_*` 等计算命名区域。

每个时序计算区同时提供：

```text
Excel 公式结果
递归状态（仅递归算子）
原生 Excel 公式（可复制）
```

### 5.6 普通归约

```text
first
last
length
max_value
mean
median
min_value
product
sum
std
variance
argmax
argmin
```

映射示例：

```excel
=AVERAGE(range)
=MEDIAN(range)
=PRODUCT(range)
=SUM(range)
=INDEX(range,1)
=INDEX(range,ROWS(range))
=MATCH(MAX(range),range,0)-1
```

`argmax` / `argmin` 减 1，是为了与平台从 0 开始的索引语义一致。

任意合法 `ddof` 的方差：

```excel
=DEVSQ(range)/(COUNT(range)-ddof)
```

标准差为上述方差开平方，因此不限于只支持 `VAR.P` 和 `VAR.S`。

### 5.7 统计、回归与分布

```text
correlation
covariance
skewness
excess_kurtosis
mean_absolute_deviation
root_mean_square
quantile
linear_slope
linear_intercept
linear_r_squared
regression_standard_error
normal_pdf
normal_ppf
```

主要映射：

```excel
=CORREL(x,y)
=COVARIANCE.S(x,y)
=SKEW(range)
=KURT(range)
=AVEDEV(range)
=SQRT(SUMSQ(range)/COUNT(range))
=PERCENTILE.INC(range,p)
=SLOPE(y,x)
=INTERCEPT(y,x)
=RSQ(y,x)
=STEYX(y,x)
=NORM.S.DIST(x,FALSE)
=NORM.S.INV(p)
```

单参数线性回归在 Excel 中生成可见的 `0 ... N-1` 自变量辅助列，与后端运行时一致。

### 5.8 一维线性代数

当前单产品域可到达：

```text
dot
```

映射为：

```excel
=SUMPRODUCT(x,y)
```

## 6. 生成链路

```text
IndicatorStudio 当前草稿 / 已保存指标
        ↓
validate：Typed AST / DAG + compile_token
        ↓
POST /api/custom-indicators/export-excel
        ↓
锁定指标定义和固定签名 NJIT 计划
        ↓
锁定当前 market_data_generation
        ↓
按公式依赖批量加载真实产品数据
        ↓
复用 select_variable_window_fast 选择准确窗口
        ↓
复用现有 NJIT runtime 计算平台结果
        ↓
提取 compiled_plan.context_names 对应的直接入参
        ↓
Typed DAG → Excel 函数与显式 A1 单元格引用
        ↓
生成 .xlsx 并下载
```

导出完成前再次校验 `market_data_generation`。如果导出过程中数据刷新并切换版本，返回：

```text
EXCEL_EXPORT_DATA_CHANGED
```

防止一个工作簿混入两个数据版本。

## 7. API

### 7.1 路由

```http
POST /api/custom-indicators/export-excel
```

### 7.2 请求

```json
{
  "indicator_ids": [],
  "inline_definition": {
    "name": "平均收益率",
    "expression": "mean(returns)",
    "context_kind": "single_product",
    "output_contract": "scalar",
    "dsl_version": "2.2.0"
  },
  "compile_token": "validate 返回的 token",
  "targets": [
    {"kind": "etf", "product_id": "510050.SH"}
  ],
  "period": "1Y",
  "as_of": null
}
```

规则：

- `indicator_ids` 与 `inline_definition` 二选一；
- 一个工作簿只允许一个指标；
- 1 至 10 个产品；
- inline 草稿必须先完成 `/validate`，并携带匹配的 `compile_token`；
- 只支持 typed `single_product` 指标。

### 7.3 响应

```http
200 OK
Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
Content-Disposition: attachment; filename*=UTF-8''...
Cache-Control: no-store
```

临时文件在响应发送完成后通过 `BackgroundTask` 删除。

## 8. 工作簿结构

### 8.1 `01_结果汇总`

每个产品一行：

- 产品类型、代码、名称；
- 计算状态；
- 实际窗口；
- Excel 公式结果；
- 平台 NJIT 结果；
- 绝对差异；
- 一致性；
- 警告和不可计算原因。

### 8.2 `Pxx_<产品代码>`

每个产品一个 Sheet，结构为：

```text
产品与指标信息
Excel 最终结果 / 平台 NJIT 结果 / 差异

直接入参节点 1
  日期或序号 | 直接入参值

直接入参节点 2
  日期或序号 | 直接入参值

常量节点

计算步骤节点 1
  日期或序号 | Excel 公式结果 | 必要辅助列

计算步骤节点 2
  ...

最终根节点
  Excel 公式 | 计算结果
```

所有可见计算公式都直接使用 Sheet 内的 A1 单元格或区域地址，例如：

```excel
=AVERAGE(B18:B278)
=PRODUCT(B18:B278)
=ABS(B11-B12)
```

不使用 `P01_RESULT`、`P01_VAR_returns`、`P01_NODE_0001` 等自定义命名区域作为计算引用。复杂指标可以引用前面已经公开展示的中间计算单元格，但每一步仍是可见的 Excel 函数或运算符，不形成隐藏跳转。

结果汇总 Sheet 使用带产品 Sheet 名的显式公式，例如：

```excel
=AVERAGE('P01_510050.SH'!B18:B278)
```

## 9. 正式结果与 Excel 结果

权威关系：

```text
固定签名 NJIT = 平台正式结果
Excel 公式      = 可阅读、可编辑、可审计的复现结果
```

一致性容差：

```text
absolute_difference = ABS(Excel - NJIT)
一致 ⇔ absolute_difference <= MAX(1E-12, ABS(NJIT) × 1E-10)
```

Excel 打开后设置为自动重算。生成器会把 NJIT 结果作为根节点的缓存显示值，但单元格本身仍然保存真实 Excel 公式，不是硬编码结果。

## 10. 缺失与异常语义

必须保持现有指标运行契约：

- 不使用模拟数据；
- 不 forward-fill；
- 不把缺失值补 0；
- 不把不可计算指标写成 0；
- 除数绝对值小于 `1E-12` 时返回 `#N/A`；
- 非法 `log`、`sqrt`、分位数参数等返回 `#N/A`；
- 条件集合为空时返回 `#N/A`；
- 数据或窗口不可用时仍可生成说明 Sheet，但不伪造 Excel 计算结果。

## 11. 性能与安全

- XlsxWriter 使用 `constant_memory=True`；
- 工作簿只导出公式真正依赖的直接入参；
- 默认最多生成 2,000,000 个公式单元格；
- 超限返回 `EXCEL_EXPORT_TOO_LARGE`；
- 关闭字符串自动转公式和自动转 URL；
- 不包含 VBA、宏、外部链接或外部数据连接；
- 产品名称和指标名称只作为普通文本写入；
- Sheet 名和文件名进行非法字符清理。

## 12. 文件改动

新增：

```text
backend/custom_indicators/excel_formula.py
backend/custom_indicators/excel_export.py
backend/tests/test_custom_indicator_excel_export.py
docs/custom_indicator_excel_export_design.md
```

修改：

```text
backend/custom_indicators/service.py
backend/services/custom_indicator_routes.py
backend/requirements.txt
frontend/src/services/customIndicators.ts
frontend/src/pages/IndicatorStudio.tsx
frontend/src/pages/IndicatorStudio.test.tsx
```

## 13. 测试与验收

自动测试覆盖：

1. 当前 70 个公开单产品算子全部存在 Excel 公式实现；
2. 每个算子使用一个合法 Typed DSL 表达式完成公式生成；
3. 当前 23 个单产品变量全部能作为 Excel 直接入参；
4. `mean(returns)` 的产品结果单元格直接包含 `AVERAGE(B18:B278)`；
5. 工作簿不包含 `P01_RESULT`、`P01_VAR_returns` 等自定义计算命名区域；
6. 工作簿包含 Excel 结果、NJIT 结果和一致性字段；
7. API 返回有效 `.xlsx`；
8. 前端按钮位于“预览指标”下方，并正确下载 Blob；
9. 原有自定义指标、Typed DSL、NJIT 和前端指标中心回归继续通过。

人工验收案例：

```text
指标：平均收益率
公式：mean(returns)
产品：任一具有有效数据的 ETF
周期：1Y
```

验收时打开产品 Sheet，应看到：

```text
直接入参 · 复权净值普通收益率（returns）
日期 | 直接入参值
...

最终计算步骤 · mean
Excel 公式：=AVERAGE(B18:B278)
```

修改任意一行收益率并重新计算工作簿，Excel 最终结果必须随之变化。

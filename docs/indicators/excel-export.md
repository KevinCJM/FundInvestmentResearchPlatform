# 指标 Excel 复现契约

导出用于检查实际输入、逐节点公式与正式 NJIT 结果。主入口见[指标中心](README.md)，滚动作用域另见[滚动契约](rolling-intervals.md)。

不维护重复的完整算子清单：以当前算子注册表、Excel renderer 能力与回归测试为准。新增算子只有具备等价 Excel 翻译才可导出；不能生成同名伪函数冒充支持。
## 功能边界

### 当前支持

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

### 当前不属于指标中心单产品域

组合域的以下数据结构不在本接口范围内：

- `vector<asset>`；
- `matrix<time, asset>`；
- `matrix<asset, asset>`；
- 组合权重路径；
- 组合运行快照。

因此 `matmul`、`matvec`、`solve`、`trace`、资产轴归约等组合/矩阵算子不从当前 IndicatorStudio 暴露，也不属于本次单产品 Excel 导出契约。

如果未来这些结构进入指标中心，必须新增二维 Excel 区域编译器；现阶段收到矩阵节点时明确失败，不伪造结果。
## 生成链路

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

## API

### 路由

```http
POST /api/custom-indicators/export-excel
```

### 请求

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

### 响应

```http
200 OK
Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
Content-Disposition: attachment; filename*=UTF-8''...
Cache-Control: no-store
```

临时文件在响应发送完成后通过 `BackgroundTask` 删除。

## 工作簿结构

### `01_结果汇总`

每个产品一行：

- 产品类型、代码、名称；
- 计算状态；
- 实际窗口；
- Excel 公式结果；
- 平台 NJIT 结果；
- 绝对差异；
- 一致性；
- 警告和不可计算原因。

### `Pxx_<产品代码>`

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

## 正式结果与 Excel 结果

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

## 缺失与异常语义

必须保持现有指标运行契约：

- 不使用模拟数据；
- 不 forward-fill；
- 不把缺失值补 0；
- 不把不可计算指标写成 0；
- 除数绝对值小于 `1E-12` 时返回 `#N/A`；
- 非法 `log`、`sqrt`、分位数参数等返回 `#N/A`；
- 条件集合为空时返回 `#N/A`；
- 数据或窗口不可用时仍可生成说明 Sheet，但不伪造 Excel 计算结果。

## 性能与安全

- XlsxWriter 使用 `constant_memory=True`；
- 工作簿只导出公式真正依赖的直接入参；
- 默认最多生成 2,000,000 个公式单元格；
- 超限返回 `EXCEL_EXPORT_TOO_LARGE`；
- 关闭字符串自动转公式和自动转 URL；
- 不包含 VBA、宏、外部链接或外部数据连接；
- 产品名称和指标名称只作为普通文本写入；
- Sheet 名和文件名进行非法字符清理。

## 验证与限制

锁定产品、版本、完整参数、研究日及真实窗口；保留 NaN/Inf、空样本、日期与状态语义。Excel 数值差异按规定容差展示，不能覆盖正式结果。日期／duration 独立指标与时序多通道沿当前公开契约，不恢复退役的多标量子结果类型。

逐节点／滚动展开受单元格预算限制。流式写入不得先写零后回填已刷新行；不支持的公式明确拒绝。历史测试和浏览器范围见[工程纪要](../verification/engineering.md)。

# 指标可变参数：契约派生判定与标量指标支持设计

## 1. 目标

1. **删掉可变参数的硬编码白名单**，判定规则改为从算子类型契约派生。放开新参数不再需要改表。
2. **标量指标结果同样支持可变参数**，与时序指标共用一套定义、校验、执行与缓存契约。

不变的边界（沿用 `docs/indicator_runtime_parameters_design.md`）：参数是一次计算内固定的标量；不能改算子、数据字段、复权口径、日期轴、输出结构；所有数值仍在预热的固定签名 NJIT 计划里跑，`python_fallback=0`。

---

## 2. 改动前：同一份知识有 8 份副本

同一个「`(算子, 入参位)` 是什么性质」的事实，改动前散落在 8 处，彼此不一致：

| # | 位置 | 内容 | 规模 |
|---|---|---|---|
| 1 | `backend/custom_indicators/series_parameters.py:27` `PARAMETER_CAPABILITIES` | 允许参数化的位置 + 类型/范围/标签 | 17 对 |
| 2 | `backend/custom_indicators/typed_service.py:300` `_FIXED_CONSTANT_OPERATOR_PARAMETERS` | 必须是常量的位置 | **死代码，全仓零读取** |
| 3 | `backend/custom_indicators/typed_service.py:356` `_FIXED_CONSTANT_PARAMETER_POLICIES` | 常量策略 + 默认值/范围，供 `/meta` 下发前端 | 24 对 |
| 4 | `backend/custom_indicators/typed_service.py:317` `PARAMETER_LABELS` | 入参中文名 | **`default` / `initial` 各重复一次，前者被静默覆盖** |
| 5 | `backend/custom_indicators/series_service.py:93` `_SERIES_FIXED_ARGUMENTS` | 必须是常量的位置（与 #2 内容完全相同） | 25 对 |
| 6 | `frontend/src/components/indicator-graph/indicatorGraphPresentation.ts:13` | `window/ddof/min_periods` 中文名 | 3 条 |
| 7 | `frontend/src/pages/IndicatorStudio.tsx:808-813,826-827` | 算子入参名序列 + 中文名 | 6 算子 + 8 名字 |
| 8 | `frontend/src/pages/IndicatorStudio.tsx:237` | `['ddof','periods','probability']` 不接受指标输入 | 3 条 |

集合关系（已实测）：

```
CAP(17) ⊂ POL(24) ⊂ FIX(25)
FIX − POL = { power.exponent }
```

即「能参数化」是「必须常量」的真子集，中间那 7 个位置（`rolling_std.ddof`、`std.ddof`、`variance.ddof`、
`quantile.probability`、`quantile_where.probability`、`recursive_smooth.initial`、`divide_or_default.default`）
被强制常量、却不允许用户开放——这正是要消除的产品判断。

另有一组命名不一致造成的漏网：`variance.ddof` 在 FIX 里、`var.ddof` 不在；`quantile.probability` 在、
`masked_quantile.probability` 不在；`value_at.position` 完全不受约束。手工维护的表必然出现这种缺口。

---

## 3. 需求一：判定规则改为契约派生

### 3.1 判定规则（已实现）

一个入参位是**配置输入**（configuration input），当且仅当它的签名契约带**配置语义标签**。
配置输入必须是定义级常量，也正是唯一可以被开放为运行时参数的位置。

rank-0 标量契约文法（`backend/cal_indicators/parameter_policy.py`）：

```
scalar                    普通数值位；可以是任意标量表达式
scalar<dimensionless>     普通数值位，且必须无量纲
scalar<count>             配置：整数 [1, 20000]
scalar<count:LOW..HIGH>   配置：整数，两端可省略表示无界
scalar<probability>       配置：实数 (0, 1)
scalar<const>             配置：任意有限数值
```

再加一条语法条件决定候选：

- **C2（本处是有限数值常量）**：该位置的 AST 节点是有限数值字面量、安全四则/幂运算常量，或已绑定的参数名。
  识别和编译共用 `parameter_policy.constant_number()`，不执行任意代码。

**配置输入 ∧ C2 ⇒ 可变参数候选。** 别名（`var` / `sequence_std` / `masked_quantile`）按
`spec.operator_id` 归一，不再漏判。

### 3.2 实测结果

派生结果与原 `_FIXED_CONSTANT_PARAMETER_POLICIES` 的 24 个位置**逐字段一致**
（`constant_kind` / `minimum` / `maximum`），由 `test_configuration_inputs_have_exactly_one_source` 锁定。

原设计草案里「纯 rank-0 标量即配置输入」的推导会误伤三类位置，实测后放弃：

| 位置 | 为什么不是配置输入 |
|---|---|
| `value_at.position` | 内置回撤指标写的是 `value_at(observation_dates, interval_start(...))`，本来就是表达式 |
| `annualized_return.periods_per_year` | 内置年化收益传的是标量变量 `periods_per_year`，不是常量 |
| `rolling_apply.calculation` / `require_positive.values` / `days_between.*` | 契约上是标量，真实公式里永远是子表达式 |

标签法把这三类自然排除：它们的契约保持裸 `scalar`。

**`power.exponent`（原 D1）**：原方案建议把签名收紧为 `scalar` 从而纳入配置输入。
实测后改为 `scalar<dimensionless>`，理由是 `_power_type` 本来就只接受无量纲标量指数（签名撒的是这个谎），
而把它变成配置输入会**破坏**画布里合法的 `power(product(returns + 1), periods_per_year / observation_count)`
这类年化写法——那是标量表达式，不是定义级常量。因此：签名说实话，但 `power.exponent`
不再被时序链路强制为常量（原 `_SERIES_FIXED_ARGUMENTS` 与 `_FIXED_CONSTANT_PARAMETER_POLICIES`
在这一项上本来就互相矛盾，现在两边一致）。

### 3.3 类型、范围、步长从哪来（已实现）

全部来自契约标签：`scalar<count>` → 整数 + 步长 1；`scalar<probability>` → 实数 (0,1)；
`scalar<count:1..5000>` → `rolling_apply` 的内核上限由 `rolling_scope.MAX_WINDOW` 直接拼进签名，
上限与内核守卫不可能再漂移。运行时参数需要有限上下界，无界契约由 `parameter_policy()`
回落到注册表级别的 `COUNT_LIMIT` / `NUMBER_LIMIT`。

派生不到的只剩两类，`parameter_policy.py` 里各声明一次，**都不是门禁**：

- `ARGUMENT_DEFAULTS`：画布里的预填数值（窗口 20、ddof 1、概率 0.5 …）。改它只改预填数字。
- `ARGUMENT_LABELS` / `OPERATOR_ARGUMENT_LABELS`：中文名。改它只改文案。

### 3.4 单一真源落点（已实现）

`backend/cal_indicators/parameter_policy.py`：

```python
contract_policy(contract)                 # 解析一条契约串
configuration_arguments(spec, arity)      # 某个 arity 下的全部配置输入
parameter_policy(spec, arity, name)       # 运行时参数 schema（有限上下界 + 步长）
argument_label(operator_id, name)         # /meta 下发的中文名
```

四个消费方各读一次，互不复制：

| 消费方 | 用途 |
|---|---|
| `typed_service.parameters_for_signature` | `/meta` 的 `source_policy` / `constant_kind` / 范围 / `parameterizable` / `label` |
| `typed_service._validate_fixed_constant_arguments` | 画布与 compose 的常量性校验 |
| `series_service._validate_fixed_series_configuration` | 时序计划编译前的常量性校验 |
| `series_parameters.inspect/validate/bind` | 候选识别与参数 schema |

### 3.5 不可派生、必须显式声明的三类

以下不是白名单，是算子自身的数学语义，必须声明一次，但应声明在算子 spec 上而不是散在四处：

1. **跨参数关系**：`min_periods ≤ window`、`ddof < window`、`clip.lower ≤ clip.upper`。
   今天硬编码在 `series_parameters.validate_parameter_relations()`（含一段针对 `std(rolling_window(...), ddof)`
   的特判）。改为 `TypedOperatorSpec.relations: tuple[str, ...]`，由一个通用比较求值器执行，前后端共用同一份。
2. **回看推导**：`rolling_* → lookback + window - 1`、`lag/difference → + periods`。
   已在 `_infer_series_history()` 里按算子分支实现，且**已经读运行时参数值**（`_parameter_constants`），
   本设计不改。
3. **值域证明**：`clip → [lower, upper]`、`recursive_smooth → ∪ initial`。
   已在 `_infer_series_value_ranges()` 里实现，同样已读运行时值。放开 `initial`/`lower`/`upper` 无需额外工作。

`ddof` 与 `probability` 既不影响回看也不影响已建模的值域（`rolling_std` 值域本来就是 `not_proven`），放开无风险。

### 3.6 删除清单（已完成）

| 删除 | 替代 |
|---|---|
| `series_parameters.PARAMETER_CAPABILITIES`（17 条） | 契约标签 + C2 |
| `typed_service._FIXED_CONSTANT_OPERATOR_PARAMETERS`（死代码） | 直接删 |
| `typed_service._FIXED_CONSTANT_PARAMETER_POLICIES`（24 条） | 契约标签 |
| `typed_service.PARAMETER_LABELS`（含 2 个被静默覆盖的重复键） | `parameter_policy.ARGUMENT_LABELS` |
| `series_service._SERIES_FIXED_ARGUMENTS`（25 条） | 契约标签 |
| `series_parameters.validate_parameter_relations` 的算子名集合与 `std(rolling_window(...))` 特判 | 按入参名成对比较，任何算子只要同时有这对入参就自动继承 |
| `IndicatorStudio.tsx` 的 `['ddof','periods','probability']` | `source_policy === 'fixed_constant'` |
| `IndicatorStudio.tsx` 的 `OPERATOR_PARAMETER_LABELS` 与 `parameterLabel` 内联表 | `/meta` 的 `label` |
| `indicatorGraphPresentation.ts` 的 `PARAMETER_LABELS` 与 `（ddof）` 去括号正则 | `/meta` 的 `label` |

后端净减约 220 行。

### 3.7 交互：系统全量识别，用户自己决定

候选面板列出**所有**满足 C1+C2 的位置，包含 `ddof`、`probability`、`initial`、`default`。
对会改变统计口径的参数（`ddof`、`probability`）显示一行提示，**不拦截**：

```
可调输入                        当前值    操作
MA / 滚动平均 / 窗口期数           20     [开放为参数]
VOL / 滚动标准差 / 自由度修正        0     [开放为参数]
       ⚠ 改动会改变统计量口径（样本/总体标准差），不只是时间尺度
```

---

## 4. 需求二：标量指标支持可变参数

### 4.1 链路盘点

| 环节 | 时序（今天） | 标量（今天） | 需要做什么 |
|---|---|---|---|
| 候选识别 | `inspect_parameter_inputs` | 直接抛「仅适用于时序指标」 | 解除拒绝 + 合成通道 |
| 绑定改写 | `bind_parameter_input` → `series_outputs[].expression` | 同上被拒 | 写回 `definition["expression"]` |
| 类型注入 | `parameter_variable_types()` 合入 `variable_types` | `_compile_typed_plan` 只用 `variable_types(ctx, dsl)` | 合入 + 改缓存键 |
| 常量性校验 | `_validate_fixed_series_configuration` | 无 | 复用同一函数（提到共享模块） |
| 单根执行 | `compile_numba_series_plan` | `compile_numba_plan` | **无需改动**，见 §4.3 |
| 批量执行 | 不适用 | `compile_numba_batch_plan` | **唯一真改造**，见 §4.4 |
| 运行时注入 | `_build_series_runtime_context` | `_evaluate_runtime` 的 context | 注入 parameters |
| 结果缓存 | `_series_cache_key` 含 `parameter_hash` | `_definition_cache_key` **不含** | 加 `parameter_hash` |
| API | `SeriesIndicatorInstance.parameters` | 5 个请求模型没有 | 补字段 |

### 4.2 定义侧

`series_parameters._trees()` 对标量产出一个合成通道：

```python
if definition.get("result_kind") == "time_series":
    outputs = definition.get("series_outputs") or []
else:
    outputs = [{"id": "value", "label": definition.get("name") or "结果",
                "expression": definition.get("expression") or ""}]
```

`inspect_parameter_inputs` / `bind_parameter_input` 去掉 `result_kind != "time_series"` 的抛错；
`bind_parameter_input` 回写时按 `result_kind` 决定落到 `series_outputs[].expression` 还是 `expression`。
候选 id 仍是 `f"{output_id}:{ordinal}:{name}:{tree_hash}"`，标量的 `output_id` 固定为 `value`，
陈旧检测（`STALE_PARAMETER_CANDIDATE`）行为不变。

`IndicatorDraft.parameter_schema` 本来就与 `result_kind` 无关，Pydantic 侧无需改。

### 4.3 单根标量执行：零改动

这是本需求最省事的一段。`compile_numba_plan()` 生成的是：

```python
def generated_plan(v0, v1, ..., vn):   # 每个 context_requirement 一个形参
```

参数一旦成为 `ValueType.scalar()` 的 context 变量，就自动变成其中一个 `float64` 形参，
**一个编译签名服务所有取值**，与时序侧完全同构。需要做的只有三件小事：

1. `_compile_typed_plan()` 的 `variable_types` 合入 `parameter_variable_types(definition)`；
   `@lru_cache` 与 `_WARMED_TYPED_PLANS` 的键追加参数 schema 指纹。
2. `_evaluate_runtime()` 构造 context 时 `context.update(parameters)`。
3. `_physical_dependency_signature()` 已经会跳过未注册变量（`get_variable()` 返回 `None`），
   参数**不会**污染物理数据列。补一条测试锁住这个行为，防止以后有人改成「未知即物理列」。

### 4.4 批量标量执行：加参数向量（唯一真改造）

今天的阻塞点有两个：

```python
# typed_numba_plan.py:741
def _batch_variable_expression(name, *, column_index, definition) -> str:
    ...
    annual, per_observation = _risk_free_scalars(definition)
    if name == "annual_risk_free_rate_decimal":
        return repr(annual)              # ← 标量被 repr() 烘进生成源码
    ...
    raise ValueError(f"batch physical variable unavailable: {name}")   # ← 参数名撞这里

# typed_numba_plan.py:783
"risk_free": [d.get("annual_risk_free_rate_percent", 0.0) for d in definitions]  # ← 值进 plan id
```

照抄 risk_free 的做法 = **每换一个参数取值就要重新编译一个批量计划**，违反 AGENTS.md 的预热约束
（`request_time_compilation=0`，缓存未命中必须 fail-closed）。

改为给批量核加一个参数向量形参：

```python
generated_batch(values, starts, ends, elapsed_days, output, statuses, enabled, params)
generated_row (values, start,  end,  elapsed_days, out_row, status_row, enabled, params)
#                                                                                ^^^^^^ float64[::1]
```

- `_batch_variable_expression` 对参数名返回 `f"params[{slot}]"`，槽位由「批次内 `(metric_index, parameter_id)`
  的稳定排序」决定。
- `_batch_plan_id` 里 **`risk_free` 的取值换成槽位布局**（`[(metric_index, name)]` 列表）。
  值不再进 plan id，一次预热服务所有取值。
- `CompiledNumbaBatchPlan.compute()` 多收一个 `params` 数组；`SelectedNumbaBatchPlan` 按 `indices` 取子集时
  同步裁剪槽位。

**CSE 安全性**：`shared_batch_graph.merge_metric_plans()` 的节点合并键里，变量节点用的就是
`variable_expression()` 的返回串。参数返回 `params[7]` 这类含槽位的串，两个指标即便当前取值相同也拿到不同槽位，
因而**不会被错误合并**。这比今天常量走 `float(...).hex()` 合并更保守，方向安全，只是少省一点节点。

**顺带修掉同类问题**：把 `annual_risk_free_rate_decimal` 等 5 个 risk-free 标量一并搬进 `params`。
今天每个不同的年化无风险利率都会生成一个独立批量计划——这是同一个病。签名只想改一次，建议同期做。

### 4.5 缓存与预热

- `_definition_cache_key()`（`service.py:2808`）追加 `"parameters": parameter_hash(parameters)`。
- 请求级 `request_cache_key`（`service.py:4130`）已按 definition key 聚合，随之生效。
- `_warm_single_product_definition()` 不变：因为参数值不进 plan id，**每个指标仍只需预热一次**，
  仍然是「单根 + 单例批量」两条都必须暖。

### 4.6 API 契约

| 模型 | 现状 | 动作 |
|---|---|---|
| `SeriesIndicatorInstance` | 有 `parameters` | — |
| `ExportExcelRequest` | 有 `parameters` | — |
| `IndicatorReference` | 无 | 加 `parameters: dict[str, StrictFloat]` |
| `EvaluateRequest` | 无 | 通过 `indicator_refs[].parameters` 承载 |
| `SnapshotIndicatorItem` | 无 | 加，快照必须记录取值才可复现 |
| `PlanIndicatorInput` | 无 | 加，但**保存即锁定**，见 D2 |

`/api/custom-indicators/parameters/inspect` 与 `/bind` 两个端点签名不变，只是不再拒绝 `result_kind == "scalar"`。

---

## 5. 分期

| 期 | 内容 | 状态 |
|---|---|---|
| **0–2** | 契约标签文法；`parameter_policy.py`；删四张表与三处前端副本；`relations` 改为按入参名配对；一致性测试 | **已完成（需求一）** |
| **3** | 标量定义侧：`_trees` 合成通道、`bind` 回写、类型注入、常量性校验复用 | **已完成（需求二）** |
| **4** | 标量执行侧：`_evaluate_runtime` 注入、`_definition_cache_key` 加 hash、API 模型补字段 | **已完成（需求二）** |
| **5** | 批量参数向量：改 `generated_batch`/`generated_row` 签名、`_batch_plan_id` 换槽位布局、risk_free 一并搬入 | **已完成（需求二）** |

### 5.1 实现与设计稿的差异

1. **编译器里还有一张硬编码表**（设计稿漏盘的）。`typed_dsl.py` 的 `probability_index`
   与 `integer_parameters` 两张字典，把 11 个算子的参数位置、整数性和上下界又写了一遍，
   并且只对 `quantile/quantile_where` 之外的位置开放运行时参数——所以标量参数在编译阶段就会被挡住。
   已整段删除，改为 `configuration_arguments(spec, arity)` 驱动。连带：
   - `_literal_number` 补上一元正负号折叠，`clip(x, -0.5, 0.5)` 这类负字面量才算常量；
   - `variance/std` 的 `ddof` 单独判分支删除，由 `scalar<count:0..>` 承担；
   - `rolling_apply` 只保留 `min_periods ≤ window` 这条跨参数关系（`> 5000` 由契约上限接管）；
   - `lag.periods` 签名放宽为 `scalar<count:0..20000>`，保持原来允许 `lag(x, 0)` 的行为；
   - `probability` 标签增加 `exclusive` 标志，`(0,1)` 开区间由标签表声明一次，
     编译门禁与 `_probability` 都读它；运行时 schema 用 `exclusive_minimum/exclusive_maximum` 明确表达开区间，步长与数学边界分离，允许 `0.995` 和 `0.005`。
2. **「已声明标量参数」的放行条件收紧**。原逻辑是「任何 count/dimensionless 的标量上下文变量」，
   于是 `rolling_mean(returns, periods_per_year)` 这类用数据标量当窗口的写法能通过 AST 门禁
   （这是改动前就存在的漏洞）。现在编译器接收 `parameter_names`（唯一来源是 `parameter_schema`），
   只有真正开放的参数才能占配置位；诊断码统一为 `SERIES_CONFIGURATION_MUST_BE_CONSTANT`，
   与计划级 `configuration_violations` 同一句话、同一个码。
3. **常量性校验提到 `parameter_policy.configuration_violations`**，返回诊断而不抛错，
   `cal_indicators` 不必依赖 `custom_indicators`。时序侧和标量侧各自包一层
   `raise_on_configuration_violations`。
4. **`SnapshotIndicatorItem.parameters` 需要参与列名与去重键**：同一指标在不同取值下是两列，
   取值在配置保存时就解析成完整映射，快照作业按 `(period, parameter_hash)` 分批。
5. **`EvaluateRequest.parameters`**（设计稿没列）：未保存草稿只能走 `inline_definition`，
   没有 `indicator_refs` 可挂取值，加一个请求级字段承载，指标中心预览才能改参数看结果。
6. **批量槽位布局**：每个指标固定 2 个 risk-free 槽 + 其参数 ID 排序，
   布局只取决于定义形状、与取值无关，因此单指标计划 id 在融合批次里仍可被子集复用。
7. **`SHARED_GRAPH_VERSION` → `independent-metrics-cse-3`**，融合核签名多了 `params`，
   派生编译产物在启动预热时按新签名重建。`PLAN_COMPILE_CONTRACT_MARKER` 仅记录编译契约；
   不清空评价方案、历史版本或运行结果。
8. **已开放参数的标量指标不支持派生滚动时序指标**，`rolling_series_compatibility` 明确返回
   `ROLLING_SOURCE_PARAMETERS_UNSUPPORTED`，而不是抛一个「未知变量 probability_1」。

## 6. 决策点

- **~~D1 `power.exponent`~~**：已在实现中定案，见 §3.2 —— 签名改为 `scalar<dimensionless>`，
  不纳入配置输入。选择与原推荐相反，理由是收紧为常量会破坏画布里合法的标量表达式指数。
- **~~D2 评估方案里的参数~~**：已按推荐锁定。`PlanIndicatorInput.parameters` 在
  `_normalize_plan` 解析成完整映射并随方案修订保存，运行时只回放、不接受覆盖。
- **~~D3 第 5 期做不做~~**：已做。取值进 `params` 向量而不进 plan id，因此
  `quantile(returns, probability_1)` 在 0.5 与 0.9 两次运行里命中**同一个** `compiled_plan_id`；
  risk-free 的 5 个标量一并搬入，原来每个年化利率各编译一份批量计划的问题同时消失。

## 7. 不做的事

- 不引入参数搜索/优化器。参数在一次计算内是固定标量。
- 不自动合并数值相同的不同常量；共享参数仍必须用户显式选择。
- 不改写已有指标历史版本；`parameter_contract_version` 不匹配的旧定义继续按保存时的固定公式执行。
- 不放开数据字段、复权口径、日期轴与输出结构。C2 的字面量门禁就是这条边界的机械保证。


## 7. 审核修复（2026-09-14）

- 子集缓存按请求中每种公式的出现次数匹配预热根，逐次消费不同槽位；不能用集合包含或 `index()` 复用首个根。参数值仍不进入编译 ID。
- 时序快照分桶键包含 `(period, indicator_id, revision, parameter_hash)`，求值请求传入完整参数；只有同一参数实例的多通道共享一次计算。
- DSL 保持 `2.4.0`，新增注册表契约 `2.4.1` 作为新建定义默认值。注册表 `2.0.0` 至 `2.4.0` 中原先的普通 scalar 输入保持表达式能力；旧版签名适配委托唯一当前数值实现。旧版内置指标显式锁定 `2.4.0`，不因全局默认值升级而改写定义。
- 常量表达式（如 `-1 / 2`）由统一安全语法折叠器判定。新版配置位拒绝数据表达式；历史标量 `clip` 上下界表达式仍能编译。
- 参数化标量插入另一公式时，将来源锁定 revision 的默认参数固化为常量，返回 `indicator_origin.parameters`；不污染来源定义，也不把同名参数混入目标草稿。以后需要开放时可重新识别对应常量。
- 概率 schema 传递显式开边界，API、持久化、画布和前端校验保持一致。作者的步长约束继续有效；开放 `0.995` 自动选择 `0.001`，不会把数学有效范围裁成 `[0.01, 0.99]`。
- 启动升级只记录编译契约，不删除业务方案及其历史；预热仍基于锁定指标和参数布局重建执行产物，失败时不进入正式计算。

本次没有新增算子或数值算法。改动属于签名元数据、参数绑定、缓存根选择与编排；数值仍由已有固定签名 NJIT 内核执行。配置表达式折叠只处理小规模定义元数据。

## 8. 验收记录（2026-09-14）

远端 `origin/Dev` 已快进合并至本地 `Dev`：`d4b881e4482087e0efbc4ef828d5483ad04d8b48`。原有未提交内容保留，路由文档完成三方合并。未创建提交或推送。

| 验证 | 结果 |
| --- | --- |
| 参数、编译器、批量 NJIT、目录、内置时序、画布及公式往返 | 244 个不同用例通过，含修复后的定向复跑 |
| 其他相关后端模块 | 504 个不同用例完成；初跑 503 通过、1 个组合响应兼容断言失败，修复后与参数测试一并复跑 34/34 通过 |
| 历史快照测试 | 原夹具配置与期望值读取版本不一致；统一锁定 revision 1 后独立复跑 4/4 通过 |
| 前端单元测试 | 全量 135 文件、1007 用例通过；最后的参数界面改动定向复跑 55/55 通过 |
| 浏览器 | 320、768、1440 三个宽度，标量/时序共 6/6 通过；使用隔离数据与真实 API/NJIT，覆盖开放、保存、覆盖、端点拒绝、恢复默认和 Excel 下载 |
| 页面可读性 | 新概率流程三个宽度均通过文字对比度审计、无横向溢出、输入框宽度检查；已检查桌面和手机截图 |
| 工程检查 | TypeScript、生产构建、静态设计检查、diff 空白检查、AI Hermes validate/evolve 均通过 |

修复附带的前端问题：参数变更/恢复默认会清除旧标量结果；参数网格按面板可用宽度排列；预览产品计数徽标使用足够对比度的实色。未新增视觉依赖。

这是相关模块的分批回归与失败修正后复跑记录，不代表后端所有测试或正式环境启动验收。测试使用临时目录和固定本地夹具，未修改正式数据。构建仍存在已有的大包体积提示；本次没有宣称额外性能提升。

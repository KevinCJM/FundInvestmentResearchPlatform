# 产品研究页 · 指标面板重构与标量可变参数详细设计

适用范围：`/product-research/products/:productId`（`frontend/src/pages/ProductDetail.tsx`）的「走势与指标」页签。
依赖契约：`docs/indicator_runtime_parameters_design.md`、`docs/indicator_parameter_contract_design.md`。
设计准则：`docs/frontend-design-guidelines.md`（工作台条款）。

## 1. 现状与问题

「走势与指标」页签里，指标被拆在三个互不相干的位置，各有一套交互：

| 位置 | 选择方式 | 区间控制 | 参数 | 结果形态 |
| --- | --- | --- | --- | --- |
| 技术辅助线设置（图表内 `<details>`） | 开关卡片 | 跟随主图 | 硬编码控件 | 叠加在主图上 |
| 自定义研究指标 | `MetricSelector` 多选弹层 | 每卡一个下拉 | **完全没有** | 数值卡栅格 |
| 时序指标研究（`TimeSeriesIndicatorPanel`） | 单选下拉，一次只看一个 | 面板级下拉 | `<details>` 折叠 | 折线图 |

具体缺陷：

1. **标量指标不支持可变参数。** 指标中心已经可以把 `quantile(returns, probability_1)` 存成带参数的标量指标，但产品研究页只发 `indicator_ids`，后端按锁定默认值算，页面上没有任何提示。用户以为自己在看 0.995 分位，实际看的是 0.9。
2. **同一个选择器会选出打不开的指标。** `researchIndicators` 用 `indicatorsForContext(items, 'single_product')` 过滤，**没有排除时序指标**。选中一个时序指标，`evaluate` 抛 `INDICATOR_RESULT_KIND_MISMATCH`，同一区间分组里的全部标量结果一起消失，页面只显示"该指标当前无法计算"。
3. **两套选择模型。** 上面多选弹层、下面单选下拉；时序指标一次只能看一个。
4. **三层卡片嵌套。** `section(rounded-xl border) > div(rounded-xl border-accent-100) > article(rounded-xl border)`，违反准则 6.5「`VISUAL_DENSITY: 7` 下卡片套卡片是禁止的」。
5. **参数藏在 `<details>` 里**，摘要是 `窗口期数=20 · 概率=0.9` 这样的连写串；打开前看不出这个数是用什么参数算的。
6. **后端 `prepare_evaluation` 丢参数。**（见 §2.1）

## 2. 后端设计

### 2.1 唯一需要改的地方：`prepare_evaluation` 回传参数

标量参数链路本身已完整：`IndicatorReference.parameters` → `evaluate(..., parameters_by_indicator=...)` → `_resolve_evaluation_definitions` 内 `normalize_series_parameters` 解析并校验 → 结果带 `parameters` 与 `parameter_hash`；由 `test_scalar_parameter_values_run_on_one_prewarmed_batch_plan` 覆盖。

但前端 `evaluateCustomIndicators` 在指标数 > 1 时先调 `/prepare`，再**用 `prepared.indicator_refs` 替换请求里的 refs**（为了防止准备与执行之间目录被改版本）。而 `prepare_evaluation` 返回的 refs 只有 `indicator_id` 和 `indicator_revision`，参数被吃掉。实测：

```
prepared refs -> [{'indicator_id': ...}, {'indicator_id': ...}]          # 参数没了
values with prepared refs  -> [{'probability_1': 0.9}, {'probability_1': 0.9}]   # 两个都回落默认值
values with original refs  -> [{'probability_1': 0.25}, {'probability_1': 0.75}] # 正确
```

参数不进编译计划标识（这正是批量参数向量的设计目的），所以准备阶段不需要知道参数值，只需要**原样回传**调用方给的那一份：

```python
supplied = {str(item["indicator_id"]): dict(item.get("parameters") or {}) for item in references}
return {"prepared": True, "plans": audits, "indicator_refs": [
    {"indicator_id": item["id"], "indicator_revision": item["revision"],
     **({"parameters": supplied[item["id"]]} if supplied.get(item["id"]) else {})}
    for item in definitions if item.get("id")]}
```

回归：`test_prepare_evaluation_returns_the_parameters_it_was_given`。

### 2.2 不改的地方及理由

- **不放开 `DUPLICATE_INDICATOR`。** 一次请求里同一指标只能出现一次，所以本页每个指标只有一组参数。要同时比较同一指标的两组取值，应在指标中心另存一个版本，或使用产品对比页；本次不扩大范围。
- **不加新接口。** `presentation.parameter_schema` 已随每条结果下发，前端不需要再查目录。
- **快照命中键已含 `parameter_hash`**，参数不同不会串用快照值，无需改动。

## 3. 前端设计

### 3.1 信息结构：一个区块、一份清单

删除 `自定义研究指标` 与 `时序指标研究` 两个并列区块，合并为单个 **「研究指标」** 区块。用户的心智模型收敛成一句话：**选指标 → 每张卡上调区间和参数 → 读结果**。

> 后续变更（`docs/product_trend_chart_design.md`）：时序指标已迁到走势图叠加，本区块随后只保留标量指标，下面的「时序指标」分组不再存在。

```
研究指标                                          [在指标中心分析]
用指标中心已保存的版本计算。标量指标给出区间内的一个数，时序指标给出整条曲线。
─────────────────────────────────────────────────────────────────
[选择研究指标 · 已选 3/8 ▾]        截止日 [______]  未填则跟随平台研究日 2026-09-14
─────────────────────────────────────────────────────────────────
标量指标 · 2 项
┌──────────────────────┐ ┌──────────────────────┐
│ 风险 · 内置 v3        │ │ 收益 · 工作区 v2      │
│ 年化波动率      [移除] │ │ 分位收益        [移除] │
│ 12.34%               │ │ 0.85%                │
│ 1Y · 起止 · 250 个观察值│ │ ALL · 起止 · 1250 个   │
│ 区间 [1Y ▾]           │ │ 区间 [ALL ▾]          │
│ 概率 [0.995]  [应用]   │ │ 本版本无可调参数       │
│ 查看定义与口径         │ │ 查看定义与口径         │
└──────────────────────┘ └──────────────────────┘
─────────────────────────────────────────────────────────────────
时序指标 · 1 项
┌────────────────────────────────────────────────────────────────┐
│ 技术 · 工作区 v3   可调均线                             [移除]   │
│ 区间 [1Y ▾]   窗口期数 [60]  1–500 · 默认 20 · 步长 1  [应用][恢复]│
│ 实际计算参数：窗口期数=60 · 2025-09-14 – 2026-09-12              │
│ [────────────── ECharts 折线 ──────────────]                    │
└────────────────────────────────────────────────────────────────┘
```

要点：

1. **一个选择器管两类指标**，按 `result_kind` 自动分组渲染。`MetricSelector` 不再能选出打不开的指标。
2. **层级降到两层**：区块（唯一卡片外壳）→ 指标卡。中间那层 `border-accent-100` 包裹删除，内部分组用 `divide-y` + 小标题，符合准则 6.5。
3. **每张卡自包含**：名称 → 数值 → 口径 → 控件（区间、参数）→ 定义入口。两类卡片的控件行顺序一致。
4. **参数常驻可见**，不再折叠。准则 8.6：label 在输入框上方，帮助文字在 DOM 里，错误在下方。
5. **空态说明下一步**（准则 3.5）：未选指标时给出「打开选择器」和「去指标中心」两条路径，局部面板不带吉祥物。

### 3.2 标量指标的运行参数

数据流：

```
MetricSelector 选中
  → ProductDetail 的 parametersByIndicator[indicatorId]
  → groupIndicatorsByPeriod 按区间分组
  → evaluateCustomIndicators({ indicator_refs: [{indicator_id, indicator_revision, parameters}], ... })
  → (>1 指标时) /prepare 回传含参数的 refs → /evaluate
  → result.parameters 回显到卡片
```

改动点：

- **请求从 `indicator_ids` 改为 `indicator_refs`。** 顺带把版本钉住：现在发裸 id，准备与执行之间目录被改会静默换算法。
- `parametersByIndicator` 只放组件 state，**不持久化到 localStorage**。区间和选中项是"我常看这些"，参数是"我现在想问这个"；持久化会让下次进页面看到一个自己不记得设过的窗口期。
- 参数编辑沿用「编辑不计算、点应用才重算」，与时序面板一致。只提交与默认值不同的键，空覆盖表示恢复默认。
- 结果卡展示的是 **`result.parameters`（后端解析后的实际取值）**，不是输入框里的草稿值，避免"改了没应用"被误读成已生效。

### 3.3 组件改动

| 文件 | 改动 |
| --- | --- |
| `components/indicator-parameters/IndicatorParameterInputs.tsx` | 去掉 `<details>` 外壳和 `border-accent-200 bg-accent-50/40` 嵌套卡片，改为平铺栅格；新增可选 `effective` 行显示后端实际取值 |
| `components/indicator-parameters/SeriesIndicatorCard.tsx` | 由 `TimeSeriesIndicatorPanel.tsx` 改名而来。保留原 `SeriesCalculation` 的请求与竞态处理，外壳改为与标量卡一致的指标卡，新增 `period` / `onPeriodChange` / `onRemove`。**删除**原先的单选下拉包装组件 |
| `components/metrics/MetricDisplay.tsx` | `MetricResultCard` 新增 `parameters` / `onParametersChange`；参数 schema 取自 `presentation.parameter_schema`；状态非 `ok` 时显示状态徽章 |
| `pages/ProductDetail.tsx` | 合并两个区块；按 `result_kind` 分流；请求改 `indicator_refs` 并带参数 |
| `services/customIndicators.ts` | `IndicatorReference.parameters?`；`EvaluationResult.parameters?` / `parameter_hash?` |
| `locales/system.json` | 新增区块与空态文案键 |

### 3.4 遵循的准则条款

- 3.1：只用 `accent` / `slate` / `emerald` / `amber` / `rose`；焦点环统一 `focus-visible:ring-2 focus-visible:ring-accent-500`。
- 3.3：最小字号 `text-xs`（12px）。3.4：卡片 `rounded-xl`，控件 `rounded-lg`，阴影只用 `shadow-sm`。
- 5.5：按钮 `min-h-10`，输入框 `min-h-11`。
- 6.5：分组用 `divide-y`，不再加第三层卡片。6.9：响应式坍缩逐区块显式声明（`md:grid-cols-2 xl:grid-cols-3`）。
- 8：四态齐备 —— 加载（`aria-live` 文本）、空态（说明下一步）、错误（就近 `role="alert"`）、禁用（参数区在计算中禁用并说明）。

## 4. 验证

- 后端：`test_series_runtime_parameters.py`（含新增 prepare 回传用例）、`test_custom_indicator_routes.py`。
- 前端单测：`IndicatorParameters.test.tsx`、`MetricDisplay.test.tsx`、`ProductDetail.test.tsx`。
- 端到端：`metric-display.spec.ts`、`contrast.spec.ts`。
- 设计检查：`npm run design:check --prefix frontend`；文案：`node scripts/check_i18n.mjs`。

## 5. 明确不做

- 不做同一指标多组参数并排对比（受 `DUPLICATE_INDICATOR` 约束，见 §2.2）。
- ~~不动主图的「技术辅助线设置」~~：已由 `docs/product_trend_chart_design.md` 接手。时序指标改为画在走势图上，本区块只留标量指标。
- 不做参数预设的保存/命名。要复用一组取值，应在指标中心另存版本——那里才有版本和审计。
